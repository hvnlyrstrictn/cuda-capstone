#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void bgrToGrayscaleKernel(unsigned char* d_in, unsigned char* d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int colorOffset = (y * width + x) * 3;
        int grayOffset = y * width + x;

        unsigned char b = d_in[colorOffset];
        unsigned char g = d_in[colorOffset + 1];
        unsigned char r = d_in[colorOffset + 2];

        d_out[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

__global__ void gaussianBlurKernel(unsigned char* d_in, unsigned char* d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float kernel[3][3] = {
        {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
        {2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},
        {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}
    };

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float blurValue = 0.0f;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int pixelVal = d_in[(y + ky) * width + (x + kx)];
                blurValue += pixelVal * kernel[ky + 1][kx + 1];
            }
        }
        d_out[y * width + x] = (unsigned char)blurValue;
    }
}

int main(int argc, char** argv) {
    std::string inputDir = "";
    std::string outputDir = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) inputDir = argv[++i];
        else if (arg == "-o" && i + 1 < argc) outputDir = argv[++i];
    }

    if (inputDir.empty() || outputDir.empty()) return 1;

    std::vector<cv::String> image_paths;
    cv::glob(inputDir + "/*.tiff", image_paths, false);

    for (const auto& img_path : image_paths) {
        cv::Mat h_img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (h_img.empty()) continue;

        size_t last_slash = img_path.find_last_of("\\/");
        std::string filename = img_path.substr(last_slash + 1);
        
        int width = h_img.cols;
        int height = h_img.rows;
        size_t color_bytes = width * height * 3 * sizeof(unsigned char);
        size_t gray_bytes = width * height * 1 * sizeof(unsigned char);

        unsigned char *d_color, *d_gray, *d_blur;
        cudaMalloc(&d_color, color_bytes);
        cudaMalloc(&d_gray, gray_bytes);
        cudaMalloc(&d_blur, gray_bytes);

        cudaMemcpy(d_color, h_img.ptr(), color_bytes, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        bgrToGrayscaleKernel<<<grid, block>>>(d_color, d_gray, width, height);
        cudaDeviceSynchronize();

        gaussianBlurKernel<<<grid, block>>>(d_gray, d_blur, width, height);
        cudaDeviceSynchronize();

        // Retrieve AND SAVE the intermediate Grayscale result
        cv::Mat h_gray(height, width, CV_8UC1);
        cudaMemcpy(h_gray.ptr(), d_gray, gray_bytes, cudaMemcpyDeviceToHost);
        cv::imwrite(outputDir + "/gray_" + filename, h_gray);

        // Retrieve AND SAVE the final Blurred result
        cv::Mat h_blur(height, width, CV_8UC1);
        cudaMemcpy(h_blur.ptr(), d_blur, gray_bytes, cudaMemcpyDeviceToHost);
        cv::imwrite(outputDir + "/blur_" + filename, h_blur);

        cudaFree(d_color);
        cudaFree(d_gray);
        cudaFree(d_blur);
    }
    std::cout << "Batch processing complete with intermediate saves!\n";
    return 0;
}
