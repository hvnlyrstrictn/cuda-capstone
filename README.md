# CUDA Batch Image Processor: Grayscale & Gaussian Blur Pipeline

## 📌 Project Purpose
The goal of this Capstone project is to demonstrate the massive performance benefits of GPU-accelerated parallel computing for image processing tasks. Instead of sequentially looping through pixels on a CPU, this project utilizes a custom CUDA C++ pipeline to process entire batches of high-resolution images (up to 1024x1024 pixels) simultaneously on an NVIDIA GPU.

## ⚙️ Algorithms & Kernels
This pipeline applies two sequential custom CUDA kernels to each image:
1. **Grayscale Kernel:** Maps the 2D image coordinates to a grid of GPU threads. Each thread independently applies the standard luminance formula (`Y = 0.299R + 0.587G + 0.114B`) to its assigned pixel.
2. **Gaussian Blur Kernel:** A 3x3 convolution matrix is applied to the grayscale output. Each GPU thread reads its pixel and the 8 surrounding neighbor pixels, calculates the weighted average, and writes the softened pixel to a new device array.

## 🧠 Lessons Learned
* **Memory Management is Critical:** Allocating memory (`cudaMalloc`) and transferring data across the PCIe bus (`cudaMemcpy`) is an expensive operation. This project reinforced the importance of minimizing host-to-device transfers. Instead of copying the intermediate grayscale image back to the CPU, it remains on the GPU to immediately feed the Gaussian Blur kernel.
* **Thread Mapping:** Translating 2D image coordinates into 1D array indexes using `blockIdx`, `blockDim`, and `threadIdx` was initially challenging but became highly intuitive once the grid/block hierarchy was understood.
* **Boundary Conditions:** When applying the 3x3 blur, I had to implement boundary checks (`x > 0 && x < width - 1`) to ensure GPU threads didn't attempt to access memory outside the image array.

## 🚀 How to Compile and Run
This project includes a Command Line Interface (CLI) that accepts custom arguments for the input and output directories. 

**Prerequisites:** * NVIDIA GPU with CUDA Toolkit installed
* OpenCV 4 (C++ development libraries)

**Build the Project:**
make build

**Execute the Project:**
You can run the executable directly via the CLI:
./image_processor.exe -i <path_to_input_dir> -o <path_to_output_dir>

Alternatively, you can use the provided run script:
bash run.sh

## 📊 Proof of Execution
Please see the `proof_of_execution` folder in this repository for sample outputs showing the step-by-step transformation (Original -> Grayscale -> Blurred) of classic 512x512 and 1024x1024 test images from the USC-SIPI Image Database.
