CXX = nvcc
CXXFLAGS = -std=c++17 -Wno-deprecated-gpu-targets
# We will link OpenCV for easy image loading/saving
OPENCV_FLAGS = `pkg-config opencv4 --cflags --libs`

all: clean build

build: main.cu
	$(CXX) main.cu $(CXXFLAGS) $(OPENCV_FLAGS) -o image_processor.exe

run:
	./image_processor.exe $(ARGS)

clean:
	rm -f image_processor.exe
