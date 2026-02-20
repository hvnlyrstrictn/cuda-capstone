#!/bin/bash
echo "Building the CUDA Image Processor..."
make build

echo "Running the batch pipeline..."
./image_processor.exe -i ./input_images/misc -o ./output_images

echo "Done!"