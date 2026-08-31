#!/bin/bash
# Build CUDA version
set -e

echo "Building CUDA GPU library..."
mkdir -p gpu_lib
gcc -shared -fPIC -o gpu_lib/libgpu_crop.so gpu_lib/gpu_crop.cu

echo "Building CUDA binary..."
CGO_ENABLED=1 go build -tags cuda -o bin/pdf-processor-cuda ./cmd/pdf-processor-cuda

echo "Done: bin/pdf-processor-cuda"