#!/bin/bash
# Build ROCm version
set -e

echo "Building ROCm GPU library..."
mkdir -p gpu_lib
gcc -shared -fPIC -o gpu_lib/libgpu_crop_rocm.so gpu_lib/gpu_crop_rocm.c

echo "Building ROCm binary..."
CGO_ENABLED=1 go build -tags rocm -o bin/pdf-processor-rocm ./cmd/pdf-processor-rocm

echo "Done: bin/pdf-processor-rocm"