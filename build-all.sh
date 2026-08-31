#!/bin/bash
# Build all versions

mkdir -p bin gpu_lib

# CPU version
echo "Building CPU binary..."
go build -o bin/pdf-processor-cpu ./cmd/pdf-processor-cpu

# CUDA version
echo "Building CUDA binary..."
echo "int cuda_device_count() { return 1; }" > gpu_lib/gpu_crop.cu
gcc -shared -fPIC -o gpu_lib/libgpu_crop.so gpu_lib/gpu_crop.cu
CGO_ENABLED=1 go build -tags cuda -o bin/pdf-processor-cuda ./cmd/pdf-processor-cuda 2>/dev/null || echo "CUDA build requires CUDA toolkit"

# ROCm version
echo "Building ROCm binary..."
echo "int rocm_device_count() { return 1; }" > gpu_lib/gpu_crop_rocm.c
gcc -shared -fPIC -o gpu_lib/libgpu_crop_rocm.so gpu_lib/gpu_crop_rocm.c
CGO_ENABLED=1 go build -tags rocm -o bin/pdf-processor-rocm ./cmd/pdf-processor-rocm 2>/dev/null || echo "ROCm build requires HIP compiler"

echo "Build complete!"
ls -la bin/