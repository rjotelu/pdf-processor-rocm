# pdf-processor-rocm

GPU-accelerated PDF to cropped Markdown converter using **AMD ROCm**.

## Build

```bash
# Prerequisites
# - ROCm 5.x+
# - poppler-utils, tesseract-ocr

# Build
./build-rocm.sh
```

## Binary

```
bin/pdf-processor-rocm   # Main ROCm binary
```

## Usage

```bash
./bin/pdf-processor-rocm -dir /path/to/pdfs -workers 8
```

## Architecture

- `cmd/pdf-processor-rocm/` - Main ROCm binary (CGO with ROCm bindings)
- `cmd/pdf-processor-cpu/` - CPU-only version (fallback)
- `gpu_lib/libgpu_crop_rocm.so` - Stub GPU library for CPU fallback

## Real ROCm Implementation

Replace `gpu_lib/gpu_crop_rocm.c` with actual HIP kernels:

```hip
// Example kernel
extern "C" __global__ void crop_kernel(...) {
    // HIP GPU implementation
}
```

Build with `hipcc`:
```bash
hipcc -shared -fPIC gpu_kernels.hs -o gpu_lib/libgpu_crop_rocm.so
```