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
# Basic usage
./bin/pdf-processor-rocm -dir /path/to/pdfs -workers 8

# Concurrent PDF processing
./bin/pdf-processor-rocm -dir /path/to/pdfs -pdf-workers 4 -workers 8

# Resume interrupted processing
./bin/pdf-processor-rocm -dir /path/to/pdfs -checkpoint progress.json

# Pretty text formatting
./bin/pdf-processor-rocm -dir /path/to/pdfs -pretty

# Skip empty pages
./bin/pdf-processor-rocm -dir /path/to/pdfs -skip-empty

# JSON output
./bin/pdf-processor-rocm -dir /path/to/pdfs -format json

# Strip headers from individual pages
./bin/pdf-processor-rocm -dir /path/to/pdfs -strip-header
```

## Command Line Options

```
  -dir string        Directory to scan for PDF files (default: ".")
  -pdf-workers int   Number of PDF files to process concurrently (default: 2)
  -workers int       Number of parallel crop/ocr workers per PDF (default: CPU cores)
  -dpi int           PDF render DPI (default: 300)
  -x1 int            Crop box X1 (default: 154)
  -y1 int            Crop box Y1 (default: 236)
  -x2 int            Crop box X2 (default: 2392)
  -y2 int            Crop box Y2 (default: 3007)
  -lang string       Tesseract OCR language (default: "eng")
  -format string     Output format: md, txt, json (default: "md")
  -checkpoint string Path to checkpoint file for resume support
  -skip-empty        Skip pages with no extracted text
  -strip-header      Remove header and image from individual page files
  -pretty            Format text for better readability
  -no-cleanup        Keep intermediate image files
  -v                 Enable verbose logging
```

## Performance

| Feature | Improvement |
|---------|-------------|
| Concurrent PDFs | Multiple PDFs processed simultaneously |
| Resume support | Skip already processed pages on restart |
| OCR retry | Auto-retry failed OCR (3 attempts) |
| Buffered I/O | Faster file writes |

## Architecture

- `cmd/` - Main application with GPU-accelerated processing
- `gpu_lib/` - GPU library stubs for CUDA/ROCm

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
