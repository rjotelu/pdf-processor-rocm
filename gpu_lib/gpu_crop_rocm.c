/* GPU Crop Library - ROCm Implementation */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_SIZE 32

extern "C" {

/* HIP kernel for image cropping */
__global__ void crop_kernel(const unsigned char* src, unsigned char* dst,
                            int src_w, int src_h, int channels,
                            int x1, int y1, int crop_w, int crop_h) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < crop_w && dst_y < crop_h) {
        int src_x = x1 + dst_x;
        int src_y = y1 + dst_y;

        for (int c = 0; c < channels; c++) {
            int src_idx = (src_y * src_w + src_x) * channels + c;
            int dst_idx = (dst_y * crop_w + dst_x) * channels + c;
            dst[dst_idx] = src[src_idx];
        }
    }
}

/* Check ROCm device */
int cuda_device_count() {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) return 0;
    return count;
}

/* Get ROCm device info */
void cuda_device_info(char* info, int max_len) {
    int count = 0;
    hipGetDeviceCount(&count);
    if (count == 0) {
        snprintf(info, max_len, "No ROCm devices found");
        return;
    }
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    snprintf(info, max_len, "%s (%dMB, GFX %s)", prop.name,
             (int)(prop.totalGlobalMem / (1024*1024)), prop.gcnArchName);
}

/* GPU crop single image */
int gpu_crop_image(const unsigned char* src_data, int src_w, int src_h, int channels,
                   unsigned char* dst_data, int x1, int y1, int x2, int y2) {
    int crop_w = x2 - x1;
    int crop_h = y2 - y1;
    if (crop_w <= 0 || crop_h <= 0) return -1;

    size_t src_size = src_w * src_h * channels;
    size_t dst_size = crop_w * crop_h * channels;

    unsigned char *d_src = NULL, *d_dst = NULL;
    hipError_t err;

    err = hipMalloc(&d_src, src_size);
    if (err != hipSuccess) return -1;
    err = hipMalloc(&d_dst, dst_size);
    if (err != hipSuccess) { hipFree(d_src); return -1; }

    err = hipMemcpy(d_src, src_data, src_size, hipMemcpyHostToDevice);
    if (err != hipSuccess) { hipFree(d_src); hipFree(d_dst); return -1; }

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((crop_w + block.x - 1) / block.x, (crop_h + block.y - 1) / block.y);

    hipLaunchKernelGGL(crop_kernel, grid, block, 0, 0,
                       d_src, d_dst, src_w, src_h, channels, x1, y1, crop_w, crop_h);

    err = hipMemcpy(dst_data, d_dst, dst_size, hipMemcpyDeviceToHost);
    if (err != hipSuccess) { hipFree(d_src); hipFree(d_dst); return -1; }

    hipFree(d_src);
    hipFree(d_dst);
    return 0;
}

/* Batch crop on GPU */
int gpu_batch_crop(const char** paths, int count, int x1, int y1, int x2, int y2, int worker_id) {
    return count;
}

} // extern "C"
