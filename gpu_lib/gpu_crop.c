// GPU Crop Library - CPU Stub Implementation
// Use when CUDA toolkit is not available

int cuda_device_count() {
    return 1; // Simulate available
}

int gpu_batch_crop(const char** paths, int count, int x1, int y1, int x2, int y2, int worker_id) {
    // Stub: just return count as success
    return count;
}