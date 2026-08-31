<<<<<<< HEAD
// ROCm GPU Crop Library - CPU Stub Implementation
// Use when ROCm/HIP is not available

int rocm_device_count() {
    return 1; // Simulate available
}

int rocm_batch_crop(const char** paths, int count, int x1, int y1, int x2, int y2, int worker_id) {
    // Stub: just return count as success
=======
/* ROCm GPU Crop Library - CPU Stub */
/* Replace with actual HIP kernels when available */

int rocm_device_count() {
    return 1;
}

int rocm_batch_crop(const char** paths, int count, int x1, int y1, int x2, int y2, int worker_id) {
>>>>>>> 8bff987 (Initial commit: ROCm version with GPU acceleration)
    return count;
}