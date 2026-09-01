//go:build cuda
// +build cuda

package main

/*
#cgo LDFLAGS: -L${SRCDIR}/../gpu_lib -lgpu_crop -ldl
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>

static void* crop_lib = NULL;
static int (*gpu_crop_fn)(const unsigned char*, int, int, int, unsigned char*, int, int, int, int) = NULL;

static int init_crop_lib() {
    if (crop_lib) return 1;
    crop_lib = dlopen("./gpu_lib/libgpu_crop.so", RTLD_NOW);
    if (!crop_lib) crop_lib = dlopen("gpu_lib/libgpu_crop.so", RTLD_NOW);
    if (!crop_lib) return 0;
    gpu_crop_fn = (int(*)(const unsigned char*,int,int,int,unsigned char*,int,int,int,int))dlsym(crop_lib, "gpu_crop_image");
    return gpu_crop_fn != NULL;
}

int gpu_crop_image(const unsigned char* src, int src_w, int src_h, int channels,
                   unsigned char* dst, int x1, int y1, int x2, int y2) {
    if (!init_crop_lib()) return -1;
    return gpu_crop_fn(src, src_w, src_h, channels, dst, x1, y1, x2, y2);
}
*/
import "C"
import "unsafe"

func gpuCropImage(srcData []byte, srcW, srcH, channels int, dstData []byte, x1, y1, x2, y2 int) int {
	ret := C.gpu_crop_image(
		(*C.uchar)(unsafe.Pointer(&srcData[0])),
		C.int(srcW), C.int(srcH), C.int(channels),
		(*C.uchar)(unsafe.Pointer(&dstData[0])),
		C.int(x1), C.int(y1), C.int(x2), C.int(y2),
	)
	return int(ret)
}
