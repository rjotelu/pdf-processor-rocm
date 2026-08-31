//go:build rocm
// +build rocm

package main

/*
#cgo LDFLAGS: -ldl

#include <stdlib.h>
#include <dlfcn.h>
#include <stdio.h>

static void* rocm_lib = NULL;
static int (*rocm_dev_count_fn)() = NULL;
static int (*batch_crop_fn)(const char**, int, int, int, int, int, int) = NULL;

static int init_lib() {
    if (rocm_lib) return 1;
    rocm_lib = dlopen("./gpu_lib/libgpu_crop_rocm.so", RTLD_NOW);
    if (!rocm_lib) rocm_lib = dlopen("gpu_lib/libgpu_crop_rocm.so", RTLD_NOW);
    if (!rocm_lib) return 0;
    rocm_dev_count_fn = (int(*)())dlsym(rocm_lib, "rocm_device_count");
    batch_crop_fn = (int(*)(const char**,int,int,int,int,int,int))dlsym(rocm_lib, "rocm_batch_crop");
    return rocm_dev_count_fn && batch_crop_fn;
}

int check_rocm() {
    return init_lib() && rocm_dev_count_fn() > 0;
}
*/
import "C"
import "fmt"

type GPUProcessor struct {
	available bool
}

func NewGPUProcessor() *GPUProcessor {
	available := C.check_rocm() == 1
	if available {
		fmt.Println("[✓] ROCm GPU detected")
	} else {
		fmt.Println("[!] ROCm GPU not available")
	}
	return &GPUProcessor{available: available}
}

func (p *GPUProcessor) Available() bool { return p.available }