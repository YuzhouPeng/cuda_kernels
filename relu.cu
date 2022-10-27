#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define CUDA_KERNEL_LOOP(1,n) \
for (int i=blockIdx.x*blockDim.x+threadIdx.x; \
i<(n); \
i+=blockDim.x*gridDim.x)

#define FLT_MAX 999999999999

__global__ void relu(const float* A,float* B,const int nthreads)
{
    CUDA_KERNEL_LOOP(index,nthreads){
        if(A[index]>0){
            B[index]=A[index];
        }else{
            B[index]=0;
        }
    }
}