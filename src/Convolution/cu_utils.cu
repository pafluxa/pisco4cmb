#include "cu_utils.h"
#include <iostream>

__global__ void cudaUtilities::__kernel__ewxty(int n, float* x, float* y, float* z)
{
    int tid;
    int off;
    int idx;
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    off = blockDim.x * gridDim.x;

    idx = tid;
    while(idx < n) 
    {
        z[idx] = x[idx] * y[idx];
        idx += off;
    }
}

void cudaUtilities::ewxty(int n, float* x, float* y, cudaStream_t stream, float *z)
{
    dim3 gridcfg(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y, 1);
    dim3 blockcfg(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);

    __kernel__ewxty<<<gridcfg, blockcfg, 0, stream>>>(n, x, y, z);

    /*
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Error in kernel launch: ";
        std::cerr << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    */ 
}