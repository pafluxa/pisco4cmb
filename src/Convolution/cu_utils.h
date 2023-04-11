#ifndef __CU_UTILS__
#define __CU_UTILS__

#include <cuda.h>

// needed to launch kernel
#define BLOCKS_PER_GRID_X 256
#define BLOCKS_PER_GRID_Y 1
#define THREADS_PER_BLOCK_X 256
#define THREADS_PER_BLOCK_Y 1


namespace cudaUtilities
{
    
/*   
 * cuda_ewxty()
 * 
 * Calculates the element wise multiplication of arrays "x" and "y".
 * 
 * Stores the result in "z"
 */

// cuda kernel
__global__ void __kernel__ewxty(int n, float* x, float* y, float *z);
// callable function
void ewxty(int n, float* x, float *y, cudaStream_t stream, float* z);

// end namespace
} 
#endif