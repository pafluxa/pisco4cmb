#ifndef __CUDAUTILSH__
#define __CUDAUTILSH__

#include <stdio.h>
#include <cuda.h>

// Not that fancy error checking routine                                                                      
// CUDA ERROR CHECKING CODE                                                                                   
#define gpu_error_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }     
                                   
inline void gpuAssert(cudaError_t code, const char file[], int line, bool abort=true)                         
{                                                                                                             
   if (code != cudaSuccess)                                                                                   
   {                                                                                                          
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);                          
      if (abort){ getchar(); }                                                                                  
   }                                                                                                          
}                                                                                                             

#endif
