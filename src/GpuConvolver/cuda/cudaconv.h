#ifndef _CUDACONV
#define _CUDACONV

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CUDA_BLOCK_SIZE    32
#define CUDA_NUM_BLOCKS   128

extern "C" void
_cuda_convolve( 
	// pointing
	double ra_bc, double dec_bc, double pa_bc, 
	// detector polarization angle
	double pol_angle,
	// beams
	// TODO: move this inside an object
	int bnside, float *tI, float* tQ, float *tU,
	// sky
	// TODO: move this inside an object
	int snside, float *sI, float* sQ, float *sU,
	// disc of pixels inside the beam
	int npixDisc, int* discPixels,
	// buffer to hold the result
	float *gpuBuff,
	// stream
	cudaStream_t stream );

#endif
