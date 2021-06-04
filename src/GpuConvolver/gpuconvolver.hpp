/*
 * gpuconvolver.hpp 
 * 
 */
#ifndef _GPUCONVOLVERH
#define _GPUCONVOLVERH

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Polbeam/polbeam.hpp"
#include "GpuConvolver/cuda/cudaconv.h"

#define N_POINTING_COORDS 4
#define N_SKY_COMP 4
#define N_POLBEAMS 4

class GPUConvolver
{
    public:
    
        GPUConvolver(Scan& scan, CUDACONV::RunConfig);
       ~GPUConvolver();
        
        /* Transfers beams from host to device. First call also 
         * allocates respective buffers.
         */
		void update_beam(PolBeam* beam);
		/* Transfer sky from host to devices. First call also allocates
         * respective buffers.
         */
        void update_sky(Sky* sky);
        /* Runs convolution algorithm using a Scan. Store the resulting
         * data stream in data_a/b, which is expected to be allocated.
         * polFlag indicates which of the buffers (data_a, data_b or 
         * both) will be used ('a', 'b' or 'p' for pair).
         */
		void exec_convolution( 
            CUDACONV::RunConfig c, 
            float* data_a,
            float* data_b,
            Scan* scan, Sky* sky, PolBeam* beam);
		
    private:

		int nsamples;
		bool hasBeam;
		bool hasSky;
        int pixPerDisc;
        int ptgPerConv;
        /* Sky has 4 Stokes parameters (I, Q, U, V)*/
        float* skyGPU;
        size_t skyBufferSize;
        /* Each beam has 4 componets*/
		float* aBeamsGPU;
		float* bBeamsGPU;
        size_t beamBufferSize;
        /* buffers to store number of intra-beam sky pixels. */
		int* nPixelsInDisc;
		int* nPixelsInDiscGPU;
        size_t nPixelInDiscBufferSize;
        /* buffers to store intra-beam sky pixels. */
        int* skyPixelsInBeam;
		int* skyPixelsInBeamGPU;
        size_t skyPixelsInBeamBufferSize;
        /* buffer to store pointing*/
        float* ptgBuffer;
        float* ptgBufferGPU;
        size_t ptgBufferSize;
        /* buffer to store result.*/
        float* result;     
        float* resultGPU;     
        size_t resultBufferSize;
};

#endif
