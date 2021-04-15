/*
 * gpuconvolver.hpp 
 * 
 */
#ifndef _GPUCONVOLVERH
#define _GPUCONVOLVERH

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Polbeam/polbeam.hpp"
#include "GpuConvolver/cuda/beam_times_sky.h"

class GPUConvolver
{
    public:
    
        GPUConvolver(int nsamples, CUDACONV::RunConfig);
       ~GPUConvolver();
        
        /* Transfers beams from host to device. First call also 
         * allocates respective buffers.
         */
		void update_beam(PolBeam beam);
		/* Transfer sky from host to devices. First call also allocates
         * respective buffers.
         */
        void update_sky(CUDACONV::RunConfig cfg, Sky sky);
        /* Runs convolution algorithm using a Scan. Store the resulting
         * data stream in data_a/b, which is expected to be allocated.
         * polFlag indicates which of the buffers (data_a, data_b or 
         * both) will be used ('a', 'b' or 'p' for pair).
         */
		void exec_convolution( 
            CUDACONV::RunConfig c, 
            float* data_a,
            float* data_b,
            char polFlag,
            Scan scan, Sky sky, PolBeam beam  );
		
    private:

		int nsamples;
		bool hasBeam;
		bool hasSky;
        int chunkSize;
        /* Sky has 4 Stokes parameters (I, Q, U, V)*/
        float *skyGPU;
        size_t skyBufferSize;
        /* Each beam has 6 componets*/
		float* aBeamsGPU;
		float* bBeamsGPU;
        size_t beamBufferSize;
		/* buffers to store intra-beam sky pixels. */
		int* nPixelsInDisc;
		int* nPixelsInDiscGPU;
        size_t nPixelInDiscBufferSize;
        int* skyPixelsInBeam;
		int* skyPixelsInBeamGPU;
        size_t skyPixelsInBeamBufferSize;
        /* buffer to store pointing*/
        double* ptgBuffer;
        double* ptgBufferGPU;
        size_t ptgBufferSize;
        /* buffer to store result.*/
        float* result;     
        float* resultGPU;     
        size_t resultBufferSize;
};

#endif
