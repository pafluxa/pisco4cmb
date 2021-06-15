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
#define CUDACONV_MAXSTREAMS (8)

class GPUConvolver
{
    public:
    
        GPUConvolver(CUDACONV::RunConfig);
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
            float* data_a,
            float* data_b,
            Scan* scan, Sky* sky, PolBeam* beam);
		
    private:
        CUDACONV::RunConfig cfg;
        
		int nsamples;
		bool hasBeam;
		bool hasSky;
        bool hasValidConfig;
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
		int* nPixelsInDisc[CUDACONV_MAXSTREAMS];
		int* nPixelsInDiscGPU[CUDACONV_MAXSTREAMS];
        size_t nPixelInDiscBufferSize;
        /* buffers to store intra-beam sky pixels. */
        int* skyPixelsInBeam[CUDACONV_MAXSTREAMS];
		int* skyPixelsInBeamGPU[CUDACONV_MAXSTREAMS];
        size_t skyPixelsInBeamBufferSize;
        /* buffer to store pointing*/
        float* ptgBuffer[CUDACONV_MAXSTREAMS];
        float* ptgBufferGPU[CUDACONV_MAXSTREAMS];
        size_t ptgBufferSize;
        /* buffer to store result.*/
        float* result[CUDACONV_MAXSTREAMS];     
        float* resultGPU[CUDACONV_MAXSTREAMS];     
        size_t resultBufferSize;

        cudaEvent_t streamWait[CUDACONV_MAXSTREAMS];
        cudaStream_t streams[CUDACONV_MAXSTREAMS];
        
        void allocate_buffers(void);
        
        void configure_execution(void);
};

#endif
