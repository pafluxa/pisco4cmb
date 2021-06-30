/*
 * gpuconvolver.hpp 
 * 
 */
#ifndef _GPUCONVOLVERH
#define _GPUCONVOLVERH

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Polbeam/polbeam.hpp"
#include "GpuConvolver/cuda/cudaConv2.h"

#define N_POINTING_COORDS 4
#define N_SKY_COMP 4
#define N_POLBEAMS 4

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
        void update_scan(Scan* scan);
		void exec_convolution(float* data_a, float* data_b);
        void configure_execution(void);
        
    private:
        CUDACONV::RunConfig cfg;
        
		int nsamples;
        int nsideSky;
        int npixSky;
        int nsideBeam;
        int npixBeam;
        double beamRhoMax;
        
        float *bcptg;
        
        bool hasScan;
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
		int* nPixelsInDisc;
		int* nPixelsInDiscGPU;
        size_t nPixelInDiscBufferSize;
        /* buffers to store intra-beam sky pixels. */
        int* skyPixelsInBeam;
		int* skyPixelsInBeamGPU;
        int* matchingBeamPixelsGPU;
        float* chiAnglesGPU;
        size_t skyPixelsInBeamBufferSize;
        /* buffer to store pointing*/
        float* ptgBuffer;
        float* ptgBufferGPU;
        size_t ptgBufferSize;
        /* buffer to store result.*/
        float* result;     
        float* resultGPU;     
        size_t resultBufferSize;

        void allocate_buffers(void);
};

#endif
