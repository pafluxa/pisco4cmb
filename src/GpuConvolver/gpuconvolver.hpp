/*
 * gpuconvolver.hpp 
 * 
 */
#ifndef _GPUCONVOLVERH
#define _GPUCONVOLVERH

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Polbeam/polbeam.hpp"

class GPUConvolver
{
    public:
    
        GPUConvolver(long nsamples);
       ~GPUConvolver();
        
        /* Transfers beams from host to device. First call also 
         * allocates respective buffers.
         */
		void update_beam(PolBeam beam);
		/* Transfer sky from host to devices. First call also allocates
         * respective buffers.
         */
        void update_sky(Sky sky);
        /* Runs convolution algorithm using a Scan. Store the resulting
         * data stream in data_a/b, which is expected to be allocated.
         * polFlag indicates which of the buffers (data_a, data_b or 
         * both) will be used ('a', 'b' or 'p' for pair).
         */
		void exec_convolution( 
            float* data_a,
            float* data_b,
            char polFlag,
            Scan scan, Sky sky, PolBeam beam  );
		
    private:

		long nsamples;
		bool hasBeam;
		bool hasSky;
        int parallelLaunches;
        /* Sky has 4 Stokes parameters (I, Q, U, V)*/
        size_t skySize;
        float *skyGPU;
        /* Each beam has 6 componets*/
        size_t beamSize;
		float* aBeamsGPU;
		float* bBeamsGPU;
		/* To store sky pixels seen by the beam. */
        size_t skyPixelsInBeamSize;
		std::vector<int* >skyPixelsInBeam;
		std::vector<int* >skyPixelsInBeamGPU;   
        float** resultGPU;     
        float** result;     
};

#endif
