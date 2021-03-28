/*
 * gpuconvolver.hpp 
 * 
 */
#ifndef _GPUCONVOLVERH
#define _GPUCONVOLVERH

#include <cuda.h>
#include <cuda_runtime.h>

#include "sky.hpp"
#include "scan.hpp"
#include "polbeam.hpp"

class GpuConvolver
{
    public:
    
        GpuConvolver( unsigned long nsamples, unsigned int nstreams = 1 );
       ~GpuConvolver();

		void set_beam(PolBeam& beam);
		void set_sky(Sky& sky, double rhoMax );
		void set_detector_angle( double angle ){ _detAngle = angle; }
		
		bool set_threads( unsigned int n );
		unsigned int get_threads( void );
		
		void exec_convolution( float* convData, Scan& scan, Sky& sky, PolBeam& beam  );
		
    private:

		unsigned long _nsamples;
		unsigned int  _nstreams;
		
		bool _hasBeam;
		bool _hasSky;
		
		size_t  _skyBufferSize;
	    size_t _beamBufferSize;
	    size_t _discBufferSize;
		
		float* _gpu_beamI;
		float* _gpu_beamQ;
		float* _gpu_beamU;
		
		float* _gpu_skyI;
		float* _gpu_skyQ;
		float* _gpu_skyU;

		float* _gpu_phi;
		float* _gpu_theta;
		float* _gpu_psi;
		
		double _detAngle;
		
		std::vector<int*> _cpuDiscBuffers;
		std::vector<int*> _gpuDiscBuffers;
		
		// to store n_stream outputs of the GPU
		std::vector<float*> _gpuBlockConvBuffers;
		std::vector<float*> _cpuBlockConvBuffers;
		
		// to store n_streams partial outputs
		std::vector<float*> _cpuConvBuffers;
		
		std::vector<cudaStream_t> _gpuStreams;

		void _beam_times_sky( Sky& sky, PolBeam &beam, float phi0, float theta0, float psi0, int streamId );
};

#endif
