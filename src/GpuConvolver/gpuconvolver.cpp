
#include <cstring>

#include <omp.h>
#include <pointing.h>
#include <arr.h>
#include <rangeset.h>


#include "cuda/cudaconv.h"
#include "gpuconvolver.hpp"
#include "cuda/cuda_utils.h"

GpuConvolver::GpuConvolver( 
	unsigned long nsamples, 
	unsigned  int nstreams) {
	
	_nsamples = nsamples;
	_nstreams = nstreams;

	// create streams	
	for( int i=0; i<_nstreams; i++ ) {
		cudaStream_t s;
		cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking);
		_gpuStreams.push_back(s); 
	}
	
	_hasSky  = false;
	_hasBeam = false;

	// take no risk and make all chunks slightly larger than needed
	int chunkSize = _nsamples/_nstreams;
	int reminder  = _nsamples%_nstreams;
	int samps = chunkSize + reminder;
	std::cout << samps << " " << samps*_nstreams << " " << nsamples << std::endl;
	
	size_t blockConvBuffSize = sizeof(float)*CUDA_NUM_BLOCKS;
	size_t perThreadBuffSize = sizeof(float)*(samps);

	for( int i=0; i<_nstreams; i++ ) {
		float *t1, *cb, *gb;
		t1 = (float*)malloc( perThreadBuffSize );
		gpu_error_check( cudaMallocHost( (void **)&cb, CUDA_NUM_BLOCKS*sizeof(float) ) );
		gpu_error_check( cudaMalloc(     (void **)&gb, CUDA_NUM_BLOCKS*sizeof(float) ) );
		
		_cpuConvBuffers.push_back(t1);
		
		_cpuBlockConvBuffers.push_back(cb);
		_gpuBlockConvBuffers.push_back(gb);
	}
}

GpuConvolver::~GpuConvolver() {
	
	for( int i=0; i<_nstreams; i++ ) {
		float *temp;
		int   *itemp;
		temp = _gpuBlockConvBuffers.back();
		gpu_error_check(cudaFree(temp));
		_gpuBlockConvBuffers.pop_back();
		
		temp = _cpuBlockConvBuffers.back();
		gpu_error_check(cudaFreeHost(temp))
		_cpuBlockConvBuffers.pop_back();
		
		itemp = _gpuDiscBuffers.back();
		gpu_error_check(cudaFree(itemp));
		_gpuDiscBuffers.pop_back();
		
		itemp = _cpuDiscBuffers.back();
		gpu_error_check(cudaFreeHost(itemp));
		_cpuDiscBuffers.pop_back();
	}
	
	gpu_error_check( cudaFree( _gpu_skyI ) );
	gpu_error_check( cudaFree( _gpu_skyQ ) );
	gpu_error_check( cudaFree( _gpu_skyU ) );
	
	gpu_error_check( cudaFree( _gpu_beamI ) );
	gpu_error_check( cudaFree( _gpu_beamQ ) );
	gpu_error_check( cudaFree( _gpu_beamU ) );
	
	for( int i=0; i<_nstreams; i++ ) {
		cudaStream_t s = _gpuStreams.back();
		gpu_error_check( cudaStreamDestroy( s ) );
		_gpuStreams.pop_back(); 
	}
	
	for( int i=0; i<_nstreams; i++ ) {
		float *tb = _cpuConvBuffers.back();
		free( tb );
		_cpuConvBuffers.pop_back();
	}
	
}

void GpuConvolver::set_beam( PolBeam& beam ) {
	
    // allocate beam buffers on the GPU
    _beamBufferSize = (int)(1.5*beam.size())*sizeof(float) ;
    gpu_error_check( cudaMalloc( (void **) &_gpu_beamI, _beamBufferSize ) );  
    gpu_error_check( cudaMalloc( (void **) &_gpu_beamQ, _beamBufferSize ) );  
    gpu_error_check( cudaMalloc( (void **) &_gpu_beamU, _beamBufferSize ) );  
    
    // set elements of GPU beams to zero
    cudaMemset( _gpu_beamI, 0.0, _beamBufferSize );
    cudaMemset( _gpu_beamQ, 0.0, _beamBufferSize );
    cudaMemset( _gpu_beamU, 0.0, _beamBufferSize );

    // move beams to GPU
    gpu_error_check( 
	  cudaMemcpy( 
		_gpu_beamI, beam.get_stokes_I_ptr(), 
		beam.size()*sizeof(float), cudaMemcpyHostToDevice ) );
    gpu_error_check( 
	  cudaMemcpy( 
		_gpu_beamQ, beam.get_stokes_Q_ptr(), 
		beam.size()*sizeof(float), cudaMemcpyHostToDevice ) );
    gpu_error_check( 
	  cudaMemcpy( 
		_gpu_beamU, beam.get_stokes_U_ptr(), 
		beam.size()*sizeof(float), cudaMemcpyHostToDevice ) );
	
	_hasBeam = true;
}

void GpuConvolver::set_sky( Sky& sky, double rhoMax ) {
		
    // allocate sky buffers on the GPU
    _skyBufferSize = sky.size()*sizeof(float);
    
    gpu_error_check( cudaMalloc( (void **) &_gpu_skyI, _skyBufferSize ) ); 
    gpu_error_check( cudaMalloc( (void **) &_gpu_skyQ, _skyBufferSize ) ); 
    gpu_error_check( cudaMalloc( (void **) &_gpu_skyU, _skyBufferSize ) ); 

	// move sky to GPU
    gpu_error_check( cudaMemcpy( _gpu_skyI, sky.get_stokes_I_ptr(), _skyBufferSize, cudaMemcpyHostToDevice ) );
    gpu_error_check( cudaMemcpy( _gpu_skyQ, sky.get_stokes_Q_ptr(), _skyBufferSize, cudaMemcpyHostToDevice ) );
    gpu_error_check( cudaMemcpy( _gpu_skyU, sky.get_stokes_U_ptr(), _skyBufferSize, cudaMemcpyHostToDevice ) );
   
    // allocate buffer for disc indexes + headroom. Use pinned memory
    // for better performance when transferring it. The maximum amount
    // of pixels in the disc is when the beam points at the equator.
    rangeset< int > ibr;
    pointing sc( 0.0, 0.0 );
	sky.hpxBase.query_disc( sc, rhoMax, ibr );
	
	// TODO: remove this 20% headroom!
	_discBufferSize = (int)(ibr.nval()*1.2)*sizeof(int);
	
	// allocate space for results coming from all streams
	for( int i=0; i<_nstreams; i++ ) {
		
		int* _discIdx_host = nullptr;
		int*  _discIdx_gpu = nullptr;
		
		gpu_error_check( cudaMalloc((void **)&_discIdx_gpu, _discBufferSize ) );		
		// pinned memory!
		gpu_error_check( cudaMallocHost( (void **)&_discIdx_host, _discBufferSize ) );

		_cpuDiscBuffers.push_back( _discIdx_host );
		_gpuDiscBuffers.push_back( _discIdx_gpu );

	}
    	
	gpu_error_check( cudaGetLastError() );
	
	_hasSky = true;
}

void GpuConvolver::exec_convolution( float* convData, Scan& scan, Sky& sky, PolBeam& beam )
{
	double phi0, dec0, psi0;
	
	// convenience pointers
	const float* phicrd = scan.get_phi_ptr();
	const float* thetacrd = scan.get_theta_ptr();
	const float* psicrd  = scan.get_psi_ptr();
	
	int startEnd[2*_nstreams];
	
	omp_set_num_threads(_nstreams);
	
	#pragma omp parallel default(shared)
	{ // begin parallel region
	
	// OpenMP sucks so I rather calculate start/end blocks by hand...
	int start, end;
	int thrId  = omp_get_thread_num();
	int stride = omp_get_num_threads();
	int chunkSize  = (scan.get_nsamples() / stride );
	int reminder   = (scan.get_nsamples() % stride );

	start = thrId * chunkSize;
	if( thrId + 1 == stride ) {
		end = start + chunkSize + reminder; 
	}
	else {
		end = start + chunkSize;
	}

	startEnd[2*thrId]   = start;
	startEnd[2*thrId+1] =   end;
	
	float* tempConvData = _cpuConvBuffers[thrId];
	
	double data;
	unsigned long s;
	for( s=start; s<end; s++ ) {
		
		phi0 = phicrd[s];
		dec0 = M_PI_2 - thetacrd[s];
		psi0 = psicrd[s];
		
		_beam_times_sky( sky, beam, phi0, dec0, psi0, thrId );

		// reduce
		float *convBlock = _cpuBlockConvBuffers.at(thrId);
		data = 0;
		// reduce cpu result
		for( int b=0; b<CUDA_NUM_BLOCKS; b++) {
			data += convBlock[b];
		}
		tempConvData[s] = data;
	}
	
	} // end parallel region
	
	// rebuild convolution buffer
	int t;
	size_t buffsize;
	long start, end, delta;
	for( t=0; t<_nstreams; t++) {
		
		start = startEnd[2*t];
		end   = startEnd[2*t+1];
		delta = end - start;
		buffsize = sizeof(float)*delta;
		float* td = _cpuConvBuffers.at(t);
		cudaMemcpy( convData + start, td, buffsize, cudaMemcpyHostToHost );
	}
}

void GpuConvolver::_beam_times_sky( 
	Sky& sky, PolBeam &beam,
	float ra_bc, float dec_bc, float pa_bc, 
	int streamId )
{	
	// find sky pixels around beam center up to rhoMax
	rangeset< int > intraBeamRanges;
	pointing sc( M_PI/2.0 - dec_bc, ra_bc );
	sky.hpxBase.query_disc( sc, beam.get_rho_max(), intraBeamRanges );
	
	int *_discIdx_host = _cpuDiscBuffers[streamId];
	int *_discIdx_gpu  = _gpuDiscBuffers[streamId];
	
	int idx = 0;
	// flatten rangeset into pinned buffer
	for( int r=0; r < intraBeamRanges.nranges(); r++ ) {
		int s = intraBeamRanges.ivbegin(r);
		int e = intraBeamRanges.ivend(r);
		for( int i = s; i < e; i++ ) {
			_discIdx_host[idx] = i;
			idx++;
		}
	}
	
	size_t convBuffSize = sizeof(float)*CUDA_NUM_BLOCKS;
	size_t discBuffSize = intraBeamRanges.nval()*sizeof(int);
	
	gpu_error_check(
		cudaMemcpyAsync(
			_gpuDiscBuffers.at(streamId), 
			_cpuDiscBuffers.at(streamId),
			discBuffSize, cudaMemcpyDeviceToHost, _gpuStreams[streamId] ) );
	
	_cuda_convolve( 
		ra_bc, dec_bc, pa_bc, 
		_detAngle,
		beam.get_nside(), _gpu_beamI, _gpu_beamQ, _gpu_beamU,
		sky.get_nside(), _gpu_skyI, _gpu_skyQ, _gpu_skyU,
		intraBeamRanges.nval(), _discIdx_gpu,
		_gpuBlockConvBuffers.at(streamId), _gpuStreams[streamId] );

	gpu_error_check(
		cudaMemcpyAsync(
			_cpuBlockConvBuffers.at(streamId), 
			_gpuBlockConvBuffers.at(streamId), 
			convBuffSize, cudaMemcpyHostToDevice, _gpuStreams[streamId]) );
}
