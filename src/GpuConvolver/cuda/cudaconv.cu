#include "cuda/cudaconv.h"
#include "cuda/healpix_utils.cuh"
#include "cuda/sphtrigo.cuh"
#include "cuda/cuda_utils.h"

__global__ void
gpu_convolve( 
	// pointing
	double ra_bc, double dec_bc, double pa_bc, 
	// detector polarization angle
	double pol_angle,
	// beams
	int bnside, 
	const float* __restrict__ bI, 
	const float* __restrict__ bQ, 
	const float* __restrict__ bU, 
	// sky
	int snside, 
	const float* __restrict__ sI, 
	const float* __restrict__ sQ, 
	const float* __restrict__ sU,
	// disc of pixels inside the beam
	int npixDisc, 
	const int* __restrict__ discPixels,
	// output
	float *data )
{
	int i, idx, eval_pixel;
	
	double psi_bc;
	double cmb_pol_angle;
	
	float I,Q,U;
	float I_pix, U_pix, Q_pix;
	float q, cq, sq, rotQ_pix, rotU_pix;
	
	int     neigh_pixels[4];
	double           wgt[4]; 
	
	double dec_eval, ra_eval;
	double tht_at_pix, phi_at_pix, chi_at_pix;
	double rbI, rbQ, rbU; // Rotated Beam {I,Q,U}
	
	idx = threadIdx.x + blockIdx.x*blockDim.x;	
	
	// Shared memory buffers to store block-wise computations
    __shared__ double sh_convI[ CUDA_BLOCK_SIZE ];
    __shared__ double sh_convQ[ CUDA_BLOCK_SIZE ];
    __shared__ double sh_convU[ CUDA_BLOCK_SIZE ];
    
    sh_convI[ threadIdx.x ] = 0.0;
    sh_convQ[ threadIdx.x ] = 0.0;
    sh_convU[ threadIdx.x ] = 0.0;
	
    cmb_pol_angle = -pol_angle;
    psi_bc = -pa_bc;        

	while( idx < npixDisc ) { 
		       
        eval_pixel = discPixels[ idx ];
		I_pix = sI[eval_pixel];
		Q_pix = sQ[eval_pixel]; 
		U_pix = sU[eval_pixel]; 

		// Get sky coordinates of eval_pixel
		dec_eval = 0.0; ra_eval = 0.0;
		cuHealpix_pix2ang( snside, eval_pixel, &dec_eval, &ra_eval );
		// pix2ang returns co-latitude.
		dec_eval = M_PI_2 - dec_eval;
		
		// find the corresponding antenna basis coordinates
		// safety initializers
		tht_at_pix = 0.0; phi_at_pix = 0.0; chi_at_pix = 0.0;
		rho_sigma_chi_pix( 
			&tht_at_pix, &phi_at_pix, &chi_at_pix,
			ra_bc, dec_bc, psi_bc, ra_eval, dec_eval );
		
		// rotate Stokes vector from the sky by 2 x chi
		q = 2*chi_at_pix;
		// yes, I am a cheap bastard for using cosf
		cq = cosf(q); 
		sq = sinf(q);
		rotQ_pix =  Q_pix*cq + U_pix*sq;
		rotU_pix = -Q_pix*sq + U_pix*cq;
		
		// safe initializers before interpolation
		neigh_pixels[0] = 0; neigh_pixels[1] = 0;
		neigh_pixels[2] = 0; neigh_pixels[2] = 0;
		wgt[0] = 0.0; wgt[1] = 0.0; wgt[2] = 0.0; wgt[3] = 0.0;
		
		// obtain interpolation weights and neighbors 
		cuHealpix_interpolate( 
			bnside, tht_at_pix, phi_at_pix, 
			neigh_pixels, wgt ); 
			
		// interpolate polarized beams
		rbI = 0.0; rbQ = 0.0; rbU = 0.0;
		for( i=0; i < 4; i++ ) {   
			rbI += bI[neigh_pixels[i]] * wgt[i];
			rbQ += bQ[neigh_pixels[i]] * wgt[i];
			rbU += bU[neigh_pixels[i]] * wgt[i];
		}
		
		sh_convI[threadIdx.x] +=    I_pix*rbI;
		sh_convQ[threadIdx.x] += rotQ_pix*rbQ;
		sh_convU[threadIdx.x] += rotU_pix*rbU;
		
		idx += blockDim.x * gridDim.x;
	}
	
	__syncthreads();

	// Use a tree structure to do reduce result
	for (int stride = blockDim.x/2; stride >  0; stride /= 2) {
		if ( threadIdx.x < stride) {
		   sh_convI[threadIdx.x] += sh_convI[threadIdx.x + stride];
		   sh_convQ[threadIdx.x] += sh_convQ[threadIdx.x + stride];
		   sh_convU[threadIdx.x] += sh_convU[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if( threadIdx.x == 0 ) {
		I = float( sh_convI[0] );
		Q = float( sh_convQ[0] );
		U = float( sh_convU[0] );
		data[blockIdx.x] = I + 0.5*( Q*cosf(2*cmb_pol_angle) + U*sinf(2*cmb_pol_angle) );
	}
}

extern "C" void
_cuda_convolve( 
	// pointing
	double ra_bc, double dec_bc, double pa_bc, 
	// detector polarization angle
	double pol_angle,
	// beams
	int bnside, float *bI, float* bQ, float *bU,
	// sky
	int snside, float *sI, float* sQ, float *sU,
	// disc of pixels inside the beam
	// TODO: move this inside an object
	int npixDisc, int* discPixels,
	// gpu buffer
	float *gpuBuff,
	// stream
	cudaStream_t stream )
{
	gpu_convolve<<< CUDA_NUM_BLOCKS, CUDA_BLOCK_SIZE, 0, stream >>>
	(
		ra_bc, dec_bc, pa_bc, pol_angle,
		bnside, bI, bQ, bU,
		snside, sI, sQ, sU,
		npixDisc, discPixels,
		gpuBuff
	);
}
