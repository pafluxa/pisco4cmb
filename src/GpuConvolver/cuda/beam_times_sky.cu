#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_error_check.h"

#include "cuda/sphtrigo.h"
#include "cuda/healpix_utils.h"
#include "cuda/beam_times_sky.h"

//using namespace CUDACONV;

__global__ void kernel_beamtimessky
(
    int nptgs,
    double* ptgBuffer, 
    
    int bnside, 
    int bnpixels, 
    float* beamsA, 
    float* beamsB, 
    
    int skynside, 
    float* sky,
    
    int* intraBeamPixels, 
    int* maxIntraBeamPix,
    
    float *data)
{   
    __shared__  double shA[64];
    __shared__  double shB[64];

    int ix0;
    int iy0;
    
    int sp;
    int ni;
    int nptgpix;
    int neighPixels[4];
    
    double ww;
    double I, Q, U, V;//one day V, one day
    double rasp, decsp;
    double rabc, pabc, decbc;
    double rho, sigma, chi, q, c2chi, s2chi;

    double wgt[4];
    double baval[6];
    double bbval[6];

    iy0 = blockIdx.x;
    ix0 = threadIdx.x;  
    for(int iy = iy0; iy < nptgs; iy += gridDim.x)
    {
        /* read in pointing. */
        rabc  = ptgBuffer[4*iy+0];
        decbc = ptgBuffer[4*iy+1];
        pabc  = ptgBuffer[4*iy+2];
        /* zero out shared memory.*/
        shA[threadIdx.x] = 0.0;
        shB[threadIdx.x] = 0.0;
        /* threads in the x direction process beam x sky */ 
        nptgpix = maxIntraBeamPix[iy];
        for(int ix = ix0; ix < nptgpix; ix += blockDim.x) 
        {
            /* Get evaluation pixel from intra-beam pixel buffer. */
            sp = intraBeamPixels[iy*MAXDISCSIZE + ix];
            /* Get sky coordinates of the evaluation pixel. */
            cuHealpix::pix2ang(skynside, sp, &decsp, &rasp);
            decsp = M_PI_2 - decsp;
            /* Compute rho sigma and chi at beam pixel. */
            cuSphericalTransformations::rho_sigma_chi_pix(
                &rho, &sigma, &chi, rabc, decbc, pabc, rasp, decsp);
            /* sin() and cos() of off-center polarization angle. */
            q = 2.0*chi;
            c2chi = cos(q);
            s2chi = sin(q);
            /* Interpolate beam at (rho,sigma). */		
            cuHealpix::get_interpol(
                bnside, rho, sigma, neighPixels, wgt);         
            for(int b = 0; b < 6; b++) {
                baval[b] = 0.0;
                bbval[b] = 0.0;
                for(int j=0; j < 4; j++) {
                    ni = neighPixels[j];
                    if(ni < bnpixels) {
                        ww = wgt[j];
                        baval[b] += ww*double(beamsA[6*ni + b]);
                        bbval[b] += ww*double(beamsB[6*ni + b]);
                    }
                }
            }
            /* sky beam multiplication. */
            I = sky[4*sp + 0];
            Q = sky[4*sp + 1];
            U = sky[4*sp + 2];
            //V = sky[4*sp + 3];
            //printf("%E %E %E\n", I, Q, U);
            /* I = I + Qcos + Usin + V .*/
            shA[threadIdx.x] +=
                + I*(baval[0])
                + Q*(baval[1]*c2chi - baval[2]*s2chi)
                + U*(baval[3]*c2chi + baval[4]*s2chi);
                //+ V*0.0;
            shB[threadIdx.x] +=
                + I*(bbval[0])
                + Q*(-bbval[1]*c2chi + bbval[2]*s2chi)
                + U*(-bbval[3]*c2chi - bbval[4]*s2chi);
                //+ V*0.0;
        }
        /* wait for all threads to reduce */
        __syncthreads();
        /* Use a tree structure to do reduce result. */
        for(int stride = blockDim.x/2; stride > 0; stride /= 2) 
        {
            if(threadIdx.x < stride) 
            {
                shA[threadIdx.x] += shA[threadIdx.x + stride];
                shB[threadIdx.x] += shB[threadIdx.x + stride];
            }
            __syncthreads();
        }
        /* Final write to global memory. */
        if(threadIdx.x == 0) 
        {
            data[2*iy + 0] = 0.5*float(shA[0]);
            data[2*iy + 1] = 0.5*float(shB[0]);
        }
        
        __syncthreads();
    }
}
 

void beam_times_sky( 
    CUDACONV::RunConfig cfg, 
	// pointing
    int nptg, 
    double* ptgBufferGPU, 
	// beams
	int bnside, 
    int bnpixels, 
    float* beamsAGPU, 
    float* beamsBGPU, 
	// sky
	int skynside, 
    float* skyGPU, 
	// disc of pixels inside the beam
	int* intraBeamPixelsGPU, 
    int* maxIntraBeamPixGPU,
	// output
	float *dataGPU)
{
    dim3 gridcfg(cfg.gridSizeX, cfg.gridSizeY, 1);
    dim3 blockcfg(cfg.blockSizeX, cfg.blockSizeY, 1);
    kernel_beamtimessky
    <<<gridcfg, blockcfg>>>
    (
        nptg, ptgBufferGPU,
        bnside, bnpixels, beamsAGPU, beamsBGPU, 
        skynside, skyGPU, 
        intraBeamPixelsGPU, maxIntraBeamPixGPU,
        dataGPU
    );
}
