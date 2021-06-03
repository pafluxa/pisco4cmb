#include <iostream>
#include "beam_times_sky.h"
#include "sphtrigo.h"
#include "healpix_utils.h"

#ifdef CUDACONV_SKYASTEXTURE
// Texture reference for 1D float4 texture
texture<float4, 1, cudaReadModeElementType> tex_skyGPU;
#endif

__global__ void
kernel_beam_times_sky
(
    int threadsPerBlock,
    
    int nptgs,
    float* ptgBuffer, 
    
    int bnside, 
    int bnpixels, 

    float* beamsA, 
    float* beamsB, 
    
    int skynside, 
    float* sky,
    
    int pixelsPerDisc,
    int* intraBeamPixels, 
    int* maxIntraBeamPix,
    
    float *data)
{   
    extern __shared__  double sharedMem[];

    int ix0;
    int iy0;
    int sp;
    int bp;
    int nptgpix;
    #ifdef CUDACONV_USE_INTERPOLATION
    int k;
    int m;
    int ni;
    int neighPixels[4];
    double ww;
    double wgt[4];
    #endif
    
    double rasp, decsp;
    double rabc, pabc, decbc;
    double rho, sigma, chi, q, c2chi, s2chi;

    double baval[4];
    double bbval[4];
    
    double* shA = &(sharedMem[0]);
    double* shB = &(sharedMem[threadsPerBlock]);
    
    #ifdef CUDACONV_SKYASTEXTURE
    float4 skyPixelData;
    #endif
    float I, Q, U, V;
    
    iy0 = blockIdx.x;
    ix0 = threadIdx.x;  
    for(int iy = iy0; iy < nptgs; iy += gridDim.x)
    {
        /* read in pointing. */
        rabc  = ptgBuffer[4 * iy + 0];
        decbc = ptgBuffer[4 * iy + 1];
        pabc  = ptgBuffer[4 * iy + 2];
        /* zero out shared memory.*/
        shA[threadIdx.x] = 0.0;
        shB[threadIdx.x] = 0.0;
        /* threads in the x direction process beam x sky */ 
        nptgpix = maxIntraBeamPix[iy];
        for(int ix = ix0; ix < nptgpix; ix += blockDim.x) 
        {
            /* Get evaluation pixel from intra-beam pixel buffer. */
            sp = intraBeamPixels[iy * pixelsPerDisc + ix];
            // fetch values from memory only once
            #ifdef CUDACONV_SKYASTEXTURE
            // Get value at sky_pixel
            skyPixelData  = tex1Dfetch(tex_skyGPU, sp);
            I = skyPixelData.x; 
            Q = skyPixelData.y; 
            U = skyPixelData.z; 
            V = skyPixelData.w; 
            #else
            I = sky[4 * sp + 0];
            Q = sky[4 * sp + 1];
            U = sky[4 * sp + 2];
            V = sky[4 * sp + 3];
            #endif
            /* Get sky coordinates of the evaluation pixel. */
            cudaHealpix::pix2ang(skynside, sp, &decsp, &rasp);
            /*
            if(threadIdx.x == 0 )
            {
                printf("sky pixel = %d\n", sp);
                printf("sky nside = %d\n", skynside);
                printf("sky value = %f\n", I);
                printf("tht pix = %2.8lf\n", decsp);
                printf("phi pix = %2.8lf\n", rasp);
            }
            */
            decsp = M_PI_2 - decsp;
            /* Compute rho sigma and chi at beam pixel. */
            cudaSphericalTransformations::rho_sigma_chi_pix(
                &rho, &sigma, &chi, 
                rabc, decbc, pabc, rasp, decsp);
            /* sin() and cos() of off-center polarization angle. */
            q = 2.0 * chi;
            c2chi = cos(q);
            s2chi = sin(q);
            #ifdef CUDACONV_USE_INTERPOLATION
            // Interpolate beam at (rho,sigma).	
            cudaHealpix::get_interpol(bnside, rho, sigma, neighPixels, wgt);         
            // apply beam interpolation at (rho, sigma)
            for(m = 0; m < 4; m++)
            {
                // reset beam value buffers
                baval[m] = 0.0;
                bbval[m] = 0.0;
                for(k = 0; k < 4; k++)
                {
                    ni = neighPixels[k];
                    if(ni < bnpixels)
                    {
                        ww = wgt[k];
                        baval[m] += ww * double(beamsA[4 * ni + m]);
                        bbval[m] += ww * double(beamsB[4 * ni + m]);
                    }
                }
            }
            #else
            /* Get the closest beam pixel to beam coordinates (rho, sigma). */
            bp = cudaHealpix::ang2pix(bnside, rho, sigma);
            baval[0] = beamsA[4 * bp + 0];
            baval[1] = beamsA[4 * bp + 1];
            baval[2] = beamsA[4 * bp + 2];
            baval[3] = beamsA[4 * bp + 3];
            
            bbval[0] = beamsB[4 * bp + 0];
            bbval[1] = beamsB[4 * bp + 1];
            bbval[2] = beamsB[4 * bp + 2];
            bbval[3] = beamsB[4 * bp + 3];
            #endif
            // sky - beam multiplication. 
            shA[threadIdx.x] += I * baval[0]
                + Q * (baval[1] * c2chi - baval[2] * s2chi)
                + U * (baval[2] * c2chi + baval[1] * s2chi)
                + V * baval[3];
            shB[threadIdx.x] += I * bbval[0] 
                + Q * (-bbval[1] * c2chi + bbval[2] * s2chi)
                + U * (-bbval[2] * c2chi - bbval[1] * s2chi)
                + V * bbval[3];
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
            data[2*iy + 0] = shA[0];
            data[2*iy + 1] = shB[0];
        }
    }
}
 
extern void 
cudaconv_beam_times_sky( 
    CUDACONV::RunConfig cfg, 
	// pointing
    float* ptgBufferGPU, 
	// beams
	int bnside, int bnpixels, 
    float* beamsAGPU, float* beamsBGPU, 
	// sky
	int skynside, float* skyGPU, 
	// disc of pixels inside the beam
	int* intraBeamPixelsGPU, int* maxIntraBeamPixGPU,
	// output
	float *dataGPU)
{
    dim3 gridcfg(cfg.gridSizeX, cfg.gridSizeY, 1);
    dim3 blockcfg(cfg.blockSizeX, cfg.blockSizeY, 1);
    size_t shMemBufferSize = 2 * cfg.blockSizeX * sizeof(double);
    
    #ifdef CUDACONV_SKYASTEXTURE
    // bind sky to texture memory
    size_t skyBufferSize = \
        sizeof(float) * N_SKY_COMP * 12 * skynside * skynside;
    cudaBindTexture(0, tex_skyGPU, skyGPU, skyBufferSize);
    #endif
    
    kernel_beam_times_sky
    <<<gridcfg, blockcfg, shMemBufferSize>>>
    (
        cfg.blockSizeX, 
        cfg.ptgPerConv, ptgBufferGPU,
        bnside, bnpixels, beamsAGPU, beamsBGPU, 
        skynside, skyGPU, 
        cfg.pixelsPerDisc, intraBeamPixelsGPU, maxIntraBeamPixGPU,
        dataGPU
    );
    #ifdef CUDACONV_SKYASTEXTURE
    cudaDeviceSynchronize();
    cudaUnbindTexture(tex_skyGPU);
    #endif
}
