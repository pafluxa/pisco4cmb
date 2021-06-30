#include <iostream>

#include "cudaSphtransf.h"
#include "cudaHealpix.h"
#include "cudaConv2.h"

__device__ void
skypixel_to_beampixel_and_chi
/**
 * Description
 * #####################################################################
 * Calculates the closest beam pixel that corresponds to a sky pixel
 * given the beam center pointing (ra, dec and position angle). This
 * routine also returns the cosine and sine of two times the angle 
 * between the beam co-polarization vector and the meridian of the 
 * corresponding sky pixel.
 * #####################################################################
 * 
 * Input
 * #####################################################################
 * rabc : right ascention of beam center
 * decbc : declination of beam center
 * pabc : position angle at beam center
 * nsideBeam: nside parameter (healpix specific) of beam 
 * nsideSky: nside parameter (healpix specific) of sky
 * skypix: sky pixel (healpix ring indexing) 
 * #####################################################################
 * 
 * Output
 * #####################################################################
 * beampix: closest beam pixel
 * chi: angle between co-polarization vector and meridian at sky pixel
 * #####################################################################
 *
 **/ 
(
    float rabc, float decbc, float pabc,
    int skypix,
    int nsideBeam, int nsideSky,
    int *beampix, float* chi
)
{   
    double rasp, decsp, rho, sigma, chi_;
    
    /* Get sky coordinates of the evaluation pixel. */
    cudaHealpix::pix2ang(nsideSky, skypix, &decsp, &rasp);
    decsp = M_PI_2 - decsp;
    /* Compute rho sigma and chi at beam pixel. */
    cudaSphericalTransformations::rho_sigma_chi_pix(
        &rho, &sigma, &chi_, 
        rabc, decbc, pabc, 
        rasp, decsp);
    /* calculate the beam pixel corresponding to coordinates
     * rho and sigma */
    *beampix = cudaHealpix::ang2pix(nsideBeam, rho, sigma);
    *chi = float(chi_);
}

__global__ void
fill_pixel_matching_matrix
/**
 * Description
 * #####################################################################
 * This routine fills a matrix with the matching beam pixel for every
 * sky pixel that is specified in the intraBeamSkyPixel matrix. 
 * #####################################################################
 * 
 * Input
 * #####################################################################
 * nptg: number of pointing directions
 * rabc, decbc, pabc: right ascention, declination and position angle
 *                    of beam center, respectively
 * nsideSky, nsideBeam: nside parameter (healpix specific) of sky and
 *                      beam, respectively.
 * maxPixelsPerDisc: maximum number of pixels in each disc containing 
 *                   the beam.
 * intraBeamSkyPixels: list of sky pixels inside the beam. 
 * #####################################################################
 * 
 * Output
 * #####################################################################
 * matchingBeamPixels: list of beam pixels that match the corresponding
 *                     sky pixel for every entry of intraBeamSkyPixels.
 * chi: angle between the co-polarization vector of the beam and the 
 *      meridian of the matching sky pixel.
 * #####################################################################
 **/
(
    int nptgs, float ptgBuffer[],
    int nsideSky, int nsideBeam,
    int npixperptg, int pixperptg[], int intraBeamSkyPixels[],
    // output
    int matchingBeamPixels[], float chi[]
)
{
    int ix;
    int iy;
    int npixdisc;
    int skyPixel;
    int beamPixel;
    float rabc, decbc, pabc, angleChi;
    
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    while(iy < nptgs)
    {
        npixdisc = pixperptg[iy];
        rabc = ptgBuffer[4 * iy + 0];
        decbc = ptgBuffer[4 * iy + 1];
        pabc = ptgBuffer[4 * iy + 2];
        
        ix = blockIdx.x * blockDim.x + threadIdx.x;
        while(ix < npixdisc)
        {
            skyPixel = intraBeamSkyPixels[npixperptg * iy + ix];
            skypixel_to_beampixel_and_chi
            (
                rabc, decbc, pabc,
                skyPixel, nsideBeam, nsideSky,
                &beamPixel, &angleChi
            );
            matchingBeamPixels[npixperptg * iy + ix] = beamPixel;
            chi[npixperptg * iy + ix] = angleChi;
            ix += blockDim.x * gridDim.x;
        }
        iy += blockDim.y * gridDim.y;
    }
    
}

__global__ void
partial_polarized_convolution
/**
 * Description
 * Calculate the partial convolution of a polarized sky and a polarized
 * beam as the tensor-product described in (reference to paper)
 */
(
   int nsideSky, int nSkyPixels, float sky[],
   int nsideBeam, int nBeamPixels, float beamDetA[], float beamDetB[],
   int nptgs, int maxPixelsPerDisc, int pixelsPerDisc[],
   int intraBeamSkyPixels[], int matchingBeamPixels[], float chiAngles[],
   // output
   float* detDataA, float* detDataB
)
{
    __shared__ double shA[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];
    __shared__ double shB[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];
    
    int ix, iy;
    int skyPixel, beamPixel;
    float chi, c2chi, s2chi;
    float I, Q, U, V;
    float baI, baQ, baU, baV;
    float bbI, bbQ, bbU, bbV;
    
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    while(iy < nptgs)
    {
        shA[threadIdx.y][threadIdx.x] = 0;
        shB[threadIdx.y][threadIdx.x] = 0;
        ix = blockIdx.x * blockDim.x + threadIdx.x;
        while(ix < pixelsPerDisc[iy])
        {
            // set beam a to zero
            baI = 0.0f;
            baQ = 0.0f;
            baU = 0.0f;
            baV = 0.0f;
            // set beam b to zero
            bbI = 0.0f;
            bbQ = 0.0f;
            bbU = 0.0f;
            bbV = 0.0f;
            // set sky to zero
            I = 0.0f;
            Q = 0.0f;
            U = 0.0f;
            V = 0.0f;
            // threads in the x direction process a single dot-product between
            // the polarized sky and the polarized beam
            skyPixel = intraBeamSkyPixels[maxPixelsPerDisc * iy + ix];
            beamPixel = matchingBeamPixels[maxPixelsPerDisc * iy + ix];
            chi = chiAngles[maxPixelsPerDisc * iy + ix];
            c2chi = cosf(2.0f * chi);
            s2chi = sinf(2.0f * chi);
            // this could be replaced by texture fetching?
            if(skyPixel < nSkyPixels)
            {
                I = sky[4 * skyPixel + 0];
                Q = sky[4 * skyPixel + 1];
                U = sky[4 * skyPixel + 2];
                V = sky[4 * skyPixel + 3];
            }
            if(beamPixel < nBeamPixels)
            {
                // beam det A
                baI = beamDetA[4 * beamPixel + 0];
                // baval[1] * c2chi - baval[2] * s2chi
                baQ = beamDetA[4 * beamPixel + 1] * c2chi 
                    - beamDetA[4 * beamPixel + 2] * s2chi;
                // baval[2] * c2chi + baval[1] * s2chi)
                baU = beamDetA[4 * beamPixel + 2] * c2chi 
                    + beamDetA[4 * beamPixel + 1] * s2chi;
                baV = beamDetA[4 * beamPixel + 3];
                
                // beam det B
                bbI = beamDetB[4 * beamPixel + 0];
                // -bbval[1] * c2chi + bbval[2] * s2chi
                bbQ = -beamDetB[4 * beamPixel + 1] * c2chi
                     + beamDetB[4 * beamPixel + 2] * s2chi;
                //-bbval[2] * c2chi - bbval[1] * s2chi
                bbU = -beamDetB[4 * beamPixel + 2] * c2chi 
                     - beamDetB[4 * beamPixel + 1] * s2chi;
                bbV = beamDetB[4 * beamPixel + 3];
            }
            // store result on shared buffers, for speed
            shA[threadIdx.y][threadIdx.x] += double(I * baI + Q * baQ + U * baU + V * baV);
            shB[threadIdx.y][threadIdx.x] += double(I * bbI + Q * bbQ + U * bbU + V * bbV);
            ix += blockDim.x * gridDim.x;
        }
        /* use a tree structure to do reduce along x (pixels). */
        for(int stride = blockDim.x/2; stride > 0; stride /= 2) 
        {
            if(threadIdx.x < stride) 
            {
                shA[threadIdx.y][threadIdx.x] += shA[threadIdx.y][threadIdx.x + stride];
                shB[threadIdx.y][threadIdx.x] += shB[threadIdx.y][threadIdx.x + stride];
            }
            __syncthreads();
        }
        /* only one thread per block is allowed to execute this. */
        if(threadIdx.x == 0)
        {
            atomicAdd(&(detDataA[iy]), float(shA[threadIdx.y][0]));
            atomicAdd(&(detDataB[iy]), float(shB[threadIdx.y][0]));
        }
        
        iy += blockDim.y * gridDim.y;
    } 
}

void CUDACONV::launch_fill_pixel_matching_matrix_kernel
(
    int nptgs, float* ptgBuffer,
    int nsideSky, int nsideBeam,
    int maxPixelsPerDisc, int pixelsPerDisc[], int intraBeamSkyPixels[],
    // output
    int matchingBeamPixels[], float chi[]
)
{
    dim3 gridcfg(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y, 1);
    dim3 blockcfg(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    
    fill_pixel_matching_matrix<<<gridcfg, blockcfg>>>
    (
         nptgs, ptgBuffer,
         nsideSky,  nsideBeam,
         maxPixelsPerDisc,  pixelsPerDisc, intraBeamSkyPixels,
         matchingBeamPixels, chi
    );
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error in kernel launch: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void CUDACONV::launch_partial_polarized_convolution_kernel
(
   int nsideSky, int nSkyPixels, float sky[],
   int nsideBeam, int nBeamPixels, float beamDetA[], float beamDetB[],
   int nptgs, int maxPixelsPerDisc, int pixelsPerDisc[],
   int intraBeamSkyPixels[], int matchingBeamPixels[], float chiAngles[],
   // output
   float* detDataA, float* detDataB
)
{
    dim3 gridcfg(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y, 1);
    dim3 blockcfg(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    
    partial_polarized_convolution<<<gridcfg, blockcfg>>>
    (
        nsideSky,  nSkyPixels,  sky,
        nsideBeam,  nBeamPixels,  beamDetA,  beamDetB,
        nptgs,  maxPixelsPerDisc,  pixelsPerDisc,
        intraBeamSkyPixels,  matchingBeamPixels,  chiAngles,
        detDataA,  detDataB
    );
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error in kernel launch: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

/*
extern void 
CUDACONV::beam_times_sky( 
    CUDACONV::RunConfig cfg, 
	// pointing
    int nptgs, float* ptgBufferGPU, 
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
    dim3 gridcfg(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y, 1);
    dim3 blockcfg(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    
    int* mbpm;
    float* chiangles;
    cudaMalloc((void **)&(mbpm), sizeof(int) * nptgs * cfg.pixelsPerDisc);
    cudaMalloc((void **)&(chiangles), sizeof(float) * nptgs * cfg.pixelsPerDisc);

    std::cout << "launching kernel 1" << std::endl;
    fill_pixel_matching_matrix<<<gridcfg, blockcfg>>>
    (
        nptgs, ptgBufferGPU, 
        skynside, 
        bnside, 
        cfg.pixelsPerDisc, maxIntraBeamPixGPU, intraBeamPixelsGPU,
        mbpm, chiangles
    );

    std::cout << "launching kernel 2" << std::endl;
    partial_polarized_convolution<<<gridcfg, blockcfg>>>
    (
       skynside, 12 * skynside * skynside, skyGPU,
       bnside, bnpixels, beamsAGPU, beamsBGPU,
       nptgs, cfg.pixelsPerDisc, maxIntraBeamPixGPU,
       intraBeamPixelsGPU, mbpm, chiangles,
       // output
       &(dataGPU[0]), &(dataGPU[nptgs])
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error in kernel launch: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    cudaFree(mbpm);
    cudaFree(chiangles);
}
*/
