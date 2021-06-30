#include <assert.h> 

#include <cstdlib>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <iomanip>   
  
#include <healpix_base.h>
#include <pointing.h>

#include <cuda.h>

#include "cudaHealpix.h"

#define MAXNSIDE (4096)
#define MAXNPIXELS (12 * MAXNSIDE * MAXNSIDE)

__global__ void
kernel_test_pix2ang
(
    int nside, int npix, int pixels[],
    double ra_pix[], double dec_pix[]
)
{
    int id = threadIdx.x * blockDim.x + blockIdx.x;
    while(id < npix)
    {
        cudaHealpix::pix2ang(nside, pixels[id], &(dec_pix[id]), &(ra_pix[id]));
        id += blockDim.x * gridDim.x;
    }
}

__global__ void
kernel_test_ang2pix
(
    int nside, int npix, int pixels[],
    double ra_pix[], double dec_pix[]
)
{
    int id = threadIdx.x * blockDim.x + blockIdx.x;
    while(id < npix)
    {
        pixels[id] = cudaHealpix::ang2pix(nside, dec_pix[id], ra_pix[id]);
        id += blockDim.x * gridDim.x;
    }
}

int main(int argc, char** argv )nv
{
    pointing ptg;
    
    int iter;
    int nside;
    int npixels;
    
    /** allocate arrays only once using the largest size. */
    int*    pixels;
    int*    pixels_cuda;
    double* rapix_cuda ;
    double* rapix      ;
    double* decpix     ;
    double* decpix_cuda;
    
    int* gpu_pixels;
    double* gpu_rapix;
    double* gpu_decpix;
    
    pixels = (int*)malloc(MAXNPIXELS * sizeof(int));
    pixels_cuda = (int*)malloc(MAXNPIXELS * sizeof(int));
    rapix_cuda = (double*)malloc(MAXNPIXELS * sizeof(double));
    rapix = (double*)malloc(MAXNPIXELS * sizeof(double));
    decpix = (double*)malloc(MAXNPIXELS * sizeof(double));
    decpix_cuda = (double*)malloc(MAXNPIXELS * sizeof(double));

    cudaMalloc((void **)&(gpu_pixels), MAXNPIXELS * sizeof(int));    
    cudaMalloc((void **)&(gpu_rapix), MAXNPIXELS * sizeof(double));
    cudaMalloc((void **)&(gpu_decpix), MAXNPIXELS * sizeof(double));    
    
    nside = 2;
    while(nside <= MAXNSIDE)
    {
        std::cout << "testing nside = " << nside << std::endl;
        npixels = 12 * nside * nside;
        Healpix_Base hpx(nside, RING, SET_NSIDE);
        
        int i;
        for(i = 0; i < npixels; i++)
        {
            pixels[i] = i;
            pointing ptg = hpx.pix2ang(i);   
            rapix[i] = ptg.phi;
            decpix[i] = ptg.theta;
        }
        // execute pix2ang test
        cudaMemcpy(gpu_pixels, pixels, npixels * sizeof(int), cudaMemcpyHostToDevice);
        kernel_test_pix2ang<<<64, 64>>>(nside, npixels, gpu_pixels, gpu_rapix, gpu_decpix);
        cudaMemcpy(rapix_cuda, gpu_rapix, npixels * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(decpix_cuda, gpu_decpix, npixels * sizeof(double), cudaMemcpyDeviceToHost);
        
        // execute ang2pix test
        cudaMemcpy(gpu_rapix, rapix, npixels * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_decpix, decpix, npixels * sizeof(double), cudaMemcpyHostToDevice);
        kernel_test_ang2pix<<<64, 64>>>(nside, npixels, gpu_pixels, gpu_rapix, gpu_decpix);
        cudaMemcpy(pixels_cuda, gpu_pixels, npixels * sizeof(int), cudaMemcpyDeviceToHost);
        
        for(i = 0; i < npixels; i++)
        {
            assert(abs(rapix[i] - rapix_cuda[i]) <= 1e-10);
            assert(abs(decpix[i] - decpix_cuda[i]) <= 1e-10);
            assert(pixels[i] - pixels_cuda[i] == 0);
        }
        nside = nside * 2;
    }
    
    return 0;
}
