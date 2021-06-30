
#include <assert.h> 

#include <cstdlib>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <iomanip>   

#include <time.h>
#include <healpix_base.h>
#include <pointing.h>

#include <cuda.h>

#include "Sphtrigo/sphtrigo.hpp"
#include "GpuConvolver/cuda/cudaConv2.h"

#define MAXNSIDE (128)
#define MAXNPIXELS (12 * MAXNSIDE * MAXNSIDE)

void pointing_to_pixels
(
    int nptgs, float* ptgBuffer,
    int nsideSky, float rhoMax,
    int nibsp, int npixptg[], int ibsp[]
)
{
    int ipix, iptg;
    float rabc, decbc;
    Healpix_Base hpxSky(nsideSky, RING, SET_NSIDE);
    
    for(iptg = 0; iptg < nptgs; iptg++)
    {
        rabc = ptgBuffer[4 * iptg + 0];
        decbc = ptgBuffer[4 * iptg + 1];
        rangeset<int> ibr;
        pointing sc(M_PI_2 - decbc, rabc);
        hpxSky.query_disc(sc, rhoMax, ibr);
        
        ipix = 0;
        for(int rn = 0; rn < ibr.nranges(); rn++)
        {
            for(int skyPix = ibr.ivbegin(rn); skyPix < ibr.ivend(rn); skyPix++)
            {
                ibsp[nibsp * iptg + ipix] = skyPix;
                ipix++;
            }
        }
        npixptg[iptg] = ipix;
    }       
}

void match_sky_to_beam_pixels
(
    int nptgs, float *ptgBuffer,
    int nsideSky, int nsideBeam,
    int npixptg, int pixperptg[], int ibsp[],
    int *b2smpm, float* chiang
)
{
    int iptg, ispix, skypixel, bpix;
    double rho, sigma, chi;
    float rabc, decbc, pabc;
    
    pointing bptg;
    Healpix_Base shpx(nsideSky, RING, SET_NSIDE);
    Healpix_Base bhpx(nsideBeam, RING, SET_NSIDE);
    for(iptg = 0; iptg < nptgs; iptg++)
    {
        rabc = ptgBuffer[4 * iptg + 0];
        decbc = ptgBuffer[4 * iptg + 1];
        pabc = ptgBuffer[4 * iptg + 2];
        for(ispix = 0; ispix < pixperptg[iptg]; ispix++)
        {
            pointing spixptg;
            // get sky pixel
            skypixel = ibsp[npixptg * iptg + ispix];
            // calculate sky coordinates (ra, dec) of sky pixel
            spixptg = shpx.pix2ang(skypixel);
            // calculate beam coordinates (rho, sigma) given beam center pointing
            // and sky pixel coordinates
            SphericalTransformations::rho_sigma_chi_pix(
                &rho, &sigma, &chi,
                rabc, decbc, pabc,
                spixptg.phi, M_PI_2 - spixptg.theta);
            // calculate beam pixel from beam coordinates (rho, sigma)
            bptg.theta = rho;
            bptg.phi = sigma;
            bpix = bhpx.ang2pix(bptg);
            b2smpm[npixptg * iptg + ispix] = bpix;
            chiang[npixptg * iptg + ispix] = chi;
        }
    }
}

int main(int argc, char** argv )
{
    srand(time(0));
    
    pointing ptg;
    
    //int skyNpix;
    int skyNside;
    
    int beamNpix;
    int beamNside;
    
    int nptgs;
    float rhoMax = 5.0 * (M_PI / 180.0); // beam is 10 degrees in extension
    skyNside = 256;
    //skyNpix = 12 * skyNside * skyNside;
    beamNside = 256;
    Healpix_Base bhpx(beamNside, RING, SET_NSIDE);
    pointing edgeptg;
    edgeptg.theta = rhoMax;
    edgeptg.phi = M_PI;
    beamNpix = bhpx.ang2pix(edgeptg);
    int npixptg = int(beamNpix * 1.2);

    nptgs = 5000;
    Healpix_Base shpx(skyNside, RING, SET_NSIDE);
    
    float* ptgBuffer = (float*)malloc(sizeof(float) * nptgs * 4);
    for(int i = 0; i < nptgs; i++)
    {
        float p = 2 * M_PI * (float(rand())/float(RAND_MAX));
        float t = M_PI * (float(rand())/float(RAND_MAX));
        float s = M_PI * (float(rand())/float(RAND_MAX));
        ptgBuffer[4 * i + 0] = p;
        ptgBuffer[4 * i + 1] = M_PI_2 - t;
        ptgBuffer[4 * i + 2] = s; // 0 deg position angle
        ptgBuffer[4 * i + 3] = 0.0; // dummy entry
    }

    int* pixperptg = (int*)malloc(nptgs * sizeof(int));
    int* ibsp = (int*)malloc(nptgs * npixptg * sizeof(int));
    // pointing to pixels
    pointing_to_pixels
    (
        nptgs, ptgBuffer, 
        skyNside, rhoMax, npixptg, pixperptg, ibsp
    );

    float* chiang = (float*)malloc(sizeof(float) * nptgs * npixptg);
    int* b2smpm = (int*)malloc(sizeof(int) * nptgs * npixptg);
    match_sky_to_beam_pixels
    (
        nptgs, ptgBuffer,
        skyNside, beamNside,
        npixptg, pixperptg, ibsp,
        b2smpm, chiang
    );
    
    // launch GPU routine
    float* gpu_ptgBuffer;
    int* gpu_pixperptg;
    int* gpu_ibsp;
    int* gpu_b2smpm;
    int* host_b2smpm;
    float* host_chiang;
    float* gpu_chiang;
    cudaMalloc((void **)&(gpu_ptgBuffer), 4 * nptgs * sizeof(float));
    cudaMalloc((void **)&(gpu_pixperptg), nptgs * sizeof(int));
    cudaMalloc((void **)&(gpu_ibsp), npixptg * nptgs * sizeof(int));
    
    cudaMemcpy(gpu_ptgBuffer, ptgBuffer, 4 * nptgs * sizeof(float), cudaMemcpyHostToDevice);     
    cudaMemcpy(gpu_pixperptg, pixperptg, nptgs * sizeof(int), cudaMemcpyHostToDevice);     
    cudaMemcpy(gpu_ibsp, ibsp, nptgs * npixptg * sizeof(int), cudaMemcpyHostToDevice);     

    cudaMalloc((void **)&(gpu_b2smpm), npixptg * nptgs * sizeof(int));
    cudaMalloc((void **)&(gpu_chiang), npixptg * nptgs * sizeof(float));

    std::cout << "launching kernel 1" << std::endl;
    CUDACONV::launch_fill_pixel_matching_matrix_kernel
    (
        nptgs, gpu_ptgBuffer, 
        skyNside, beamNside, 
        npixptg, gpu_pixperptg, gpu_ibsp,
        gpu_b2smpm, gpu_chiang
    );

    host_chiang = (float*)malloc(sizeof(float) * nptgs * npixptg);
    host_b2smpm = (int*)malloc(sizeof(int) * nptgs * npixptg);
    cudaMemcpy(host_chiang, gpu_chiang, nptgs * npixptg * sizeof(float), cudaMemcpyDeviceToHost);     
    cudaMemcpy(host_b2smpm, gpu_b2smpm, nptgs * npixptg * sizeof(int), cudaMemcpyDeviceToHost); 
    
    for(int iptg = 0; iptg < nptgs; iptg++)
    {
        int npix = pixperptg[iptg];
        for(int ipix = 0; ipix < npix; ipix++)
        {
            if(abs(chiang[npixptg * iptg + ipix] - host_chiang[npixptg * iptg + ipix]) > 1e-6 || 
               b2smpm[npixptg * iptg + ipix] - host_b2smpm[npixptg * iptg + ipix] != 0)
            {
                std::cout << std::setprecision(5);
                std::cout << "ra = " << (180.0 / M_PI) * ptgBuffer[4 * iptg + 0] << " deg ";
                std::cout << "dec = " << (180.0 / M_PI) * ptgBuffer[4 * iptg + 1] << " deg." << std::endl;
                std::cout << std::setprecision(12);
                std::cout << chiang[npixptg * iptg + ipix] << std::endl;
                std::cout << host_chiang[npixptg * iptg + ipix] << std::endl;
                std::cout << b2smpm[npixptg * iptg + ipix] << std::endl;
                std::cout << host_b2smpm[npixptg * iptg + ipix] << std::endl;
                
                pointing bp = bhpx.pix2ang(b2smpm[npixptg * iptg + ipix]);
                std::cout << "cpu sigma = " << (180.0 / M_PI) * bp.phi << " deg ";
                std::cout << "cpu rho = " << (180.0 / M_PI) * bp.theta << " deg." << std::endl;
                
                pointing bp2 = bhpx.pix2ang(host_b2smpm[npixptg * iptg + ipix]);
                std::cout << "gpu sigma = " << (180.0 / M_PI) * bp2.phi << " deg ";
                std::cout << "gpu rho = " << (180.0 / M_PI) * bp2.theta << " deg." << std::endl;
            }
            assert(abs(chiang[npixptg * iptg + ipix] - host_chiang[npixptg * iptg + ipix]) <= 1e-6);
            assert(b2smpm[npixptg * iptg + ipix] - host_b2smpm[npixptg * iptg + ipix] == 0);
        }
    }
    cudaFree(gpu_b2smpm);
    cudaFree(gpu_chiang);
    cudaFree(gpu_ibsp);
    cudaFree(gpu_pixperptg);
    cudaFree(gpu_ptgBuffer);
    
    free(b2smpm);
    free(chiang);
    free(pixperptg);
    free(ibsp);
    free(ptgBuffer);
    /*
    int*    pixels;
    int*    pixels_cuda;
    double* rapix_cuda ;
    double* rapix      ;
    double* decpix     ;
    double* decpix_cuda;
    double* papix_cuda;
    double* papix_cuda;
    
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
    */
    return 0;
}
