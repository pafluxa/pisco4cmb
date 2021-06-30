#ifndef _CUDACONVH
#define _CUDACONVH
#include <cuda.h>
#include <cuda_runtime.h>

namespace CUDACONV
{
    struct CUDACONVConf_t
    {   
        int nStreams;
        size_t maxMemUsage;
        
        int deviceId;
        int gridSizeX;
        int gridSizeY;
        int blockSizeX;
        int blockSizeY;
        int ptgPerConv;
        int pixelsPerDisc;
    };
    typedef struct CUDACONVConf_t RunConfig;

    __device__ void
    skypixel_to_beampixel_and_chi
    (
        float rabc, float decbc, float pabc,
        int skypix,
        int nsideBeam, int nsideSky,
        int *beampix, float* chi
    );

    __global__ void
    fill_pixel_matching_matrix
    (
        int nptgs, float* ptgBuffer,
        int nsideSky, int nsideBeam,
        int maxPixelsPerDisc, int pixelsPerDisc[], int intraBeamSkyPixels[],
        // output
        int matchingBeamPixels[], float chi[]
    );
     
    extern "C" void 
    beam_times_sky(
        CUDACONV::RunConfig cfg, 
        int nptgs, float* ptgBuffer,
        int bnside, int bnpixels, 
        float* beamsA, float* beamsB, 
        int skynside, float* sky,
        int* intraBeamPixels, int* maxIntraBeamPix,
        float data[]);
}
#endif
