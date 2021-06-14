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

    extern "C" void 
    beam_times_sky(
        CUDACONV::RunConfig cfg, 
        float* ptgBuffer,
        int bnside, int bnpixels, 
        float* beamsA, float* beamsB, 
        int skynside, float* sky,
        int* intraBeamPixels, int* maxIntraBeamPix,
        float *data);
        
    extern "C" void 
    streamed_beam_times_sky(
        CUDACONV::RunConfig cfg, 
        float* ptgBuffer,
        int bnside, int bnpixels, 
        float* beamsA, float* beamsB, 
        int skynside, float* sky,
        int* intraBeamPixels, int* maxIntraBeamPix,
        cudaStream_t stream,
        float *data);
}
#endif
