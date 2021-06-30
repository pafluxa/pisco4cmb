#ifndef _CUDACONVH
#define _CUDACONVH
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define BLOCKS_PER_GRID_X 16
#define BLOCKS_PER_GRID_Y 128

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

    void launch_fill_pixel_matching_matrix_kernel
    (
        int nptgs, float* ptgBuffer,
        int nsideSky, int nsideBeam,
        int maxPixelsPerDisc, int pixelsPerDisc[], int intraBeamSkyPixels[],
        // output
        int matchingBeamPixels[], float chi[]
    );

    void launch_partial_polarized_convolution_kernel
    (
       int nsideSky, int nSkyPixels, float sky[],
       int nsideBeam, int nBeamPixels, float beamDetA[], float beamDetB[],
       int nptgs, int npixperptg, 
       int pixperptg[], int intraBeamSkyPixels[], int matchingBeamPixels[], float chiAngles[],
       // output
       float* detDataA, float* detDataB
    );
}
#endif
