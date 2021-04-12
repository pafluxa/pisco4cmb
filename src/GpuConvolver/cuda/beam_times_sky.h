#ifndef CUDACONVH__
#define CUDACONVH__
#define MAXDISCSIZE 10000

namespace CUDACONV
{
    struct CUDACONVConf_t
    {
        int chunkSize;
        int gridSizeX;
        int gridSizeY;
        int blockSizeX;
        int blockSizeY;
        int MAX_PIXELS_PER_DISC;
    };
    typedef struct CUDACONVConf_t RunConfig;
}

void beam_times_sky(
    CUDACONV::RunConfig cfg, 
    int nptgs,
    double* ptgBuffer,
    int bnside, int bnpixels, 
    float* beamsA, float* beamsB, 
    int skynside, float* sky,
    int* intraBeamPixels, 
    int* maxIntraBeamPix,
    float *data);

#endif
