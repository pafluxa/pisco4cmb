#ifndef _CUDACONVH
#define _CUDACONVH

namespace CUDACONV
{
    // maximum allowed amount of pixels in a disc in a pixel
    #define CUDACONV_MAXPIXELSPERDISC 131072
    #define CUDACONV_MAXPTGPERCONV 131072
    struct CUDACONVConf_t
    {
        int gridSizeX;
        int gridSizeY;
        int blockSizeX;
        int blockSizeY;
        int ptgPerConv;
        int pixelsPerDisc;

        int MAX_PIXELS_PER_DISC = CUDACONV_MAXPIXELSPERDISC;
        int MAX_PTGS_PER_CONV = CUDACONV_MAXPTGPERCONV;
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
}
#endif
