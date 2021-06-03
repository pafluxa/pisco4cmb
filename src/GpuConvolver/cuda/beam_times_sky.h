#include "GpuConvolver/gpuconvolver.hpp"

#ifndef CUDACONVBEAMTIMESSKYH__
#define CUDACONVBEAMTIMESSKYH__

extern "C" void 
cudaconv_beam_times_sky(
    CUDACONV::RunConfig cfg, 
    float* ptgBuffer,
    int bnside, int bnpixels, 
    float* beamsA, float* beamsB, 
    int skynside, float* sky,
    int* intraBeamPixels, int* maxIntraBeamPix,
    float *data);

#endif
