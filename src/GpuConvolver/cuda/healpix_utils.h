#ifndef __CUDAHEALPIX__
#define __CUDAHEALPIX__

#include <cuda.h>

namespace cudaHealpix
{
    // Ported from Healpix_base.cc
    __device__ int ang2pix(int nside_map, float tht, float phi);
    // Ported from Healpix_base.cc
    __device__ void pix2ang(long nside, long ipix, float *theta, float *phi);                                         
    // Ported from Healpix_base.cc
    #ifdef CUDACONV_USE_INTERPOLATION
    __device__ void get_interpol(int nside_, float theta, float phi, int pix[4], float wgt[4]);
    #endif
// end namespace
}
#endif
