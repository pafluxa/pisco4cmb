#ifndef __CUDAHEALPIXH__
#define __CUDAHEALPIXH__

#include <cuda.h>

namespace cudaHealpix
{
    // Ported from Healpix_base.cc
    __device__ int ang2pix(int nside_map, double tht, double phi);
    // Ported from Healpix_base.cc
    __device__ void pix2ang(long nside, long ipix, double *theta, double *phi);                                         
    // Ported from Healpix_base.cc
    __device__ void get_interpol(int nside_, double theta, double phi, int pix[4], double wgt[4]);
};

#endif
