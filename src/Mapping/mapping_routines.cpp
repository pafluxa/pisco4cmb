#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <chealpix.h>
#include <omp.h>

#include "Mapping/healpix_map.h"


#ifndef M_PI_2
#define M_PI_2 (1.5707963267948966)
#endif

//extern "C" int dgesv_(int *n, int *nrhs, double *a,
//           int *lda, int *ipiv, double *b, int *ldb, int *info);
extern "C" {
     void dgesv_(int *n, int *nrhs,  double *a,  int  *lda,
           int *ipivot, double *b, int *ldb, int *info) ;
}

void
libmapping_project_data_to_matrices
(
    // input
    int nsamples , int ndets,
    double ra[], double dec[], double pa[],
    double pol_angles[],
    double data[] , int bad_data_samples[], int dets_to_map[],
    int map_nside,
    int map_size , int pixels_in_the_map[],
    // output
    double AtA[], double AtD[]
)
{
    // Create healpixMaps that use the Python buffers
    // Note that all maps share the same mask!
    healpixMap *mask;
    mask = healpixMap_new();
    healpixMap_allocate(map_nside, mask);
    healpixMap_set_mask(mask,  pixels_in_the_map, map_size);
    // compute central pixel (dec=0, ra=PI)
    long CENTER_INDEX;
    ang2pix_ring(map_nside, M_PI_2, M_PI, &CENTER_INDEX);
    CENTER_INDEX = healpixMap_pix2idx(mask, CENTER_INDEX);
    for(int det=0; det < ndets; det++)
    {
        if(dets_to_map[det] == 0)
        {
            double pol_angle = pol_angles[det];

            for(int sample=0; sample < nsamples; sample++)
            {
                if(bad_data_samples[det*nsamples + sample] == 0)
                {
                    double _phi, _theta, _psi;
                    _phi   = ra[det * nsamples + sample];
                    _theta = M_PI_2 - dec[det * nsamples + sample];
                    // Passed arguments are counterclockwise on the sky
                    // CMB requires clockwise
                    _psi   = -(pol_angle + pa[det*nsamples + sample]);
                    long pix;
                    ang2pix_ring(map_nside, _theta, _phi, &pix);
                    // Transform pixel number to array index using the mask.
                    // Only project if the pixel is actually in the map.
                    long idx = healpixMap_pix2idx(mask, pix);
                    //long idx = pix;
                    if(idx >= 0)
                    {
                        double d = data[det*nsamples + sample];
                        double c2psi = cos(2*_psi);
                        double s2psi = sin(2*_psi);
                        // update AtD
                        AtD[idx*3 + 0] += 1.0   * d;
                        AtD[idx*3 + 1] += c2psi * d;
                        AtD[idx*3 + 2] += s2psi * d;
                         // update matrix
                        AtA[idx*9 + 0] += 1.0;
                        AtA[idx*9 + 1] += c2psi;
                        AtA[idx*9 + 2] += s2psi;
                        AtA[idx*9 + 3] += c2psi;
                        AtA[idx*9 + 4] += c2psi*c2psi;
                        AtA[idx*9 + 5] += c2psi*s2psi;
                        AtA[idx*9 + 6] += s2psi;
                        AtA[idx*9 + 7] += c2psi*s2psi;
                        AtA[idx*9 + 8] += s2psi*s2psi;
                    }

                }
            }

        }

    }

    int idx = int(CENTER_INDEX);
    printf("AtD: %le %le %le \n"  , AtD[idx*3 + 0], AtD[idx*3 + 1], AtD[idx*3 + 2]);
    printf("AtA: %le %le %le \n"  , AtA[idx*9 + 0], AtA[idx*9 + 1], AtA[idx*9 + 2]);
    printf("     %le %le %le \n"  , AtA[idx*9 + 3], AtA[idx*9 + 4], AtA[idx*9 + 5]);
    printf("     %le %le %le \n\n", AtA[idx*9 + 6], AtA[idx*9 + 7], AtA[idx*9 + 8]);
}

void
libmapping_get_IQU_from_matrices
(
    // input
    int map_nside, int map_size ,
    double AtA[], double AtD[], int pixels_in_the_map[],
    // output
    float I[], float Q[], float U[], float W[]
)
{
    // Setup map mask to translate matrices indexes into map pixels
    // Create healpixMaps that use the Python buffers
    // Note that all maps share the same mask!
    healpixMap *mask;
    mask = healpixMap_new();
    healpixMap_allocate(map_nside, mask);
    healpixMap_set_mask(mask,  pixels_in_the_map, map_size);
    // compute central pixel (dec=0, ra=PI)
    long CENTER_INDEX;
    ang2pix_ring(map_nside, M_PI_2, M_PI, &CENTER_INDEX);
    CENTER_INDEX = healpixMap_pix2idx(mask, CENTER_INDEX);
    for(long index = 0; index < map_size; index++)
    {
        int n = 3;
        int nrhs = 1;
        int lda = 3;
        int ipiv[3];
        int ldb = 3;
        int info ;

        double AtA_pix[n][lda];
        double AtD_pix[nrhs][ldb];
        // Setup AtA_pix in column major order
        AtA_pix[0][0] = AtA[0 + 9*index];
        AtA_pix[1][0] = AtA[1 + 9*index];
        AtA_pix[2][0] = AtA[2 + 9*index];

        AtA_pix[0][1] = AtA[3 + 9*index];
        AtA_pix[1][1] = AtA[4 + 9*index];
        AtA_pix[2][1] = AtA[5 + 9*index];

        AtA_pix[0][2] = AtA[6 + 9*index];
        AtA_pix[1][2] = AtA[7 + 9*index];
        AtA_pix[2][2] = AtA[8 + 9*index];

        AtD_pix[0][0] = AtD[0 + 3*index];
        AtD_pix[0][1] = AtD[1 + 3*index];
        AtD_pix[0][2] = AtD[2 + 3*index];

        double ii = 0.0;
        double qq = 0.0;
        double uu = 0.0;
        double hits = AtA_pix[0][0];

        //int map_pixel = healpixMap_pix2idx(mask, index);
        int map_pixel = index;
        // Solve AtA_pix x X = AtD_pix
        if(hits >= 3)
        {
            dgesv_(
                &n,
                &nrhs, &AtA_pix[0][0], &lda, ipiv,
                &AtD_pix[0][0], &ldb,
                &info);

            if(info != 0)
            {
                printf("The diagonal element of the triangular factor of A,\n");
                printf("AtA_pix(%i,%i) is zero, so that A is singular;\n", info, info);
                printf("the solution could not be computed.\n");
                ii = 0.0;
                qq = 0.0;
                uu = 0.0;
                hits = 0;
            }
            else
            {
                ii = AtD_pix[0][0];
                qq = AtD_pix[0][1];
                uu = AtD_pix[0][2];
            }
            //if(map_pixel == CENTER_INDEX) printf("I Q U: %le %le %le \n", ii, qq, uu);
        }
        I[map_pixel] += ii;
        Q[map_pixel] += qq;
        U[map_pixel] += uu;
        W[map_pixel] += hits;
    }
    int idx = int(CENTER_INDEX);
    printf("I Q U: %le %le %le \n", I[idx], Q[idx], U[idx]);
}
