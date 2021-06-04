#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>
#include <healpix_base.h>
#include <pointing.h>

#include "mapping_routines.h"

extern "C" {
     void dgesv_(int *n, int *nrhs,  double *a,  int  *lda,
           int *ipivot, double *b, int *ldb, int *info) ;
}

void
libmapping_project_data_to_matrices
(
    // input
    int nsamples, 
    float ra[], float dec[], float pa[],
    int ndets, double pol_angles[],
    float data[] , 
    int map_nside,
    // output
    double AtA[], double AtD[]
)
{
    
    long pix;
    int det, sample, map_pixel;
    double _phi, _theta, _psi, pol_angle, d, twopsi, c2psi, s2psi;
    Healpix_Base hpx(map_nside, RING, SET_NSIDE);
    pointing ptgpix;
    #ifdef MAPPING_DEBUG
    int idx;
    double ra_pix, dec_pix; 
    double degr = 180.0 / M_PI;
    pointing ptg;
    long CENTER_INDEX;
    #endif
    for(det = 0; det < ndets; det++)
    {
        pol_angle = pol_angles[det];
        for(sample = 0; sample < nsamples; sample++)
        {
            ptgpix.phi = ra[det * nsamples + sample];
            ptgpix.theta = M_PI_2 - dec[det * nsamples + sample];
            /* Passed arguments are counterclockwise on the sky, while
             * CMB convention requires clockwise positive. */
            _psi   = -(pol_angle + pa[det * nsamples + sample]);
            map_pixel = hpx.ang2pix(ptgpix);
            if(map_pixel >= 0)
            {
                d = data[det * nsamples + sample];
                twopsi = 2.0 * _psi;
                c2psi = cos(twopsi);
                s2psi = sin(twopsi);
                /* update AtD. */
                AtD[map_pixel * 3 + 0] += 1.0   * d;
                AtD[map_pixel * 3 + 1] += c2psi * d;
                AtD[map_pixel * 3 + 2] += s2psi * d;
                /* update matrix. */
                AtA[map_pixel * 9 + 0] += 1.0;
                AtA[map_pixel * 9 + 1] += c2psi;
                AtA[map_pixel * 9 + 2] += s2psi;
                AtA[map_pixel * 9 + 3] += c2psi;
                AtA[map_pixel * 9 + 4] += c2psi * c2psi;
                AtA[map_pixel * 9 + 5] += c2psi * s2psi;
                AtA[map_pixel * 9 + 6] += s2psi;
                AtA[map_pixel * 9 + 7] += c2psi * s2psi;
                AtA[map_pixel * 9 + 8] += s2psi * s2psi;
            }
        }
    }
    #ifdef MAPPING_DEBUG
    // compute central pixel (dec=0, ra=PI)
    ptgpix.phi = M_PI;
    ptgpix.theta = M_PI_2;
    CENTER_INDEX = hpx.ang2pix(ptgpix);
    idx = int(CENTER_INDEX);
    ptg = hpx.pix2ang(idx);
    ra_pix = ptg.phi;
    dec_pix = M_PI_2 - ptg.theta;
    printf("ra_pix %lf dec_pix %lf \n", ra_pix * degr, dec_pix * degr);
    printf("AtD: %le %le %le \n"  , AtD[idx*3 + 0], AtD[idx*3 + 1], AtD[idx*3 + 2]);
    printf("AtA: %le %le %le \n"  , AtA[idx*9 + 0], AtA[idx*9 + 1], AtA[idx*9 + 2]);
    printf("     %le %le %le \n"  , AtA[idx*9 + 3], AtA[idx*9 + 4], AtA[idx*9 + 5]);
    printf("     %le %le %le \n\n", AtA[idx*9 + 6], AtA[idx*9 + 7], AtA[idx*9 + 8]);
    #endif
}

void
libmapping_get_IQU_from_matrices
(
    // input
    int map_nside,
    double AtA[], double AtD[],
    // output
    float I[], float Q[], float U[], float W[]
)
{
    /*****************************/
    /* stuff needed for dgesv. */
    int n = 3;
    int nrhs = 1;
    int lda = 3;
    int ipiv[3];
    int ldb = 3;
    int info;
    double AtA_pix[n][lda];
    double AtD_pix[nrhs][ldb];
    /*****************************/
    long index;
    int map_pixel;
    int map_size = 12 * map_nside * map_nside;
    double ii, qq, uu, hits;
    #ifdef MAPPING_DEBUG
    int idx;
    double ra_pix, dec_pix; 
    double degr = 180.0 / M_PI;
    pointing ptg;
    long CENTER_INDEX;
    #endif
    for(map_pixel = 0; map_pixel < map_size; map_pixel++)
    {
        /* assemble AtA_pix in column major order. */
        AtA_pix[0][0] = AtA[0 + 9 * map_pixel];
        AtA_pix[1][0] = AtA[1 + 9 * map_pixel];
        AtA_pix[2][0] = AtA[2 + 9 * map_pixel];
        AtA_pix[0][1] = AtA[3 + 9 * map_pixel];
        AtA_pix[1][1] = AtA[4 + 9 * map_pixel];
        AtA_pix[2][1] = AtA[5 + 9 * map_pixel];
        AtA_pix[0][2] = AtA[6 + 9 * map_pixel];
        AtA_pix[1][2] = AtA[7 + 9 * map_pixel];
        AtA_pix[2][2] = AtA[8 + 9 * map_pixel];
        /* assemble AtA_pix in column major order. */
        AtD_pix[0][0] = AtD[0 + 3 * map_pixel];
        AtD_pix[0][1] = AtD[1 + 3 * map_pixel];
        AtD_pix[0][2] = AtD[2 + 3 * map_pixel];
        /* initialize variables that will be overwritten. */
        ii = 0.0;
        qq = 0.0;
        uu = 0.0;
        /* extract hits _before_ inverting the matrix. */
        hits = AtA_pix[0][0];
        /* solve matrix-vector system 
         *     AtA_pix X = AtD_pix ,
         * using dgesv. check the pixel has at least 3 hits. */
        if(hits >= 3)
        {
            dgesv_(
                &n,
                &nrhs, &AtA_pix[0][0], &lda, ipiv,
                &AtD_pix[0][0], &ldb,
                &info);
            if(info != 0)
            {
                std::cerr << "The diagonal element of the triangular factor of A,\n";
                std::cerr << "AtA_pix(" << info << "," << info 
                    << " is zero, so that A is singular;" 
                    << "the solution could not be computed" 
                    << std::endl;
                /* flag ill-conditioned pixels with ILLCONDITIONED. */
                ii = MAPPING_ILLCONDITIONED;
                qq = MAPPING_ILLCONDITIONED;
                uu = MAPPING_ILLCONDITIONED;
                /* ill-conditioned pixels are not taken into account as
                 * valid pixels, so we set the hits to zero. */
                hits = 0;
            }
            else
            {
                ii = AtD_pix[0][0];
                qq = AtD_pix[0][1];
                uu = AtD_pix[0][2];
            }
        }
        /* pixels hit less than 3 times flagged as BADPOLCOVERAGE. */
        else 
        {
            ii = MAPPING_BADPOLCOVERAGE;
            qq = MAPPING_BADPOLCOVERAGE;
            uu = MAPPING_BADPOLCOVERAGE;
        }
        I[map_pixel] += ii;
        Q[map_pixel] += qq;
        U[map_pixel] += uu;
        W[map_pixel] += hits;
    }
    #ifdef MAPPING_DEBUG
    idx = int(CENTER_INDEX);
    printf("I Q U: %le %le %le \n", I[idx], Q[idx], U[idx]);
    #endif
}
