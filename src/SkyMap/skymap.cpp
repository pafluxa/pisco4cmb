
#include "skymap.hpp"
#include <pointing.h>

#ifdef MAPPING_DEBUG
#include <stdio.h>
#endif

/** 
 * Declare prototype of dgesv_ function, which is linked at compile
 * time from LAPACK.
 */
extern "C" {
    void dgesv_
    (
        int *n, int *nrhs,  double *a,  int  *lda,
        int *ipivot, double *b, int *ldb, int *info
    );
}

SkyMap::SkyMap(int _nside) : hpx(_nside, RING, SET_NSIDE)
{
    nside = _nside;
    nPixels = 12 * _nside * _nside;
    allocate_buffers();
}

SkyMap::~SkyMap(void)
{
    deallocate_buffers();
}

void SkyMap::allocate_buffers(void)
{
    /* allocate accumulation matrices. */
    AtA = (double*)malloc(sizeof(double) * nPixels * 9);
    AtD = (double*)malloc(sizeof(double) * nPixels * 3);
    /* allocate space to solve the maps. */
	hitsMap = (int*)malloc(sizeof(int) * nPixels);
    stokesI = (float*)malloc(sizeof(float) * nPixels);
	stokesQ = (float*)malloc(sizeof(float) * nPixels);
	stokesU = (float*)malloc(sizeof(float) * nPixels);
}

void SkyMap::deallocate_buffers(void)
{
    free(AtA);
    free(AtD);
    free(hitsMap);
    free(stokesI);
    free(stokesQ);
    free(stokesU);
}

void SkyMap::accumulate_data
(
    int nSamples, float ra[], float dec[], float pa[],
    int nDetectors, float detPolAngles[],
    float data[]
)
{
    long pix;
    int det, sample, mapPixel;
    double _phi, _theta, _psi, pol_angle, d, twopsi, c2psi, s2psi;
    pointing ptgpix;
    #ifdef MAPPING_DEBUG
    int idx;
    double ra_pix, dec_pix; 
    double degr = 180.0 / M_PI;
    pointing ptg;
    long CENTER_INDEX;
    #endif
    for(det = 0; det < nDetectors; det++)
    {
        pol_angle = detPolAngles[det];
        for(sample = 0; sample < nSamples; sample++)
        {
            ptgpix.phi = ra[det * nSamples + sample];
            ptgpix.theta = M_PI_2 - dec[det * nSamples + sample];
            /* Passed arguments are counterclockwise on the sky, while
             * CMB convention requires clockwise positive. */
            _psi   = -(pol_angle + pa[det * nSamples + sample]);
            mapPixel = hpx.ang2pix(ptgpix);
            if(mapPixel >= 0)
            {
                d = data[det * nSamples + sample];
                twopsi = 2.0 * _psi;
                c2psi = cos(twopsi);
                s2psi = sin(twopsi);
                /* update AtD. */
                AtD[mapPixel * 3 + 0] += 1.0   * d;
                AtD[mapPixel * 3 + 1] += c2psi * d;
                AtD[mapPixel * 3 + 2] += s2psi * d;
                /* update matrix. */
                AtA[mapPixel * 9 + 0] += 1.0;
                AtA[mapPixel * 9 + 1] += c2psi;
                AtA[mapPixel * 9 + 2] += s2psi;
                AtA[mapPixel * 9 + 3] += c2psi;
                AtA[mapPixel * 9 + 4] += c2psi * c2psi;
                AtA[mapPixel * 9 + 5] += c2psi * s2psi;
                AtA[mapPixel * 9 + 6] += s2psi;
                AtA[mapPixel * 9 + 7] += c2psi * s2psi;
                AtA[mapPixel * 9 + 8] += s2psi * s2psi;
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

void SkyMap::solve_map(void)
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
    int mapPixel;
    double ii, qq, uu, hits;
    #ifdef MAPPING_DEBUG
    int idx;
    double ra_pix, dec_pix; 
    double degr = 180.0 / M_PI;
    pointing ptg;
    long CENTER_INDEX;
    #endif
    for(mapPixel = 0; mapPixel < nPixels; mapPixel++)
    {
        /* assemble AtA_pix in column major order. */
        AtA_pix[0][0] = AtA[0 + 9 * mapPixel];
        AtA_pix[1][0] = AtA[1 + 9 * mapPixel];
        AtA_pix[2][0] = AtA[2 + 9 * mapPixel];
        AtA_pix[0][1] = AtA[3 + 9 * mapPixel];
        AtA_pix[1][1] = AtA[4 + 9 * mapPixel];
        AtA_pix[2][1] = AtA[5 + 9 * mapPixel];
        AtA_pix[0][2] = AtA[6 + 9 * mapPixel];
        AtA_pix[1][2] = AtA[7 + 9 * mapPixel];
        AtA_pix[2][2] = AtA[8 + 9 * mapPixel];
        /* assemble AtA_pix in column major order. */
        AtD_pix[0][0] = AtD[0 + 3 * mapPixel];
        AtD_pix[0][1] = AtD[1 + 3 * mapPixel];
        AtD_pix[0][2] = AtD[2 + 3 * mapPixel];
        /* initialize variables that will be overwritten. */
        ii = 0.0;
        qq = 0.0;
        uu = 0.0;
        /* extract hits *before* inverting the matrix. */
        hits = int(AtA_pix[0][0]);
        /* solve matrix-vector system 
         *     AtA_pix X = AtD_pix ,
         * using dgesv. 
         *
         * also, check the pixel has at least 3 hits. */
        if(hits >= 3)
        {
            dgesv_
            (
                &n,
                &nrhs, &AtA_pix[0][0], &lda, ipiv,
                &AtD_pix[0][0], &ldb,
                &info
            );
            ii = AtD_pix[0][0];
            qq = AtD_pix[0][1];
            uu = AtD_pix[0][2];
            /* handle case where dgesv_ fails to solve the system. */
            if(info != 0)
            {
                std::cerr << "The diagonal element of the";
                std::cerr << " triangular factor of A,\n";
                std::cerr 
                    << "AtA_pix(" << info << "," << info 
                    << " is zero, so that A is singular;" 
                    << "the solution could not be computed" 
                    << std::endl;
                /* ill-conditioned pixels are not taken into account as
                 * valid pixels, so the hits are set to zero. */
                hits = 0;
            }
        }
        /* flag pixels with less than 3 hits as INSUFFCOV (insufficient
         * coverage flag). Hits are not touched. */
        else 
        {
            std::cerr 
                << "WARNING: pixel " << mapPixel 
                << "has less than 3 hits." << std::endl;
            ii = SkyMap::INSUFFCOV;
            qq = SkyMap::INSUFFCOV;
            uu = SkyMap::INSUFFCOV;
        }
        stokesI[mapPixel] = ii;
        stokesQ[mapPixel] = qq;
        stokesU[mapPixel] = uu;
        hitsMap[mapPixel] = hits;
    }
    #ifdef MAPPING_DEBUG
    idx = int(CENTER_INDEX);
    printf("I Q U: %le %le %le \n", stokesI[idx], stokesQ[idx], stokesU[idx]);
    #endif
}
