#include "mapper.hpp"
#include <pointing.h>
#include <cstring>

#include "mkl_lapacke.h"

Mapper::Mapper(int _nside) : hpx(_nside, RING, SET_NSIDE)
{
    nside = _nside;
    nPixels = 12 * _nside * _nside;
    allocate_buffers();
}

Mapper::~Mapper(void)
{
    deallocate_buffers();
}

void Mapper::allocate_buffers(void)
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

void Mapper::deallocate_buffers(void)
{
    free(AtA);
    free(AtD);
    free(hitsMap);
    free(stokesI);
    free(stokesQ);
    free(stokesU);
}

void Mapper::accumulate_data(int nSamples, 
    const float* ra, const float* dec, const float *pa, 
    float detPolAngle,
    float *detdata
)
{
    long pix;
    long sample;
    int mapPixel;
    double psi;
    double d;
    double twopsi;
    double c2psi;
    double s2psi;
    pointing ptgpix;

    for(sample = 0; sample < nSamples; sample++)
    {
        ptgpix.phi = double(ra[sample]);
        ptgpix.theta = M_PI_2 - double(dec[sample]);
        /* Passed arguments are counterclockwise on the sky, while
         * CMB convention requires clockwise positive. */
        psi = double(detPolAngle + pa[sample]);
        mapPixel = hpx.ang2pix(ptgpix);
        d = detdata[sample];
        twopsi = 2.0 * psi;
        c2psi = cos(twopsi);
        s2psi = sin(twopsi);
        /* update AtD. */
        AtD[mapPixel * 3 + 0] += 1.0   * d;
        AtD[mapPixel * 3 + 1] += c2psi * d;
        AtD[mapPixel * 3 + 2] += s2psi * d;
        /* update AtA. */
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

void Mapper::solve_map(void)
{
    /*****************************/
    /* stuff needed for dgesv. */
    int n = 3;
    int nrhs = 1;
    int lda = 3;
    int ipiv[3];
    int ldb = 1;
    int info;
    double AtA_pix[n * lda];
    double AtD_pix[n * nrhs];
    /*****************************/
    long index;
    int mapPixel;
    double ii, qq, uu, hits;
    for(mapPixel = 0; mapPixel < nPixels; mapPixel++)
    {
        /* assemble AtA_pix in column major order. */
        AtA_pix[0] = AtA[0 + 9 * mapPixel];
        AtA_pix[1] = AtA[1 + 9 * mapPixel];
        AtA_pix[2] = AtA[2 + 9 * mapPixel];
        AtA_pix[3] = AtA[3 + 9 * mapPixel];
        AtA_pix[4] = AtA[4 + 9 * mapPixel];
        AtA_pix[5] = AtA[5 + 9 * mapPixel];
        AtA_pix[6] = AtA[6 + 9 * mapPixel];
        AtA_pix[7] = AtA[7 + 9 * mapPixel];
        AtA_pix[8] = AtA[8 + 9 * mapPixel];
        /* assemble AtA_pix in column major order. */
        AtD_pix[0] = AtD[0 + 3 * mapPixel];
        AtD_pix[1] = AtD[1 + 3 * mapPixel];
        AtD_pix[2] = AtD[2 + 3 * mapPixel];
        /* initialize variables that will be overwritten. */
        ii = 0.0;
        qq = 0.0;
        uu = 0.0;
        /* extract hits *before* inverting the matrix. */
        hits = int(AtA_pix[0]);
         /* check the pixel has at least 3 hits. */
        if(hits >= 3)
        {
            /* solve matrix-vector system 
            *
            *     AtA_pix X = AtD_pix ,
            * 
            * using dgesv, which is OVERKILL.
            *
            */
            /*
            dgesv_
            (
                &n,
                &nrhs, &AtA_pix[0], &lda, ipiv,
                &AtD_pix[0], &ldb,
                &info
            );
            */
            info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, AtA_pix, lda, ipiv, AtD_pix, ldb);
            if(info == 0) 
            {
                ii = AtD_pix[0];
                qq = AtD_pix[1];
                uu = AtD_pix[2];
            }
            /* handle case where dgesv_ fails to solve the system. */
            else
            {
                std::cerr << "error on pixel " << mapPixel << std::endl;
                int idx = mapPixel;
                printf("AtA: %le %le %le \n"  , AtA[idx*9 + 0], AtA[idx*9 + 1], AtA[idx*9 + 2]);
                printf("     %le %le %le \n"  , AtA[idx*9 + 3], AtA[idx*9 + 4], AtA[idx*9 + 5]);
                printf("     %le %le %le \n\n", AtA[idx*9 + 6], AtA[idx*9 + 7], AtA[idx*9 + 8]);
                printf("AtD: %le \n"  , AtD[idx * 3 + 0]);
                printf("     %le \n"  , AtD[idx * 3 + 1]);
                printf("     %le \n\n", AtD[idx * 3 + 2]);
                ii = Mapper::ILLCOND;
                qq = Mapper::ILLCOND;
                uu = Mapper::ILLCOND;
                /* ill-conditioned pixels are not taken into account as
                 * valid pixels, so the hits are set to zero. */
                hits = 0;
            }
        }
        /* flag pixels with less than 3 hits as INSUFFCOV (insufficient
         * coverage flag). Hits are not touched. */
        else 
        {
            ii = Mapper::INSUFFCOV;
            qq = Mapper::INSUFFCOV;
            uu = Mapper::INSUFFCOV;
        }
        stokesI[mapPixel] = ii;
        stokesQ[mapPixel] = qq;
        stokesU[mapPixel] = uu;
        hitsMap[mapPixel] = hits;
    }
}
