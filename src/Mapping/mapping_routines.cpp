#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <chealpix.h>
#include <omp.h>

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
    #ifdef MAPPING_DEBUG
    // compute central pixel (dec=0, ra=PI)
    long CENTER_INDEX;
    ang2pix_ring(map_nside, 
        M_PI_2 - 0.3 / 32.0, M_PI - 0.3 / 32.0, &CENTER_INDEX);
    #endif
    for(int det = 0; det < ndets; det++)
    {
        double pol_angle = pol_angles[det];
        for(int sample = 0; sample < nsamples; sample++)
        {
            double _phi, _theta, _psi;
            _phi   = ra[det * nsamples + sample];
            _theta = M_PI_2 - dec[det * nsamples + sample];
            // Passed arguments are counterclockwise on the sky
            // CMB requires clockwise
            _psi   = -(pol_angle + pa[det * nsamples + sample]);
            ang2pix_ring(map_nside, _theta, _phi, &pix);
            if(pix >= 0)
            {
                double d = data[det*nsamples + sample];
                double c2psi = cos(2.0*_psi);
                double s2psi = sin(2.0*_psi);
                // update AtD
                AtD[pix*3 + 0] += 1.0   * d;
                AtD[pix*3 + 1] += c2psi * d;
                AtD[pix*3 + 2] += s2psi * d;
                 // update matrix
                AtA[pix*9 + 0] += 1.0;
                AtA[pix*9 + 1] += c2psi;
                AtA[pix*9 + 2] += s2psi;
                AtA[pix*9 + 3] += c2psi;
                AtA[pix*9 + 4] += c2psi*c2psi;
                AtA[pix*9 + 5] += c2psi*s2psi;
                AtA[pix*9 + 6] += s2psi;
                AtA[pix*9 + 7] += c2psi*s2psi;
                AtA[pix*9 + 8] += s2psi*s2psi;
            }
        }
    }
    #ifdef MAPPING_DEBUG
    int idx = int(CENTER_INDEX);
    double ra_pix, dec_pix; 
    double degr = 180.0 / M_PI;
    pix2ang_ring(map_nside, idx, &dec_pix, &ra_pix);
    dec_pix = M_PI_2 - dec_pix;
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
    long index;
    int map_size = 12 * map_nside * map_nside;
    #ifdef MAPPING_DEBUG
    // compute central pixel (dec=0, ra=PI)
    long CENTER_INDEX;
    ang2pix_ring(map_nside, 
        M_PI_2 - 0.3 / 32.0, M_PI - 0.3 / 32.0, &CENTER_INDEX);
    #endif
    for(index = 0; index < map_size; index++)
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
        }
        I[map_pixel] += ii;
        Q[map_pixel] += qq;
        U[map_pixel] += uu;
        W[map_pixel] += hits;
    }
    #ifdef MAPPING_DEBUG
    int idx = int(CENTER_INDEX);
    printf("I Q U: %le %le %le \n", I[idx], Q[idx], U[idx]);
    #endif
}
