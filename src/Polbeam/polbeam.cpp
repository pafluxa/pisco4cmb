#include "polbeam.hpp"
//using namespace PolarizedBeam;

#include <complex>
#include <cstdlib>

// to make use of Healpix_base pointing
#include <pointing.h>

#ifdef POLBEAM_DUMPBEAMS
#include <fstream>
#endif

PolBeam::PolBeam(int _nside, long _nPixels)
{
    nside = _nside;
    nPixels = _nPixels;
    // epsilon is set to zero for now
    epsilon = 0.0;
    
    hpxBase = new Healpix_Base(nside, RING, SET_NSIDE);
    pointing p = hpxBase->pix2ang(nPixels);
    rhoMax = p.theta;
    
    alloc_buffers();
}

PolBeam::~PolBeam()
{
    free_buffers();
}

void PolBeam::alloc_buffers()
{
    size_t buffSize = sizeof(float)*nPixels;
    aBeams[0] = (float*)malloc(buffSize);
    aBeams[1] = (float*)malloc(buffSize);
    aBeams[2] = (float*)malloc(buffSize);
    aBeams[3] = (float*)malloc(buffSize);
    aBeams[4] = (float*)malloc(buffSize);
    aBeams[5] = (float*)malloc(buffSize);

    bBeams[0] = (float*)malloc(buffSize);
    bBeams[1] = (float*)malloc(buffSize);
    bBeams[2] = (float*)malloc(buffSize);
    bBeams[3] = (float*)malloc(buffSize);
    bBeams[4] = (float*)malloc(buffSize);
    bBeams[5] = (float*)malloc(buffSize);

    Ia = (float*)malloc(buffSize);
    Qa = (float*)malloc(buffSize);
    Ua = (float*)malloc(buffSize);
    Va = (float*)malloc(buffSize);

    Ib = (float*)malloc(buffSize);
    Qb = (float*)malloc(buffSize);
    Ub = (float*)malloc(buffSize);
    Vb = (float*)malloc(buffSize);
}

void PolBeam::free_buffers()
{
    free(aBeams[0]);
    free(aBeams[1]);
    free(aBeams[2]);
    free(aBeams[3]);
    free(aBeams[4]);
    free(aBeams[5]);

    free(bBeams[0]);
    free(bBeams[1]);
    free(bBeams[2]);
    free(bBeams[3]);
    free(bBeams[4]);
    free(bBeams[5]);

    free(Ia);
    free(Qa);
    free(Ua);
    free(Va);

    free(Ib);
    free(Qb);
    free(Ub);
    free(Vb);
}

void
PolBeam::beam_from_fields
(
    char polFlag,
    double* magEco, double* phaseEco,
    double* magEcx, double* phaseEcx
)
/*
 * Computes the beamsor of a single PSB detector a or b using the
 * electric field power density (EFPD) of an antenna. This routine
 * is based on the formalism found in Rosset el at. 2010, with significant
 * contribution from Michael K. Brewer.
 *
 * The field must be specified as 4 arrays containing the magnitude and
 * phase of the EFPD along the co and cross polar directions of the
 * antenna basis.
 *
 * \param polFlag: set to 'a' ('b') on the particular detector beam of
 *        the PSB that is being loaded.
 * \param magEco: array with co-polar EFPD magnitude.
 * \param phaseEco: array with the phase of co-polar EFPD.
 * \param magEcx: array with cross-polar EFPD magnitude along.
 * \param phaseEcx: array with the phase of cross-polar EFPD.
 *
 */
{
    float *I;
    float *Q;
    float *U;
    float *V;
    std::complex<double> Eco;
    std::complex<double> Ecx;

    if(polFlag == 'a')
    {
        I = Ia;
        Q = Qa;
        U = Ua;
        V = Va;
    }
    if(polFlag == 'b')
    {
        I = Ib;
        Q = Qb;
        U = Ub;
        V = Vb;
    }
    for(int i = 0; i < nPixels; i++)
    {
        // component along x basis vector
        Eco = std::polar(magEco[i], phaseEco[i]);
        // component along y basis vector
        Ecx = std::polar(magEcx[i], phaseEcx[i]);
        // compute \tilde{I}
        I[i] = std::norm(Eco) + std::norm(Ecx);
        // compute \tilde{Q}
        Q[i] = std::norm(Eco) - std::norm(Ecx);
        // compute \tilde{U}
        U[i] = 2*std::real(Eco*std::conj(Ecx));
        // compute \tilde{V}
        V[i] = -2*std::imag(Eco*std::conj(Ecx));
    }
}

void PolBeam::build_beams(void)
{
    #ifdef POLBEAM_DUMPBEAMS
    std::string dumpfilepatha = "dump_detector_a.txt";
    std::string dumpfilepathb = "dump_detector_b.txt";
    std::ofstream dumpfilea(dumpfilepatha);
    std::ofstream dumpfileb(dumpfilepathb);
    std::cerr << "INFO: DUMPING I, Q, U and V beams to "
              << dumpfilepatha << " and " << dumpfilepathb << std::endl;
    #endif
    
    long i;
    for(i = 0; i < nPixels; i++)
    {
        aBeams[0][i] = Ia[i] + epsilon*Ib[i];
        aBeams[1][i] = Qa[i] - epsilon*Qb[i];
        aBeams[2][i] = Ua[i] - epsilon*Ub[i];
        aBeams[3][i] = Ua[i] - epsilon*Ub[i];
        aBeams[4][i] = Qa[i] - epsilon*Qb[i];
        aBeams[5][i] = Va[i] + epsilon*Vb[i];
        #ifdef POLBEAM_DUMPBEAMS
        dumpfilea 
                 << aBeams[0][i] << " "
                 << aBeams[1][i] << " "
                 << aBeams[2][i] << " "
                 << aBeams[3][i] << " "
                 << aBeams[4][i] << " "
                 << aBeams[5][i] << std::endl;
        #endif
        bBeams[0][i] = Ib[i] + epsilon*Ia[i];
        bBeams[1][i] = Qb[i] - epsilon*Qa[i];
        bBeams[2][i] = Ub[i] - epsilon*Ua[i];
        bBeams[3][i] = Ub[i] - epsilon*Ua[i];
        bBeams[4][i] = Qb[i] - epsilon*Qa[i];
        bBeams[5][i] = Vb[i] + epsilon*Va[i];
        #ifdef POLBEAM_DUMPBEAMS
        dumpfileb 
                 << bBeams[0][i] << " "
                 << bBeams[1][i] << " "
                 << bBeams[2][i] << " "
                 << bBeams[3][i] << " "
                 << bBeams[4][i] << " "
                 << bBeams[5][i] << std::endl;
        #endif
    }
    #ifdef POLBEAM_DUMPBEAMS
    dumpfilea.close();
    dumpfileb.close();
    #endif
}

void PolBeam::make_unpol_gaussian_elliptical_beams(
    double fwhmx,
    double fwhmy,
    double phi0)
/*
 *  Creates a Gaussian Elliptical Beam as a HEALPix grid.
 *
 *  Authors: Michael K. Brewer (CLASS collaboration, 2018)
 *           C++ adaptation by Pedro Fluxa (PUC, 2019)
 *  input:
 *
 *      fwhm_x  : Full Width at Half Maximum, in the x (South-North)
 *                direction if phi_0=0 (degrees)
 *      fwhm_y  : Full Width at Half Maximum, in the y (East-West)
 *                direction if phi=0, (degrees)
 *      phi_0   : Angle between Noth-South direction and x-axis of the
 *                ellipse (degrees). phi_0 increases clockwise towards
 *                East.
 *
 *  In this routine, the same elliptical gaussian is assigned to
 *  I,Q, U and V beams. All beams are normalized so that
 *
 *  \int_{4\pi} b(\rho,\sigma) d\Omega = 1.0
 *
 */
{
    // temporal buffers to store fields
    size_t buffSize = sizeof(double)*nPixels;
    double* magEco = (double*)malloc(buffSize);
    double* magEcx = (double*)malloc(buffSize);
    double* phsEco = (double*)malloc(buffSize);
    double* phsEcx = (double*)malloc(buffSize);

    // Convert FWHM in degres to sigma, in radians
    double deg2rad = M_PI/180.0;
    // From https://en.wikipedia.org/wiki/Full_width_at_half_maximum:
    //
    //     FWHM = 2 \sqrt{2 \ln{2}} \sigma
    //
    // where \sigma is the standard deviation.
    // (2 \sqrt{2 \ln{2}}) ~ 2.35482
    double sigma_x = (deg2rad*fwhmx)/2.35482;
    double sigma_y = (deg2rad*fwhmy)/2.35482;
    //Convert phi_0 to radians
    double phi_0 = deg2rad * phi0;
    // Compute coefficients to rotate the ellipse.
    double a =  (cos(phi_0)*cos(phi_0))/(2*sigma_x*sigma_x) +
                (sin(phi_0)*sin(phi_0))/(2*sigma_y*sigma_y);
    double b = -(sin(2*phi_0) )/(4*sigma_x*sigma_x) +
                (sin(2*phi_0 ))/(4*sigma_y*sigma_y);
    double c =  (sin(phi_0)*sin(phi_0))/(2*sigma_x*sigma_x) +
                (cos(phi_0)*cos(phi_0))/(2*sigma_y*sigma_y);
    double invsq2 = 1./sqrt(2);
    double rho,sig,val;
    for(int bpix = 0; bpix < nPixels; bpix++)
    {
        pointing bp = hpxBase->pix2ang(bpix);
        rho = bp.theta;
        sig = bp.phi;

        val = exp(-(a*cos(sig)*cos(sig)
                  + 2*b*cos(sig)*sin(sig)
                  + c*sin(sig)*sin(sig))*rho*rho);
        if(val < 1e-6)
            val = 0.0;
        // co-polarized beam!
        magEco[bpix] = sqrt(val);
        magEcx[bpix] = 0.0;
        phsEco[bpix] = 0.0;
        phsEcx[bpix] = 0.0;
    }
    // build beams from dummy fields
    this->beam_from_fields('a', magEco, phsEco, magEcx, phsEcx);
    this->beam_from_fields('b', magEco, phsEco, magEcx, phsEcx);
    // free temporal storage
    free(magEco);
    free(phsEco);
    free(magEcx);
    free(phsEcx);
}
