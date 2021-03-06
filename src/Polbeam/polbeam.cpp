#include <complex>
#include <cstdlib>
#include <cstring>
// to make use of Healpix_base pointing
#include <pointing.h>

#include "polbeam.hpp"

#ifdef POLBEAM_DUMPBEAMS
#include <fstream>
#endif

PolBeam::PolBeam
(
    int _nside, long _nPixels, 
    double _epsilon, char _enabledDets 
) : hpxBase(_nside, RING, SET_NSIDE)
/*
 * PolBeam constructor.
 * 
 * \param _nside: NSIDE parameter of internal Healpix maps.
 * \param _nPixels: number of pixels in every map.
 * \param _epsilon: 1 - polarization efficiency of detectors.
 * \param _enabledDets: set to 'a' or 'b' to use a single detector. Use
 * 'p' to use both detectors in the PSB.
 */
{
    nside = _nside;
    nPixels = _nPixels;
    epsilon = _epsilon;
    enabledDets = _enabledDets;
    std::cerr << "PolBeam::constructor: ";
    std::cerr << "building polbeam with dets " << enabledDets;
    std::cerr <<" detectors." << std::endl;
    pointing p = hpxBase.pix2ang(nPixels);
    rhoMax = p.theta;
    alloc_buffers();
}

PolBeam::~PolBeam()
/* Destructor.
 * 
 * \note: frees all buffers before destruction.
 */
{
    free_buffers();
}

void PolBeam::alloc_buffers()
/* 
 * Allocate buffers to store beams.
 *
 */
{
    size_t buffSize = sizeof(float) * nPixels;
    std::cerr << "PolBeam::alloc_buffers: ";
    std::cerr << "buffer size is " << buffSize << " bytes (";
    std::cerr << nPixels << " pixels)" << std::endl;
    std::cerr << "PolBeam::alloc_buffers: ";
    std::cerr << "allocating beam buffers for detector a." << std::endl;
    aBeams[0] = (float*)malloc(buffSize);
    aBeams[1] = (float*)malloc(buffSize);
    aBeams[2] = (float*)malloc(buffSize);
    aBeams[3] = (float*)malloc(buffSize);
    Ia = (float*)malloc(buffSize);
    Qa = (float*)malloc(buffSize);
    Ua = (float*)malloc(buffSize);
    Va = (float*)malloc(buffSize); 
    std::cerr << "PolBeam::alloc_buffers: ";
    std::cerr << "allocating beam buffers for detector b." << std::endl;
    bBeams[0] = (float*)malloc(buffSize);
    bBeams[1] = (float*)malloc(buffSize);
    bBeams[2] = (float*)malloc(buffSize);
    bBeams[3] = (float*)malloc(buffSize);
    Ib = (float*)malloc(buffSize);
    Qb = (float*)malloc(buffSize);
    Ub = (float*)malloc(buffSize);
    Vb = (float*)malloc(buffSize);
    /* Set memory of beams to zero by default. */
    std::memset(Ia, 0, buffSize);
    std::memset(Ib, 0, buffSize);
    std::memset(Qa, 0, buffSize);
    std::memset(Qb, 0, buffSize);
    std::memset(Ua, 0, buffSize);
    std::memset(Ub, 0, buffSize);
    std::memset(Va, 0, buffSize);
    std::memset(Vb, 0, buffSize);
}

void PolBeam::free_buffers()
/* 
 * Routine to free all buffers before deleting the object.
 */
{
    std::cerr << "PolBeam::free_buffers: ";
    std::cerr << "freeing beam buffers for detector a." << std::endl;
    free(Ia);
    free(Qa);
    free(Ua);
    free(Va);
    free(aBeams[0]);
    free(aBeams[1]);
    free(aBeams[2]);
    free(aBeams[3]);
    std::cerr << "PolBeam::free_buffers: ";
    std::cerr << "freeing beam buffers for detector b." << std::endl;
    free(Ib);
    free(Qb);
    free(Ub);
    free(Vb);
    free(bBeams[0]);
    free(bBeams[1]);
    free(bBeams[2]);
    free(bBeams[3]);
}

void
PolBeam::beam_from_fields
(
    char polFlag,
    double* magEco, double* phaseEco,
    double* magEcx, double* phaseEcx
)
/*
 * Computes the polarized beam of a single PSB detector (a or b) using 
 * the electric field power density (EFPD) of an antenna. This routine
 * is based on the formalism found in Rosset el at. 2010, with heavy
 * contributions from Michael K. Brewer.
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
        std::cerr << "PolBeam::beam_from_fields: ";
        std::cerr << "building (I,Q,U,V) beams from fields for detector a.";
        std::cerr << std::endl;
        I = Ia;
        Q = Qa;
        U = Ua;
        V = Va;
    }
    if(polFlag == 'b')
    {
        std::cerr << "PolBeam::beam_from_fields: ";
        std::cerr << "building (I,Q,U,V) beams from fields for detector b.";
        std::cerr << std::endl;
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
        I[i] = float(std::norm(Eco) + std::norm(Ecx));
        // compute \tilde{Q}
        Q[i] = float(std::norm(Eco) - std::norm(Ecx));
        // compute \tilde{U}
        U[i] = float(2*std::real(Eco*std::conj(Ecx)));
        // compute \tilde{V}
        V[i] = float(-2*std::imag(Eco*std::conj(Ecx)));
    }
}

void PolBeam::build(int nsideSky)
/*
 * Build the actual polarized beams from tilde I, Q, U and V. These 
 * beams take into account possible cross-talk produced by finite 
 * polarization efficiency of detectors.
 * 
 * The NSIDE parameter of the sky is needed to calculate a normalization 
 * factor such that the time ordered data generated by the convolution
 * process keeps being in brightness temperature units (Kelvin).
 * 
 * \param nsideSky: NSIDE parameter of the sky map that will be used
 * in the convolution process.
 *
 */
{
    std::cerr << "PolBeam::build: ";
    std::cerr << "building polarized beams." << std::endl;
    // compute solid angle
    double sumIa = 0;
    double sumIb = 0;
    for(int i = 0; i < nPixels; i++) 
    {
        if(enabledDets == 'a' || enabledDets == 'p')
        {
            sumIa += Ia[i];
        }
        if(enabledDets == 'b' || enabledDets == 'p')
        {        
            sumIb += Ib[i];
        }
    }   
    // compute normalization factor
    double ocmp = nsideSky / double(nside);
    std::cerr << "PolBeam::build: ";
    std::cerr << "solid angle of detector a = ";
    std::cerr << sumIa * ocmp << " strad" << std::endl;
    std::cerr << "PolBeam::build: ";
    std::cerr << "solid angle of detector b = "; 
    std::cerr << sumIb * ocmp << " strad" << std::endl;
    ocmp = ocmp * ocmp;
    double aNorm = sumIa * ocmp;
    double bNorm = sumIb * ocmp;
    // hack to avoid the beam from blowing up
    if(enabledDets == 'a')
    {
        bNorm += 1e-8;
    }
    if(enabledDets == 'b')
    {
        aNorm += 1e-8;
    }
    #ifdef POLBEAM_DUMPBEAMS
    std::string dumpfilepatha = "dump_detector_a.txt";
    std::string dumpfilepathb = "dump_detector_b.txt";
    std::ofstream dumpfilea;
    std::ofstream dumpfileb;
    if(enabledDets == 'a' || enabledDets == 'p')
    {
        dumpfilea.open(dumpfilepatha);
    }
    if(enabledDets == 'b' || enabledDets == 'p')
    {        
        dumpfileb.open(dumpfilepathb);
    }
    std::cerr << "PolBeam::build: ";
    std::cerr << "dumping beams to ./dump_detector_a.txt";
    std::cerr << "and ./dump_detector_b.txt";
    std::cerr << std::endl;
    #endif
    std::cerr << "PolBeam::build: ";
    std::cerr << "building polarized beams.";
    std::cerr << std::endl;
    for(int i = 0; i < nPixels; i++)
    {
        aBeams[0][i] = (Ia[i] + epsilon * Ib[i]) / aNorm;
        aBeams[1][i] = (Qa[i] - epsilon * Qb[i]) / aNorm;
        aBeams[2][i] = (Ua[i] - epsilon * Ub[i]) / aNorm;
        aBeams[3][i] = (Va[i] + epsilon * Vb[i]) / aNorm;
        
        bBeams[0][i] = (Ib[i] + epsilon * Ia[i]) / bNorm;
        bBeams[1][i] = (Qb[i] - epsilon * Qa[i]) / bNorm;
        bBeams[2][i] = (Ub[i] - epsilon * Ua[i]) / bNorm;
        bBeams[3][i] = (Vb[i] + epsilon * Va[i]) / bNorm;
        #ifdef POLBEAM_DUMPBEAMS
        /* dump data for detector a. */
        dumpfilea
        << aBeams[0][i] << " "
        << aBeams[1][i] << " "
        << aBeams[2][i] << " "
        << aBeams[3][i] << std::endl;
        /* dump data for detector b. */
        dumpfileb
        << bBeams[0][i] << " "
        << bBeams[1][i] << " "
        << bBeams[2][i] << " "
        << bBeams[3][i] << std::endl;
        #endif
    }
    #ifdef POLBEAM_DUMPBEAMS
    dumpfilea.close();
    dumpfileb.close();
    #endif
}

void PolBeam::make_unpol_gaussian_elliptical_beams
(
    double fwhmx,
    double fwhmy,
    double phi0
)
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
    // FWHM = 2 \sqrt{2 \ln{2}} \sigma
    //
    // where \sigma is the standard deviation and 
    // 2 \sqrt{2 \ln{2}} ~ 2.35482
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
    double rho, sig, val;
    for(int bpix = 0; bpix < nPixels; bpix++)
    {
        pointing bp = hpxBase.pix2ang(bpix);
        rho = bp.theta;
        sig = bp.phi;
        val = exp(-(a*cos(sig)*cos(sig)
                  + 2.0*b*cos(sig)*sin(sig)
                  + c*sin(sig)*sin(sig))*rho*rho);
        // build fields for a perfectly co-polarized beams
        magEco[bpix] = sqrt(val);
        magEcx[bpix] = 0.0;
        phsEco[bpix] = 0.0;
        phsEcx[bpix] = 0.0;
    }
    // build beams from dummy fields
    if(enabledDets == 'a' || enabledDets == 'p')
    {
        beam_from_fields('a', magEco, phsEco, magEcx, phsEcx);
    }
    if(enabledDets == 'b' || enabledDets == 'p')
    {
        beam_from_fields('b', magEco, phsEco, magEcx, phsEcx);
    }
    // free temporal storage
    free(magEco);
    free(phsEco);
    free(magEcx);
    free(phsEcx);
}
