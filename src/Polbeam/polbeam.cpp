#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "polbeam.hpp"

#include <pointing.h>

PolBeam::PolBeam(int _nside) : hpxBase(_nside, RING, SET_NSIDE)
/** Constructor.
 * 
 * \param _nside: resolution parameter of beams (HealPix).
 */
{
    nside = _nside;
    nPixels = 12 * nside * nside;
    // 1 - polarization efficiency
    epsilon = 0.0;
    psbmode = 'p';
    buffersOK = false;

    pointing p = hpxBase.pix2ang(nPixels);
    rhoMax = p.theta;
    
    beamBufferSize = sizeof(float) * nPixels;
    #ifdef POLBEAM_DEBUG
    std::cerr << "PolBeam::constructor" << std::endl;
    std::cerr << "  PSB mode: " << psbmode << std::endl;
    std::cerr << "  Beam extension: " << rhoMax * (180.0 / M_PI) << " degrees." << std::endl;
    std::cerr << "  Polarization efficiency: " << 1.0 - epsilon << std::endl;
    std::cerr << "  Number of pixels per beam: " << nPixels << std::endl;
    std::cerr << "  Memory required to store beam data: " << 8 * beamBufferSize << std::endl;
    #endif 
}

PolBeam::~PolBeam()
/** Destructor.
 *  Free all buffers before destruction, if allocated.
 */
{
    if(buffersOK) {
        free_buffers();
    }
}

int PolBeam::get_nside(void) const {
    return nside;
};

int PolBeam::get_npixels(void) const {
    return nPixels;
};

char PolBeam::get_psb_mode(void) const {
    return psbmode;
}

double PolBeam::get_rho_max() const {
    return rhoMax;
};

const float* PolBeam::get_I_beam(char det) const {
    const float* nothing;
    
    if(det == 'a') {
        const float* x = Ia;
        return x;
    }
    else if(det == 'b') {
        const float* x = Ib;
        return x;
    }
    else {
        throw std::invalid_argument("[ERROR] Please provide 'a' or 'b' as `det` argument.");
    }
    /* just so compiler does not yell at me. */
    return nothing;
}

const float* PolBeam::get_Q_beam(char det) const {
    const float* nothing;
    
    if(det == 'a') {
        const float* x = Qa;
        return x;
    }
    else if(det == 'b') {
        const float* x = Qb;
        return x;
    }
    else {
        throw std::invalid_argument("[ERROR] Please provide 'a' or 'b' as `det` argument.");
    }
    /* just so compiler does not yell at me. */
    return nothing;
}

const float* PolBeam::get_U_beam(char det) const {
    const float* nothing;
    
    if(det == 'a') {
        const float* x = Ua;
        return x;
    }
    else if(det == 'b') {
        const float* x = Ub;
        return x;
    }
    else {
        throw std::invalid_argument("[ERROR] Please provide 'a' or 'b' as `det` argument.");
    }
    /* just so compiler does not yell at me. */
    return nothing;
}

const float* PolBeam::get_V_beam(char det) const {
    const float* nothing;
    
    if(det == 'a') {
        const float* x = Va;
        return x;
    }
    else if(det == 'b') {
        const float* x = Vb;
        return x;
    }
    else {
        throw std::invalid_argument("[ERROR] Please provide 'a' or 'b' as `det` argument.");
    }
    /* just so compiler does not yell at me. */
    return nothing;
}

void PolBeam::allocate_buffers()
/** Allocates memory to store beam data. 
 * 
 * This routine allocates memory for both beams, even if only one will
 * be used in the end (because the right thing to do is to set it to zero!)
 */
{
    #ifdef POLBEAM_DEBUG
    std::cerr << "PolBeam::alloc_buffers" << std::endl;
    std::cerr << "  Allocating beam buffers for detector a." << std::endl;
    #endif
    /* allocate buffers for bolometer a. */
    Ia = (float*)malloc(beamBufferSize);
    Qa = (float*)malloc(beamBufferSize);
    Ua = (float*)malloc(beamBufferSize);
    Va = (float*)malloc(beamBufferSize);
     
    #ifdef POLBEAM_DEBUG
    std::cerr << "PolBeam::alloc_buffers" << std::endl;
    std::cerr << "  Allocating beam buffers for detector b." << std::endl;
    #endif
    /* allocate buffers for bolometer b. */
    Ib = (float*)malloc(beamBufferSize);
    Qb = (float*)malloc(beamBufferSize);
    Ub = (float*)malloc(beamBufferSize);
    Vb = (float*)malloc(beamBufferSize);
    /* set beams to zero by default. */
    /* detector a. */
    std::memset(Ia, 0, beamBufferSize);
    std::memset(Qa, 0, beamBufferSize);
    std::memset(Ua, 0, beamBufferSize);
    std::memset(Va, 0, beamBufferSize);
    /* detector b. */
    std::memset(Ib, 0, beamBufferSize);
    std::memset(Qb, 0, beamBufferSize);
    std::memset(Ub, 0, beamBufferSize);
    std::memset(Vb, 0, beamBufferSize);

    buffersOK = true;
}

void PolBeam::free_buffers()
/**Free all buffers.
 */
{
    #ifdef POLBEAM_DEBUG
    std::cerr << "PolBeam::free_buffers" << std::endl;
    std::cerr << "  Freeing beam buffers for detector a." << std::endl;
    #endif
    free(Ia);
    free(Qa);
    free(Ua);
    free(Va);
    #ifdef POLBEAM_DEBUG
    std::cerr << "PolBeam::free_buffers" << std::endl;
    std::cerr << "  Freeing beam buffers for detector b." << std::endl;
    #endif
    free(Ib);
    free(Qb);
    free(Ub);
    free(Vb);
}

void PolBeam::load_beam_data_from_txt(char det, std::string path)
/**Load electric field data from a text file.
 * 
 * File must contain at least `nPixels` lines and have 4 (four) columns:
 *   column 0 is magnitude of co-polarized component of electric field.
 *   column 1 is phase of co-polarized component of electric field.
 *   column 2 is magnitude of cross-polarized component of electric field.
 *   column 3 is phase of cross-polarized component of electric field.
 *
 * The computation of the polarized beam of a single PSB detector (a or b) 
 * using the electric field power density (EFPD) is based in the formalism 
 * found in Rosset el at. 2010, with massive contributions from Michael K. Brewer.

 * This routine throws an invalid argument error if `det` is not 'a' nor 'b'.
 * This routine throws a runtime error if buffers have not been allocated.
 * This routine throws a runtime error if the file cannot be opened.
 * This routine throws a runtime error if the file fails to be parsed.
 */
{
    /* begin variable declarations. */
    int i;
    
    float *I;
    float *Q;
    float *U;
    float *V;
    
    std::complex<double> Eco;
    std::complex<double> Ecx;
    double magEco;
    double phsEco;
    double magEcx;
    double phsEcx;
    
    std::string line;
    std::ifstream beamDataFile(path.c_str());
    /* end variable declarations. */
    
    #ifdef POLBEAM_DEBUG
    std::cerr << "PolBeam::load_beam_data_from_txt" << std::endl;
    std::cerr << "  Reading beam data from file " << path << std::endl;
    #endif
    /* check if detector flag is valid. */
    if(det == 'a'){
        I = Ia;
        Q = Qa;
        U = Ua;
        V = Va;
    }
    else if(det == 'b') {
        I = Ib;
        Q = Qb;
        U = Ub;
        V = Vb;
    }     
    else {
        throw std::invalid_argument("[ERROR] Please provide 'a' or 'b' as detector.");
    }
    /* check buffers have been allocated. */
    if(!buffersOK)
    {
        throw std::runtime_error("[ERROR] Buffers have not been successfully allocated.");
    }
    /* check that the file can be opened. */
    if(!beamDataFile.is_open())
    {
        throw std::runtime_error("[ERROR] Could not open file.");
    }
    /* read contents of file using a rather hacky way :D */
    i = 0;
    while(std::getline(beamDataFile, line) && i < nPixels) 
    {
        std::istringstream iss(line);
        if( !(iss >> magEco >> phsEco >> magEcx >> phsEcx) ) {
            beamDataFile.close();
            throw std::runtime_error("[ERROR] Could not parse the contents of the file.");
        }        
        /* scalar co-polarized component. */
        Eco = std::polar(magEco, phsEco);
        /* scalar cross-polarized component. */
        Ecx = std::polar(magEcx, phsEcx);
        // compute \tilde{I}
        I[i] = float(std::norm(Eco) + std::norm(Ecx));
        // compute \tilde{Q}
        Q[i] = float(std::norm(Eco) - std::norm(Ecx));
        // compute \tilde{U}
        U[i] = float(2*std::real(Eco*std::conj(Ecx)));
        // compute \tilde{V}
        V[i] = float(-2*std::imag(Eco*std::conj(Ecx)));
        i++;
    }
    /* check that we read all the pixels we needed. */
    if(i != nPixels) {
        std::cerr << "[WARNING] Only " << i << " lines were read while " << nPixels << " were expected." << std::endl;
    }
    /* close file. */
    beamDataFile.close();
}    

void PolBeam::normalize(int nsideSky)
/**Build the actual polarized beams from tilde I, Q, U and V. These 
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
    /* begin variable declarations. */
    double sumIa;
    double sumIb;
    double ocmp;
    double aNorm;
    double bNorm;
    float temp1;
    float temp2;
    float temp3;
    float temp4;
    /* end variable declarations. */
    
    #ifdef POLBEAM_DEBUG
    std::cerr << "PolBeam::normalize" << std::endl;
    std::cerr << "  normalizing detector beams." << std::endl;
    #endif
    /* compute solid angle */
    sumIa = 0;
    sumIb = 0;
    for(int i = 0; i < nPixels; i++) 
    {
        if(psbmode == 'a' || psbmode == 'p')
        {
            sumIa += Ia[i];
        }
        if(psbmode == 'b' || psbmode == 'p')
        {        
            sumIb += Ib[i];
        }
    }   
    #ifdef POLBEAM_DEBUG
    std::cerr << "PolBeam::normalize" << std::endl;
    std::cerr << "  solid angle of detector a = ";
    std::cerr << 1000 * sumIa * ((4.0 * M_PI) / (nPixels)) << " milistrad" << std::endl;
    std::cerr << "PolBeam::normalize" << std::endl;
    std::cerr << "  solid angle of detector b = "; 
    std::cerr << 1000 * sumIb * ((4.0 * M_PI) / (nPixels)) << " milistrad" << std::endl;
    #endif
    // compute normalization factor to take into account the size of 
    // the sky pixel in the convolution
    ocmp = nsideSky / double(nside);
    ocmp = ocmp * ocmp;
    aNorm = sumIa * ocmp;
    bNorm = sumIb * ocmp;
    // hack to avoid the beam from blowing up when one detector is off
    if(psbmode == 'a')
    {
        bNorm += 1e-8;
    }
    if(psbmode == 'b')
    {
        aNorm += 1e-8;
    }
    #ifdef POLBEAM_DEBUG
    std::cerr << "PolBeam::normalize" << std::endl;
    std::cerr << "  normalizing and compensating for polarization efficiency.";
    std::cerr << std::endl;
    #endif
    for(int i = 0; i < nPixels; i++)
    {
        /* for detector a. */
        temp1 = (Ia[i] + epsilon * Ib[i]) / aNorm;
        temp2 = (Qa[i] - epsilon * Qb[i]) / aNorm;
        temp3 = (Ua[i] - epsilon * Ub[i]) / aNorm;
        temp4 = (Va[i] + epsilon * Vb[i]) / aNorm;
        Ia[i] = temp1;
        Qa[i] = temp2;
        Ua[i] = temp3;
        Va[i] = temp4;
        /* for detector b. */
        temp1 = (Ib[i] + epsilon * Ia[i]) / bNorm;
        temp2 = (Qb[i] - epsilon * Qa[i]) / bNorm;
        temp3 = (Ub[i] - epsilon * Ua[i]) / bNorm;
        temp4 = (Vb[i] + epsilon * Va[i]) / bNorm;
        Ib[i] = temp1;
        Qb[i] = temp2;
        Ub[i] = temp3;
        Vb[i] = temp4;
    }
}

void PolBeam::make_unpol_gaussian_elliptical_beam(char det, double fwhmx, double fwhmy, double phi0)
/**Creates a Gaussian Elliptical Beam as a HEALPix grid.
 *
 *  Authors: Michael K. Brewer (CLASS collaboration, 2018)
 *           C++ adaptation by Pedro Fluxa (PUC, 2019)
 *  input:
 *
 *      det     : 'a' for detector a, 'b' for detector b.
 *      fwhm_x  : Full Width at Half Maximum, in the x (South-North)
 *                direction if phi_0=0 (degrees)
 *      fwhm_y  : Full Width at Half Maximum, in the y (East-West)
 *                direction if phi=0, (degrees)
 *      phi0   : Angle between Noth-South direction and x-axis of the
 *                ellipse (degrees). phi_0 increases clockwise towards
 *                East.
 *
 *  In this routine, the same elliptical gaussian is assigned to
 *  I,Q, U and V beams. All beams are normalized so that
 *
 *  \int_{4\pi} b(\rho,\sigma) d\Omega = 1.0
 * 
 *  This routine throws an invalid argument error if `det` is not 'a' nor 'b'.
 *  This routine throws a runtime error if buffers have not been allocated.
 *  
 *  Notes:
 *      Some adaptations were made to make the code compliant with C++11 standard.
 */
{
    /* begin variable declarations. */
    int bpix;

    float *I;
    float *Q;
    float *U;
    float *V;
    
    double magEco;
    double phsEco;
    double magEcx;
    double phsEcx;
    std::complex<double> Eco;
    std::complex<double> Ecx;
    
    double a;
    double b;
    double c;
    double rho;
    double sig;
    double phi_0;
    double sigma_x;
    double sigma_y;
    double deg2rad;
    double val;
    
    pointing bp;
    /* end variable declarations. */

    /* check buffers have been allocated. */
    if(!buffersOK)
    {
        throw std::runtime_error("[ERROR] Buffers have not been successfully allocated.");
    }
    /* check if detector flag is valid. */
    if(det == 'a'){
        I = Ia;
        Q = Qa;
        U = Ua;
        V = Va;
    }
    else if(det == 'b') {
        I = Ib;
        Q = Qb;
        U = Ub;
        V = Vb;
    }     
    else {
        throw std::invalid_argument("[ERROR] Please provide 'a' or 'b' as detector.");
    }
    // to convert FWHM in degres to sigma, in radians
    deg2rad = M_PI/180.0;
    // From https://en.wikipedia.org/wiki/Full_width_at_half_maximum:
    //
    // FWHM = 2 \sqrt{2 \ln{2}} \sigma
    //
    // where \sigma is the standard deviation and 
    // 2 \sqrt{2 \ln{2}} ~ 2.35482
    sigma_x = (deg2rad*fwhmx)/2.35482;
    sigma_y = (deg2rad*fwhmy)/2.35482;
    //Convert phi_0 to radians
    phi_0 = deg2rad * phi0;
    // Compute coefficients to rotate the ellipse.
    a =  (cos(phi_0)*cos(phi_0))/(2*sigma_x*sigma_x) +
         (sin(phi_0)*sin(phi_0))/(2*sigma_y*sigma_y);
    b = -(sin(2*phi_0) )/(4*sigma_x*sigma_x) +
         (sin(2*phi_0 ))/(4*sigma_y*sigma_y);
    c =  (sin(phi_0)*sin(phi_0))/(2*sigma_x*sigma_x) +
                (cos(phi_0)*cos(phi_0))/(2*sigma_y*sigma_y);
    for(bpix = 0; bpix < nPixels; bpix++)
    {
        Healpix_Base hpxBase(nside, RING, SET_NSIDE);
        bp = hpxBase.pix2ang(bpix);
        rho = bp.theta;
        sig = bp.phi;
        val = exp(-(a*cos(sig)*cos(sig)
                  + 2.0*b*cos(sig)*sin(sig)
                  + c*sin(sig)*sin(sig))*rho*rho);
        // build fields for a perfectly co-polarized beams. 
        // yes, this is VERY explicit.
        magEco = sqrt(val);
        phsEco = 0.0;
        magEcx = 0.0;
        phsEcx = 0.0;
        /* scalar co-polarized component. */
        Eco = std::polar(magEco, phsEco);
        /* scalar cross-polarized component. */
        Ecx = std::polar(magEcx, phsEcx);
        // compute \tilde{I}
        I[bpix] = float(std::norm(Eco) + std::norm(Ecx));
        // compute \tilde{Q}
        Q[bpix] = float(std::norm(Eco) - std::norm(Ecx));
        // compute \tilde{U}
        U[bpix] = float(2*std::real(Eco*std::conj(Ecx)));
        // compute \tilde{V}
        V[bpix] = float(-2*std::imag(Eco*std::conj(Ecx)));
    }
}
