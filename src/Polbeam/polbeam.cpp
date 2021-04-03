#include "polbeam.hpp"
//using namespace PolarizedBeam;

#include <complex>
#include <cstdlib>

// to make use of Healpix_base pointing
#include <pointing.h>

PolBeam::PolBeam(int nside, long nPixels) :
    nside{nside}, \
    nPixels{nPixels}, \
    epsilon{0.0}
{
    hpxBase.SetNside(nside, RING);
    pointing p = hpxBase.pix2ang(nPixels );

    rhoMax = p.theta;
    alloc_buffers();

    aBeams[0] = Da_I;
    bBeams[0] = Db_I; 

    aBeams[1] = Da_Qcos;  
    bBeams[1] = Db_Qcos;  
    
    aBeams[2] = Da_Qsin;
    bBeams[2] = Db_Qsin;
    
    aBeams[3] = Da_Ucos;
    bBeams[3] = Db_Ucos;
    
    aBeams[4] = Da_Usin;
    bBeams[4] = Db_Usin;

    aBeams[5] = Da_V; 
    bBeams[5] = Db_V; 
}

PolBeam::~PolBeam()
{
    //free_buffers();
}

void PolBeam::alloc_buffers()
{
    size_t buffSize = sizeof(float)*nPixels;

    Da_I = (float*)malloc(buffSize);
    Db_I = (float*)malloc(buffSize);
    Da_V = (float*)malloc(buffSize);
    Db_V = (float*)malloc(buffSize);

    Da_Qcos = (float*)malloc(buffSize);
    Db_Qcos = (float*)malloc(buffSize);
    Da_Qsin = (float*)malloc(buffSize);
    Db_Qsin = (float*)malloc(buffSize);
    Da_Ucos = (float*)malloc(buffSize);
    Db_Ucos = (float*)malloc(buffSize);
    Da_Usin = (float*)malloc(buffSize);
    Db_Usin = (float*)malloc(buffSize);

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
    free(Da_I);
    free(Da_V);
    free(Da_Qcos);
    free(Da_Qsin);
    free(Da_Ucos);
    free(Da_Usin);

    free(Db_I);
    free(Db_V);
    free(Db_Qcos);
    free(Db_Qsin);
    free(Db_Ucos);
    free(Db_Usin);
}

void
PolBeam::beam_from_fields
(
    char polFlag,
    // Jones vectors
    float* magEco_x, float* phaseEco_x,
    float* magEco_y, float* phaseEco_y,
    float* magEcx_x, float* phaseEcx_x,
    float* magEcx_y, float* phaseEcx_y
)
/*
 *
 */
{
    long i;

    float *Ip;
    float *Qp;
    float *Up;
    float *Vp;

    double ii;
    double qq;
    double uu;
    double vv;

    double omI;
    double omQ;
    double omU;
    double omV;

    std::complex<double> Eco_x;
    std::complex<double> Eco_y;
    std::complex<double> Ecx_x;
    std::complex<double> Ecx_y;

    if (polFlag == 'a')
    {
        Ip = Ia;
        Qp = Qa;
        Up = Ua;
        Vp = Va;
    }
    else if (polFlag == 'b')
    {
        Ip = Ib;
        Qp = Qb;
        Up = Ub;
        Vp = Vb;
    }
    else
    {
        throw std::invalid_argument(\
            "valid polFlags are 'a' and 'b'");
    }

    omI = 0.0;
    omQ = 0.0;
    omU = 0.0;
    omV = 0.0;
    for(i = 0; i < nPixels; i++)
    {
        // build co-polar jones vectors from phase/magnitude
        Eco_x = std::polar(magEco_x[i], phaseEco_x[i]);
        Eco_y = std::polar(magEco_y[i], phaseEco_y[i]);
        // build cross-polar jones vectors from phase/magnitude
        Ecx_x = std::polar(magEcx_x[i], phaseEcx_x[i]);
        Ecx_y = std::polar(magEcx_y[i], phaseEcx_y[i]);

        // compute \tilde{I}
        ii = std::norm(Eco_x) + std::norm(Eco_y) \
           + std::norm(Ecx_x) + std::norm(Ecx_y);
        // compute \tilde{Q}
        qq = std::norm(Eco_x) + std::norm(Eco_y) \
           - std::norm(Ecx_x) + std::norm(Ecx_y);
        // compute \tilde{U}
        uu = 2*std::real(Eco_x*Ecx_x + Eco_y*Ecx_y);
        // TODO: add V, possibly equals to this
        //vv = -2*std::imag(Eco_x*Ecx_x + Eco_y*Ecx_y);
        vv = 0.0;

        Ip[i] = sqrt(ii);
        Qp[i] = sqrt(qq);
        Up[i] = sqrt(uu);
        Vp[i] = sqrt(vv);

        omI += ii;
        omQ += qq;
        omU += uu;
        omV += vv;
    }

    // normalize
    /*
    for(i = 0; i < nPixels; i++)
    {
        if (omI > 0)
        {
            Ip[i] = Ip[i]/omI;
        }
        if (omQ > 0)
        {
            Qp[i] = Qp[i]/omQ;
        }
        if (omU > 0)
        {
            Up[i] = Up[i]/omU;
        }
        //  don't normalize because it is zero!!
        //Vp[i] = Ip[i]/omI;
    }
    */
}

void PolBeam::build_beams(void)
{
    long i;
    for(i = 0; i < nPixels; i++)
    {
        // Da[1]
        Da_I[i] = Ia[i] + epsilon*Ib[i];
        // Da[2]
        Da_Qcos[i] = Qa[i] - epsilon*Qb[i];
        Da_Qsin[i] = -(Ua[i] - epsilon*Ub[i]);
        // Da[3]
        Da_Ucos[i] = Ua[i] - epsilon*Ub[i];
        Da_Usin[i] = Qa[i] - epsilon*Qb[i];
        // Da[4]
        Da_V[i] = 0.0;

        // Db[1]
        Db_I[i] = Ib[i] + epsilon*Ia[i];
        // Db[2]
        Db_Qcos[i] = -(Qb[i] - epsilon*Qa[i]);
        Db_Qsin[i] = Ub[i] - epsilon*Ua[i];
        // Db[3]
        Db_Ucos[i] = -(Ub[i] - epsilon*Ua[i]);
        Db_Usin[i] = -(Qb[i] - epsilon*Qa[i]);
        // Db[4]
        Db_V[i] = 0.0;
    }

    free(Ia);
    free(Qa);
    free(Ua);
    free(Va);
    free(Ib);
    free(Qb);
    free(Ub);
    free(Vb);
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
    size_t buffSize = sizeof(float)*nPixels;

    float* magEco_x = (float*)malloc(buffSize);
    float* magEco_y = (float*)malloc(buffSize);
    float* magEcx_x = (float*)malloc(buffSize);
    float* magEcx_y = (float*)malloc(buffSize);

    float* phaseEco_x = (float*)malloc(buffSize);
    float* phaseEco_y = (float*)malloc(buffSize);
    float* phaseEcx_x = (float*)malloc(buffSize);
    float* phaseEcx_y = (float*)malloc(buffSize);

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
        pointing bp = hpxBase.pix2ang(bpix);
        rho = bp.theta;
        sig = bp.phi;

        val = exp(-(a*cos(sig)*cos(sig)
                  + 2*b*cos(sig)*sin(sig)
                  + c*sin(sig)*sin(sig))*rho*rho);
        if(val < 1e-6)
            val = 0.0;
        // perfectly polarized beam
        magEco_x[bpix] = val;
        magEco_y[bpix] = val;
        magEcx_x[bpix] = 0.0;
        magEcx_y[bpix] = 0.0;

        phaseEco_x[bpix] = 0.0;
        phaseEco_y[bpix] = 0.0;
        phaseEcx_x[bpix] = 0.0;
        phaseEcx_y[bpix] = 0.0;
    }
    
    // build beams from dummy fields
    this->beam_from_fields(
        'a',
        magEco_x, phaseEco_x,
        magEco_y, phaseEco_y,
        magEcx_x, phaseEcx_x,
        magEcx_y, phaseEcx_y);

    this->beam_from_fields(
        'b',
        magEco_x, phaseEco_x,
        magEco_y, phaseEco_y,
        magEcx_x, phaseEcx_x,
        magEcx_y, phaseEcx_y);
        
    // free temporal storage
    free(magEco_x);
    free(magEco_y);
    free(magEcx_x);
    free(magEcx_y);
    free(phaseEco_x);
    free(phaseEco_y);
    free(phaseEcx_x);
    free(phaseEcx_y);
}
