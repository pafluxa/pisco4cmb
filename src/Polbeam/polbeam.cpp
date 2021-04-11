#include "polbeam.hpp"
//using namespace PolarizedBeam;

#include <complex>
#include <cstdlib>

// to make use of Healpix_base pointing
#include <pointing.h>

bool normalize_by_sum(float v[], int N)
{
    double sum = 0.0;
    for(int i = 0; i < N; i++)
    {
        sum += double(v[i]);
    }
    if(abs(sum) < 1e-6) 
        return false;
    for(int i = 0; i < N; i++)
    {
        v[i] = v[i]/sum;
    }
    
    return true;
}

std::complex<double> complex_2ddot(
    std::complex<double> w[2], 
    std::complex<double> z[2])
/*
 * computes the dot product of vectors w and z, where w = (w_x, w_y) and
 * z = (z_x, z_y) are vectors with complex components (w_x for instance,
 * is a complex number). Note the dot product of two complex vectors is
 * actually a complex number!
 *
 */
{
    std::complex<double> dp;
    dp = w[0]*std::conj(z[0]) + w[1]*std::conj(z[1]);
        
    return dp;
}


double complex_2dnorm(std::complex<double> w[2])
/*
 * computes the norm of a vector (w,z). w and z are complex numbers.
 */
{
    double norm;
    norm = std::real(complex_2ddot(w, w));
    
    return norm;
}

PolBeam::PolBeam(int nside, long nPixels) :
    nside{nside}, \
    nPixels{nPixels}, \
    epsilon{0.0}
{
    hpxBase.SetNside(nside, RING);
    pointing p = hpxBase.pix2ang(nPixels );

    rhoMax = p.theta;
    alloc_buffers();
}

PolBeam::~PolBeam()
{
    //free_buffers();
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
    // Jones vectors
    float* magEco_x, float* magEco_y, float* phaseEco,
    float* magEcx_x, float* magEcx_y, float* phaseEcx
)
{
    double ii;
    double qq;
    double uu;
    double vv;
    std::complex<double> E[2];
    std::complex<double> Eco[2];
    std::complex<double> Ecx[2];

    for(int i = 0; i < nPixels; i++)
    {
        // build co-polar jones vectors from phase/magnitude
        Eco[0] = std::polar(magEco_x[i], phaseEco[i]);
        Eco[1] = std::polar(magEco_y[i], phaseEco[i]);
        // build cross-polar jones vectors from phase/magnitude
        Ecx[0] = std::polar(0*magEcx_x[i], phaseEcx[i]);
        Ecx[1] = std::polar(0*magEcx_y[i], phaseEcx[i]);
        
        E[0] = Eco[0] + Ecx[0];
        E[1] = Eco[1] + Ecx[1];
        // compute \tilde{I}
        ii = complex_2dnorm(E);
        //std::cout << ii << std::endl;
        // compute \tilde{Q}
        qq = complex_2dnorm(Eco) - complex_2dnorm(Ecx);
        // compute \tilde{U}
        uu = 2*std::real(complex_2ddot(Eco, Ecx));
        // TODO: add V, possibly equals to this
        Ecx[0] = -Ecx[0];
        Ecx[1] = -Ecx[1];
        vv = -2*std::imag(complex_2ddot(Eco, Ecx));
    	
        if(polFlag == 'a')
	{	
            Ia[i] = ii;
            Qa[i] = qq;
            Ua[i] = uu;
            Va[i] = vv;
	}
	if(polFlag == 'b')
	{
            Ib[i] = ii;
            Qb[i] = qq;
            Ub[i] = uu;
            Vb[i] = vv;
    	}
    }
}

void PolBeam::build_beams(void)
{
    long i;
    epsilon = 0.0;
    for(i = 0; i < nPixels; i++)
    {
        aBeams[0][i] = Ia[i] + epsilon*Ib[i];
        
	aBeams[1][i] = Qa[i] - epsilon*Qb[i];
        aBeams[2][i] = Ua[i] - epsilon*Ub[i];

        aBeams[3][i] = Ua[i] - epsilon*Ub[i];
        aBeams[4][i] = Qa[i] - epsilon*Qb[i];
        
	aBeams[5][i] = 0;//(Va[i] + epsilon*Vb[i]);
        
	bBeams[0][i] = Ib[i] + epsilon*Ia[i];
        
	bBeams[1][i] = Qb[i] - epsilon*Qa[i];
        bBeams[2][i] = Ub[i] - epsilon*Ua[i];
        
	bBeams[3][i] = Ub[i] - epsilon*Ua[i];
        bBeams[4][i] = Qb[i] - epsilon*Qa[i];
       
	bBeams[5][i] = 0;//(Vb[i] + epsilon*Va[i]);            
    }
    //normalize so integral below beams is 1.0
    for(int i = 0; i < 5; i++)
    {   
        if(!normalize_by_sum(aBeams[i], nPixels))
        {
            std::cerr << "WARNING: A-beam idx " << i << " is zero" << std::endl;
        }
        if(!normalize_by_sum(bBeams[i], nPixels))
        {
            std::cerr << "WARNING: B-beam idx " << i << " is zero" << std::endl;
        }
    }
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
    float* phsEco = (float*)malloc(buffSize);
    float* phsEcx = (float*)malloc(buffSize);
    
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
        magEco_x[bpix] = sqrt(val);
        magEco_y[bpix] = 0.0;
        phsEco[bpix] = 0.0;

        magEcx_x[bpix] = 0.0;
        magEcx_y[bpix] = 0.0;
        phsEcx[bpix] = 0.0;
    }
    
    // build beams from dummy fields
    this->beam_from_fields(
        'a',
        magEco_x, magEco_y, phsEco,
        magEcx_x, magEcx_y, phsEcx);
    this->beam_from_fields(
        'b',
        magEco_x, magEco_y, phsEco,
        magEcx_x, magEcx_y, phsEcx);
    // free temporal storage
    free(magEco_x);
    free(magEco_y);
    free(magEcx_x);
    free(magEcx_y);
    free(phsEco);
    free(phsEcx);
}
