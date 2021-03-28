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
}

PolBeam::~PolBeam() 
{
	free_buffers();
}

void PolBeam::alloc_buffers() 
{
	size_t buffSize = sizeof(float)*nPixels;
	
    Da_I = (float*)malloc(buffSize);
    Da_V = (float*)malloc(buffSize);
	Da_Qcos = (float*)malloc(buffSize);
	Da_Qsin = (float*)malloc(buffSize);
	Da_Ucos = (float*)malloc(buffSize);
	Da_Usin = (float*)malloc(buffSize);
    
	Db_I = (float*)malloc(buffSize);
    Db_V = (float*)malloc(buffSize);
	Db_Qcos = (float*)malloc(buffSize);
	Db_Qsin = (float*)malloc(buffSize);
	Db_Ucos = (float*)malloc(buffSize);
	Db_Usin = (float*)malloc(buffSize);
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
PolBeam::half_beam_from_fields
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
    
    if(polFlag != 'a' and polFlag != 'b') 
    {
        throw std::invalid_argument( "valid polFlags are 'a' and 'b'" );
    }

    if(polFlag == 'a')
    {
        Ip = Ia;
        Qp = Qa;
        Up = Ua;
        Vp = Va;
    }
    else
    {
        Ip = Ib;
        Qp = Qb;
        Up = Ub;
        Vp = Vb;
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
        Ecx_x = std::polar(magEco_x[i], phaseEco_x[i]);
        Ecx_y = std::polar(magEco_y[i], phaseEco_y[i]);
        
        // compute \tilde{I}
        ii = std::norm(Eco_x) + std::norm(Eco_y) \
           + std::norm(Ecx_x) + std::norm(Ecx_y);
        // compute \tilde{Q}
        qq = std::norm(Eco_x) + std::norm(Eco_y) \
           - std::norm(Ecx_x) + std::norm(Ecx_y);
        // compute \tilde{U}
        uu = 2*std::real(Eco_x*Ecx_x + Eco_y*Ecx_y);
        // TODO: add V, possibly equals this
        //vv = -2*std::imag(Eco_x*Ecx_x + Eco_y*Ecx_y);
        vv = 0.0;

        Ip[i] = ii;
        Qp[i] = qq;
        Up[i] = uu;
        Vp[i] = vv;
        
        omI += ii;
        omQ += qq;
        omU += uu;
        omV += vv;
    }
    
    // normalize
    for(i = 0; i < nPixels; i++)
    {
        Ip[i] = Ip[i]/omI;
        Qp[i] = Qp[i]/omQ;
        Up[i] = Up[i]/omU;
        //  don't normalize because omV is zero!!
        //Vp[i] = Ip[i]/omI;
    }
}

void PolBeam::build_beams(void)
{
    long i;
    for(i = 0; i < nPixels; i++)
    {
        // Da[1]
        Da_I[i] = Ia[i] + epsilon*Ib[i];
        // Da[2]
        Da_Qcos[i] = Ua[i] - epsilon*Ub[i];
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
}

void PolBeam::make_unpol_gaussian_elliptical_beam ( 
	double fwhmx, 
    double fwhmy, 
    double phi0 )
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
    
    // Compute coefficients to properly rotate the ellipse.
    double a =  (cos(phi_0)*cos(phi_0))/(2*sigma_x*sigma_x) + 
                (sin(phi_0)*sin(phi_0))/(2*sigma_y*sigma_y);
                
    double b = -(sin(2*phi_0) )/(4*sigma_x*sigma_x) + 
                (sin(2*phi_0 ))/(4*sigma_y*sigma_y);
                
    double c =  (sin(phi_0)*sin(phi_0))/(2*sigma_x*sigma_x) + 
                (cos(phi_0)*cos(phi_0))/(2*sigma_y*sigma_y);
    
    int bpix;
    double rho,sig,val;
    double omega = 0.0;
	for( bpix = 0; bpix < nPixels; bpix++ )
	{
		pointing bp = hpxBase.pix2ang(bpix);
		rho = bp.theta;
		sig = bp.phi;
		
		val = exp( -(a*cos(sig)*cos(sig) + 
		           2*b*cos(sig)*sin(sig) + 
		             c*sin(sig)*sin(sig))*rho*rho );
		
		// unpolarized beam means I,Q and U beams are all the same.
		Da_I[bpix] = val;
		Da_Qcos[bpix] = val;
		Da_Qsin[bpix] = val;
		Da_Ucos[bpix] = val;
		Da_Usin[bpix] = val;
		Da_V[bpix] = 0.0;

		Db_I[bpix] = val;
		Db_Qcos[bpix] = val;
		Db_Qsin[bpix] = val;
		Db_Ucos[bpix] = val;
		Db_Usin[bpix] = val;
		Db_V[bpix] = 0.0;
		
		omega = omega + val;
	}
	
	// normalize so beam solid angle equals 1.0
	for( bpix = 0; bpix < nPixels; bpix++ )
	{
		Da_I[bpix]    /= omega;
		Da_Qcos[bpix] /= omega;
		Da_Qsin[bpix] /= omega;
		Da_Ucos[bpix] /= omega;
		Da_Usin[bpix] /= omega;

		Db_I[bpix]    /= omega;
		Db_Qcos[bpix] /= omega;
		Db_Qsin[bpix] /= omega;
		Db_Ucos[bpix] /= omega;
		Db_Usin[bpix] /= omega;
	}
}
