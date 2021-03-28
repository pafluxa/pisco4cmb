#include "sphtrigo.hpp"

void SphericalTransformations::rho_sigma_chi_pix(
	double   *rho, double  *sigma, double    *chi,
    double  ra_bc, double  dec_bc, double  psi_bc,
    double ra_pix, double dec_pix )
/*   
 *   Calculates radial, polar offsets and position angle of the co-polarization
 *   unit vector using Ludwig's 3rd definition at a location offset from beam center. 
 *   
 *   All outputs use the HEALPix coordinate system and the CMB polarization angle 
 *   convention as described in Fluxa et al. 2019 (https://arxiv.org/abs/1908.05662)
 *
 *   Outputs:  rho is the radial offset
 *             sigma is the polar offset clockwise positive on the sky from South
 *             chi is the position angle clockwise positive from South
 *
 *   Inputs:   ra_bc   Right Ascension of beam center
 *             dec_bc  Declination of beam center
 *             psi_bc  Position angle of beam center clockwise positive from North
 *             ra_pix  Right Ascension of offset position
 *             dec_pix Declination of offset position
 * 
 *   Author: Michael K. Brewer (2019).
 */
    
{
  double cdc = cos(dec_bc);
  double sdc = sin(dec_bc);
  double cdp = cos(dec_pix);
  double sdp = sin(dec_pix);
  double dra = ra_pix - ra_bc;
  double sd = sin(dra);
  double cd = cos(dra);
                                                                                                              
  double gamma = atan2(sd * cdp, sdp * cdc - cdp * sdc * cd);  
  double delta = atan2(sd * cdc, sdc * cdp - cdc * sdp * cd);  
  double crho  = sdc * sdp + cdc * cdp * cd;                                                                                   

  if(crho >= 1.0) {
    *rho = 0.0;
    *sigma = 0.0;
    *chi = psi_bc;
  }
  else {
    *rho = acos(crho);
    *sigma = M_PI - gamma - psi_bc;
    if(*sigma < 0.0)*sigma += 2.0 * M_PI;
    else if(*sigma > 2.0 * M_PI)*sigma -= 2.0 * M_PI;
    *chi = delta - *sigma;
    if(*chi < -M_PI)*chi += 2.0 * M_PI;
  }
}
