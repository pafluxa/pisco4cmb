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
/*
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
*/
{
  double alpha, beta;
  double cdc = cos(dec_bc);
  double sdc = sin(dec_bc);
  double cdp = cos(dec_pix);
  double sdp = sin(dec_pix);
  double dra = ra_pix - ra_bc;
  double sd = sin(dra);
  double cd = cos(dra);
  
  double crho  = sdc * sdp + cdc * cdp * cd;
  if(crho >= 1.0)*rho = 0.0;
  else *rho = acos(crho);
  if(*rho > 1.0e-6) { 
    beta  = atan2(sd * cdc, sdc * cdp - cdc * sdp * cd);  
    alpha = atan2(sd * cdp, sdp * cdc - cdp * sdc * cd);
  } 
  else {
    // atan2(0,0) returns 0 instead of pi/2
    beta  = M_PI_2;
    alpha = M_PI_2;
  } 
  *sigma = M_PI - alpha - psi_bc;
  if(*sigma < 0.0)*sigma += 2.0 * M_PI;
  else if(*sigma > 2.0 * M_PI)*sigma -= 2.0 * M_PI;
  *chi = beta - *sigma;
  if(*chi < -M_PI)*chi += 2.0 * M_PI;
  
}

void SphericalTransformations::theta_phi_psi_pix
(
    double *theta, double *phi, double *psi,
    double  ra_bc, double dec_bc, double  psi_bc,
    double ra_pix, double dec_pix
)
/*   Calculates radial, polar offsets and position angle at a location
 *   offset from beam center. All outputs use the HEALPix coordinate system
 *   and the CMB polarization angle convention.
 *
 *   Outputs:  theta is the radial offset
 *             phi is the polar offset clockwise positive on the sky from South
 *             psi is the position angle clockwise positive from North
 *
 *   Inputs:   ra_bc   Right Ascension of beam center
 *             dec_bc  Declination of beam center
 *             psi_bc  Position angle of beam center clockwise positive from North
 *             ra_pix  Right Ascension of offset position
 *             dec_pix Declination of offset position
 */
    
{
  double cdc = cos(dec_bc);
  double sdc = sin(dec_bc);
  double cdp = cos(dec_pix);
  double sdp = sin(dec_pix);
  double dra = ra_pix - ra_bc;
  double sd = sin(dra);
  double cd = cos(dra);
  double spc = sin(psi_bc);
  double cpc = cos(psi_bc);
  // FEBeCoP Mitra, et. al. Formulas (3-11) (3-13) (3-16)
  double sdelx = cdp * sdc * cd - cdc * sdp;
  double sdely = cdp * sd;
  double xbeam =  sdelx * cpc + sdely * spc;
  double ybeam = -sdelx * spc + sdely * cpc;
  double sdx = xbeam;
  double cdx = sqrt(1.0 - sdx * sdx);
  double sdy = ybeam / cdx;
  double cdy = sqrt(1.0 - sdy * sdy);
  double cr = cdx * cdy;
  double cl = cdc*cpc;
  double sl_seps = cdc*spc*cdy - sdc * sdy;
  double sl_ceps = sdc*cdy + cdc*spc*sdy;
  *psi  = atan2(sl_seps, cl*cdx + sl_ceps*sdx);
  if(cr >= 1.0) {
    *theta = 0.0;
    *phi = 0.0;
  }
  else {
    *theta = acos(cr);
    *phi = atan2(ybeam, xbeam);
    if(*phi < 0.0)*phi += 2.0 * M_PI;
  }
}
