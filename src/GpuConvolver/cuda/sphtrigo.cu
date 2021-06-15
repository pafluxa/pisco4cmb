#include <math.h>
#include "sphtrigo.h"

__device__ void cudaSphericalTransformations::rho_sigma_chi_pix
(
    double *rho, double *sigma, double *chi,
    double  ra_bc, double dec_bc, double  psi_bc,
    double ra_pix, double dec_pix
)
/*   Calculates radial, polar offsets and position angle of the co-polarization
 *   unit vector using Ludwig's 3rd definition at a location
 *   offset from beam center. All outputs use the HEALPix coordinate system
 *   and the CMB polarization angle convention.
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
  /* NVIDIA: we are 100% IEE794 complaint. Sure you do!
     (this corner case was figured out by Michael Brewer) */ 
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
