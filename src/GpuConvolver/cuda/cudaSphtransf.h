#ifndef __CUSPHTRIGOH__
#define __CUSPHTRIGOH__

#include <cuda.h>

namespace cudaSphericalTransformations
{
    
/*   
 * rho_sigma_chi_pix()
 * 
 * Calculates radial, polar offsets and position angle of the co-polarization
 * unit vector using Ludwig's 3rd definition at a location
 * offset from beam center. All outputs use the HEALPix coordinate system
 * and the CMB polarization angle convention.
 *
 * Outputs:  rho is the radial offset
 *           sigma is the polar offset clockwise positive on the sky from South
 *           chi is the position angle clockwise positive from South
 *
 * Inputs:   ra_bc   Right Ascension of beam center
 *           dec_bc  Declination of beam center
 *           psi_bc  Position angle of beam center clockwise positive from North
 *           ra_pix  Right Ascension of offset position
 *           dec_pix Declination of offset position
 */
__device__ void rho_sigma_chi_pix
(
    double *rho, double *sigma, double *chi,
    double  ra_bc, double dec_bc, double  psi_bc,
    double ra_pix, double dec_pix
);
// end namespace
} 
#endif
