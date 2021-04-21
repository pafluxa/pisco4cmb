#ifndef __SPHTRIGOH__
#define __SPHTRIGOH__

#include <math.h>

namespace SphericalTransformations
{ //begin namespace


void rho_sigma_chi_pix(
    double   *rho, double  *sigma, double    *chi,
    double  ra_bc, double  dec_bc, double  psi_bc,
    double ra_pix, double dec_pix );

void theta_phi_psi_pix
(
    double *theta, double *phi, double *psi,
    double  ra_bc, double dec_bc, double  psi_bc,
    double ra_pix, double dec_pix
);

} // end namespace

#endif
