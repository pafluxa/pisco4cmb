#include "scan.hpp"
#include <cstdlib>

Scan::Scan( 
	unsigned long nsamples, 
	const double* phi, 
	const double* theta, 
	const double* psi ) : _nsamp{ nsamples } 
{
	_phi = phi;
	_theta = theta;
	_psi = psi;
}

const double* Scan::get_phi_ptr() const {
	return _phi;
}

const double* Scan::get_theta_ptr() const {
	return _theta;
}

const double* Scan::get_psi_ptr() const {
	return _psi;
}
