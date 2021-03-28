#include "scan.hpp"
#include <cstdlib>

Scan::Scan( 
	unsigned long nsamples, 
	const float* phi, 
	const float* theta, 
	const float* psi ) : _nsamp{ nsamples } 
{
	_phi = phi;
	_theta = theta;
	_psi = psi;
}

const float* Scan::get_phi_ptr() const {
	return _phi;
}

const float* Scan::get_theta_ptr() const {
	return _theta;
}

const float* Scan::get_psi_ptr() const {
	return _psi;
}
