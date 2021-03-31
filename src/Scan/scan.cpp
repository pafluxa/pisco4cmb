#include "scan.hpp"
#include <cstdlib>

Scan::Scan( 
	unsigned long nsamples, 
	const double* _ra, 
	const double* _dec, 
	const double* _pa ) : nsamp{ nsamples } 
{
	ra = _ra;
	dec = _dec;
	pa = _pa;
}

const double* Scan::get_ra_ptr() const 
{
	return ra;
}

const double* Scan::get_dec_ptr() const 
{
	return dec;
}

const double* Scan::get_pa_ptr() const 
{
	return pa;
}
