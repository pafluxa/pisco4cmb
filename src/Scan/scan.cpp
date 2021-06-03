#include "scan.hpp"
#include <cstdlib>

Scan::Scan( 
	long _nsamples, 
	const float* _ra, 
	const float* _dec, 
	const float* _pa )
{
    nsamp = _nsamples;
	ra = _ra;
	dec = _dec;
	pa = _pa;
}

const float* Scan::get_ra_ptr() const 
{
	return ra;
}

const float* Scan::get_dec_ptr() const 
{
	return dec;
}

const float* Scan::get_pa_ptr() const 
{
	return pa;
}
