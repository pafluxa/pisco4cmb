#include "scan.hpp"
#include <cstdlib>

Scan::Scan( 
	long _nsamples, 
	float* _ra, 
	float* _dec, 
	float* _pa )
{
    nsamp = _nsamples;
	ra = _ra;
	dec = _dec;
	pa = _pa;
}

float* Scan::get_ra_ptr()  
{
	return ra;
}

float* Scan::get_dec_ptr()  
{
	return dec;
}

float* Scan::get_pa_ptr()  
{
	return pa;
}
