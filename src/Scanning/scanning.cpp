#include <cstdlib>
// for healpix
#include <healpix_base.h>

#include "Scanning/scanning.hpp"

Scanning::Scanning(void)
{

}

Scanning::~Scanning(void)
{
	
}

void Scanning::allocate_buffers(void)
{
	size_t bytes;

	bytes = sizeof(float) * nsamp;
	ra = (float *)malloc(bytes);
	dec = (float *)malloc(bytes);
	pa = (float *)malloc(bytes);
}

void Scanning::free_buffers(void)
{
	free(ra);
	free(dec);
	free(pa);
}

void Scanning::make_raster_scan(
	int nra, int ndec, int npa, 
	float ra0, float dra, 
	float dec0, float ddec, 
	float pa0, float dpa)
{
	int ira;
	int idec;
	int ipa;
	int idx;
	float tra;
	float tdec;
	float tpa;
	
	// set new number of samples
	nsamp = nra * ndec * npa;
	
	// allocate buffers
	allocate_buffers();

	// fill buffers with raster scan
	for(ipa = 0; ipa < npa; ipa++)
	{
		tpa = pa0 + dpa * float(ipa) / float(npa);
		for(ira = 0; ira < nra; ira++)
		{
			tra = ra0 - dra / 2.0f + dra * float(ira) / float(nra);
			for(idec = 0; idec < ndec; idec++)
			{
				tdec = dec0 - ddec / 2.0f + ddec * float(idec) / float(ndec);

				idx = ipa * nra * ndec + ira * ndec + idec;
				ra[idx] = tra;
				dec[idx] = tdec;
				pa[idx] = tpa;
			}
		}
	}
}

const float* Scanning::get_ra_ptr()  
{
	const float* x = ra;
	return x;
}

const float* Scanning::get_dec_ptr()  
{
	const float* x = dec;
	return x;
}

const float* Scanning::get_pa_ptr()  
{
	const float* x = pa;
	return x;
}