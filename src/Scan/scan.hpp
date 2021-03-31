/*
 * scan.hpp 
 * 
 */

#ifndef _SCANH // begin include guard
#define _SCANH

// include healpix related routines
//#include <healpix_base.h>

class Scan
{
    public:

        Scan( unsigned long nsamples );
        Scan( 
			unsigned long nsamples,
			const double* ra,
			const double* dec,
			const double* pa);
			
       ~Scan( void ){};
        
        unsigned long get_nsamples (void) const {return nsamp;};
		
		const double* get_ra_ptr(void) const;
		const double* get_dec_ptr(void) const;
		const double* get_pa_ptr(void) const;
	
    private:
		
        const double* ra;
        const double* dec;
        const double* pa;        
        unsigned long nsamp;
};

#endif // end include guard
