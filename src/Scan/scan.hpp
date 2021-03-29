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
			const double* phi,
			const double* theta,
			const double* psi );
			
       ~Scan( void ){};
        
        unsigned long get_nsamples ( void ) const { return _nsamp; };
		
		const double* get_phi_ptr  ( void ) const;
		const double* get_theta_ptr( void ) const;
		const double* get_psi_ptr  ( void ) const;
	
    private:
		
        const double* _phi;
        const double* _theta;
        const double* _psi;
        
        unsigned long _nsamp;
};

#endif // end include guard
