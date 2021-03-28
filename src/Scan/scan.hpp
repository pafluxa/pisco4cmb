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
			const float* phi,
			const float* theta,
			const float* psi );
			
       ~Scan( void ){};
        
        unsigned long get_nsamples ( void ) const { return _nsamp; };
		
		const float* get_phi_ptr  ( void ) const;
		const float* get_theta_ptr( void ) const;
		const float* get_psi_ptr  ( void ) const;
	
    private:
		
        const float* _phi;
        const float* _theta;
        const float* _psi;
        
        unsigned long _nsamp;

        void _alloc_buffers( void );
        void _free_buffers( void );
};

#endif // end include guard
