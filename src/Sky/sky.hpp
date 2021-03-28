/*
 * sky.hpp 
 * 
 */

#ifndef _SKYH // begin include guard
#define _SKYH

// include healpix related routines
#include <healpix_base.h>

class Sky
{
    public:
        
        Sky( 
			unsigned int nside, 
			float* I, 
			float* Q, 
			float* U,
			float* V );
        
       ~Sky(){};
        
        int get_nside ( void ) const { return _nside; };
        unsigned long size ( void ) const {return _nPixels; };

		const float* get_stokes_I_ptr( void );
		const float* get_stokes_Q_ptr( void );
		const float* get_stokes_U_ptr( void );
		const float* get_stokes_V_ptr( void );
		
		Healpix_Base hpxBase;
		
    private:
		
		unsigned int  _nside;
        unsigned long _nPixels;
		
		// constant pointers to buffers
		float* _sI;
		float* _sQ;
		float* _sU;
		float* _sV;
};

#endif // end include guard
