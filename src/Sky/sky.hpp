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

		Healpix_Base hpxBase;
		
        // constant pointers to buffers
		float* sI;
		float* sQ;
		float* sU;
		float* sV;
    private:
		
		unsigned int  _nside;
        unsigned long _nPixels;
};

#endif // end include guard
