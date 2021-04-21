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
            int _nside, 
			const float* I, 
			const float* Q, 
			const float* U,
			const float* V);
       ~Sky(){};
        
        int get_nside (void) const { return nside; };
        unsigned long size (void) const { return nPixels; };

		Healpix_Base* hpxBase;
		
        // constant pointers to buffers
		const float* sI;
		const float* sQ;
		const float* sU;
		const float* sV;
        
    private:
		
		unsigned int  nside;
        unsigned long nPixels;
};

#endif // end include guard
