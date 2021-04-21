#include "sky.hpp"
#include <cstdlib>

Sky::Sky( 
	int _nside, 
	const float* I,  
	const float* Q, 
    const float* U, 
	const float* V)
{
    nside = _nside;
    nPixels = 12*nside*nside;
    hpxBase = new Healpix_Base(nside, RING, SET_NSIDE);
	sI = I;
	sQ = Q;
	sU = U;
	sV = V;
}
