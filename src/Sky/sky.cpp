#include "sky.hpp"
#include <cstdlib>

Sky::Sky( 
	int _nside, 
	const float* I,  
	const float* Q, 
    const float* U, 
	const float* V) : hpxBase(_nside, RING, SET_NSIDE)
{
    nside = _nside;
    nPixels = 12 * nside * nside;
	sI = I;
	sQ = Q;
	sU = U;
	sV = V;
}
