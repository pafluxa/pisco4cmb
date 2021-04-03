#include "sky.hpp"
#include <cstdlib>

Sky::Sky( 
	int _nside, 
	const float* I,  
	const float* Q, const float* U, 
	const float* V)
{
    nside = _nside;
    nPixels = 12*nside*nside;
	hpxBase.SetNside(nside, RING);
	
	sI = I;
	sQ = Q;
	sU = U;
	sV = V;
    
    mapPointers[0] = sI;
    mapPointers[1] = sQ;
    mapPointers[2] = sU;
    mapPointers[3] = sV;
}
