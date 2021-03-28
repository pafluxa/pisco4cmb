#include "sky.hpp"
#include <cstdlib>

Sky::Sky( 
	unsigned int nside, 
	float* I, 
	float* Q, 
	float* U, 
	float* V ) : _nside{ nside } 
{
	hpxBase.SetNside( nside, RING );
	
	_nPixels = 12*nside*nside;
	
	sI = I;
	sQ = Q;
	sU = U;
	sV = V;
}
