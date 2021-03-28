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
	
	_sI = I;
	_sQ = Q;
	_sU = U;
	_sV = V;
}


const float* Sky::get_stokes_I_ptr(){
	return _sI;
}

const float* Sky::get_stokes_Q_ptr(){
	return _sQ;
}

const float* Sky::get_stokes_U_ptr(){
	return _sU;
}

const float* Sky::get_stokes_V_ptr(){
	return _sV;
}
