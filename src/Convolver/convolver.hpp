/*
 * convolver.hpp 
 * 
 */
#ifndef _CPUCONVOLVERH
#define _CPUCONVOLVERH

#include "sky.hpp"
#include "scan.hpp"
#include "polbeam.hpp"

#include <vector>

class Convolver
{
    public:
		
        Convolver( unsigned long nsamples, unsigned int nthreads );
       ~Convolver();
		
		bool set_threads( unsigned int n );
		unsigned int get_threads( void );
		
		void exec_convolution( 
			float* convData,
			Scan& scan, 
			Sky& sky,
			PolBeam& beam );
		
    private:
		
		unsigned long _nsamples;
		unsigned int _nthreads;
				
		std::vector< float* > _tempBuffers;
		
		void _beam_times_sky( 
			Sky& sky, 
			PolBeam& beam, 
			float phi0, float theta0, float psi0,
            float da, float db);
			    
};

#endif
