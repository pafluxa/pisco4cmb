#include "convolver.hpp"
#include "polbeam.hpp"
#include "sky.hpp"
#include "scan.hpp"

#include <cstdlib>
#include <chrono>  

#define NSIDE   128	
#define NPIXELS (12*NSIDE*NSIDE)
// this number needs to be a perfect square for the test to work out!
// this is the number of pixels in an nside=128 healpix map
#define NSAMPLES 1024

#define NSIDE_BEAM 1024
// healpy.query_disc( 1024, (0,0,1), numpy.radians(5) ).size
#define NPIXELS_BEAM 23980

// routine to set the sky = 1 at the pole for I
void init_sky( float* I, float* Q, float* U, float* V )
{	
	// set I = 1 at the equator (theta=90,phi=0)
	// healpy.ang2pix( 128, numpy.radians(90), numpy.radians(0) ) 
	I[97536] = 1.0;
}

// routine to set the scan to observe the pole with a dumb grid approach
void init_scan( float* phi, float* theta, float* psi )
{	
	float deltaColat = 0.1;
	
	float _phi, _tht, _psi;
	// set psi = 0 and phi = 0 for the whole scan
	_psi = 0;
	_phi = 0;
	
	int s = 0;
	for( float i=NSAMPLES/2; i>0; i-- ) 
	{
		_tht = M_PI_2 - (i/NSAMPLES)*deltaColat;
		
		phi[s]   = _phi;
		theta[s] = _tht;
		psi[s]   = _psi;
		
		s++;
	}
	for( float i=0; i<NSAMPLES/2; i++ ) 
	{
		_tht = M_PI_2 + (i/NSAMPLES)*deltaColat;
		
		phi[s]   = _phi;
		theta[s] = _tht;
		psi[s]   = _psi;
		
		s++;
	}
}

int main( void )
{
	// timing
	auto start  = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed;
	
	float *data;
	data = (float*)malloc( sizeof(float)*NSAMPLES );
	
	float *skyI, *skyQ, *skyU, *skyV;
	skyI = (float*)calloc( NPIXELS, sizeof(float) );
	skyQ = (float*)calloc( NPIXELS, sizeof(float) );
	skyU = (float*)calloc( NPIXELS, sizeof(float) );
	skyV = (float*)calloc( NPIXELS, sizeof(float) );
	
	float *phi, *theta, *psi;
 	phi   = (float*)malloc( sizeof(float)*NSAMPLES );
	theta = (float*)malloc( sizeof(float)*NSAMPLES );
	psi   = (float*)malloc( sizeof(float)*NSAMPLES );
		
	init_sky( skyI, skyQ, skyU, skyV );
	Sky sky( NSIDE, skyI, skyQ, skyU, skyV );
    puts("#OK sky built");
	
	init_scan( phi, theta, psi );
	Scan scan( NSAMPLES, phi, theta, psi );
    puts("#OK scan built");
	
    PolBeam beam( NSIDE_BEAM, NPIXELS_BEAM );
    puts("#OK beam allocated");
	beam.make_unpol_gaussian_elliptical_beam( 1.5, 1.5, 0.0 );
    puts("#OK beams built");

    Convolver* cconv = new Convolver( NSAMPLES, 1 );
    puts("#OK convolver ready");
    puts("#WAIT computing convolution");
	start  = std::chrono::high_resolution_clock::now();
	cconv->exec_convolution( data, scan, sky, beam, 'a');
	finish = std::chrono::high_resolution_clock::now();
    puts("#OK convolution done");
    
	elapsed = finish - start;
	std::cerr << "#CPU Convolution took " << elapsed.count() << " sec\n";
	
    for(int i=0; i < NSAMPLES; i++)
        printf("%08d%14.8f\n", i, data[i]);

	delete cconv;
    
	free(phi); free(theta); free(psi);
	free(skyI); free(skyQ); free(skyU); free(skyV);
	free(data);

	return 0;
}
