/*
 * main.cpp
 * 
 * Performs a 24 hour constant elevation scan from the CLASS site.
 * Three different boresight rotations are used (-45 deg, 0 deg, 45 deg)
 * 
 */
#include <cstdlib>
#include <chrono>  
#include <cstring>
#include <fstream>
#include <cstdlib> // for exit function

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Bpoint/BPoint.h"
#include "Polbeam/polbeam.hpp"
#include "Convolver/convolver.hpp"
#include "Mapping/mapping_routines.h"

#include <healpix_base.h>
// to make use of Healpix_base pointing
#include <pointing.h>

#define NSIDE_SKY  128
#define NPIXELS_SKY (12*NSIDE_SKY*NSIDE_SKY)
// this number needs to be a perfect square for the test to work out!
// this is the number of pixels in an nside=128 healpix map
#define NSAMPLES NPIXELS_SKY


#define NSIDE_BEAM 512
// healpy.query_disc( 1024, (0,0,1), numpy.radians(5) ).size
#define NPIXELS_BEAM 23980
#define NPIXELS_BEAM_TOT (12*NSIDE_BEAM*NSIDE_BEAM)

// routine to set the sky = 1 at the pole for I
void init_sky( float* I, float* Q, float* U, float* V )
{	
	for(int i = 0; i < NPIXELS_SKY; i++)
    {
        I[i] = (1.0*(rand()%10000))/(10000);
        Q[i] = (1.0*(rand()%10000))/(10000);
        U[i] = (1.0*(rand()%10000))/(10000);
        V[i] = (1.0*(rand()%10000))/(10000);
    }
}

// routine to set the scan to observe the pole with a dumb grid approach
void init_scan( double* phi, double* theta, double* psi, double phi0, double psi0 )
{	
    Healpix_Base hpxBase;
    hpxBase.SetNside(NSIDE_SKY, RING);

    for(int pix=0; pix < NPIXELS_SKY; pix++)
    {
        pointing bp = hpxBase.pix2ang(pix);
        theta[pix] = bp.theta;
        phi[pix] = bp.phi;
        psi[pix] = psi0;
    }
}

int main( void )
{
    std::ofstream outdata;
    
	// timing
	auto start  = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed;
	
	float *data;
	data = (float*)malloc( sizeof(float)*NSAMPLES );
	
	float *skyI, *skyQ, *skyU, *skyV;
	skyI = (float*)calloc( NPIXELS_SKY, sizeof(float) );
	skyQ = (float*)calloc( NPIXELS_SKY, sizeof(float) );
	skyU = (float*)calloc( NPIXELS_SKY, sizeof(float) );
	skyV = (float*)calloc( NPIXELS_SKY, sizeof(float) );
	
	double *phi, *theta, *psi;
 	phi   = (double*)malloc( sizeof(double)*NSAMPLES );
	theta = (double*)malloc( sizeof(double)*NSAMPLES );
	psi   = (double*)malloc( sizeof(double)*NSAMPLES );
    
	init_sky( skyI, skyQ, skyU, skyV );
	Sky sky( NSIDE_SKY, skyI, skyQ, skyU, skyV );
    // opens the file
    outdata.open("maps_input.txt"); 
     // file couldn't be opened
    if( !outdata ) 
    {
        std::cerr << "Error: file could not be opened" << std::endl;
        exit(1);
    }
    for(int i=0; i<NPIXELS_SKY; ++i)
    {
        outdata << skyI[i] << " " << skyQ[i] << " " << skyU[i] << std::endl;
    }
    outdata.close();
    
    PolBeam beam( NSIDE_BEAM, NPIXELS_BEAM );
	beam.make_unpol_gaussian_elliptical_beam( 5.0, 5.0, 0.0 );

    Scan scan( NSAMPLES, phi, theta, psi);
    
    Convolver* cconv = new Convolver(NSAMPLES, 16);

    int* scanMask = (int*)malloc(sizeof(int)*NSAMPLES);
    double* polangles = (double*)malloc(sizeof(double)*1);
    std::memset(scanMask, 0, sizeof(int)*NSAMPLES);
    std::memset(polangles, 0, sizeof(double)*1);
    
    long mapNpixels = NPIXELS_SKY;
    int* mapPixels = (int*)malloc(sizeof(int)*mapNpixels);
    for(long i=0; i < mapNpixels; i++)
    {
        mapPixels[i] = i;
    }
    
    double* AtA = (double*)malloc(sizeof(double)*9*mapNpixels);
    double* AtD = (double*)malloc(sizeof(double)*3*mapNpixels);
    
    int detuids[1] = {0};
    double angles[3] = {-M_PI/4, 0.0, M_PI/4.0};
    for(double polangle: angles)
    {
        init_scan( phi, theta, psi, 0.0, polangle );
        start  = std::chrono::high_resolution_clock::now();
        cconv->exec_convolution(data, scan, sky, beam, 'a');
        finish = std::chrono::high_resolution_clock::now();
        
        elapsed = finish - start;
        std::cerr << "#CPU Convolution took " << elapsed.count() << " sec\n";
        // project to map-making matrices
        libmapping_project_data_to_matrices
        (
            NSAMPLES, 1,
            phi, theta, psi, 
            polangles,
            data, scanMask, detuids,
            NSIDE_SKY, mapNpixels, mapPixels,
            AtA, AtD
        );
    }
    
    std::memset(skyI, 0, sizeof(float)*NPIXELS_SKY);
    std::memset(skyQ, 0, sizeof(float)*NPIXELS_SKY);
    std::memset(skyU, 0, sizeof(float)*NPIXELS_SKY);
    std::memset(skyV, 0, sizeof(float)*NPIXELS_SKY);
    
    libmapping_get_IQU_from_matrices
    (
        NSIDE_SKY, mapNpixels, 
        AtA, AtD, mapPixels,
        skyI, skyQ, skyU, skyV
    );
    
    // opens the file
    outdata.open("maps_output.txt"); 
     // file couldn't be opened
    if( !outdata ) 
    {
        std::cerr << "Error: file could not be opened" << std::endl;
        exit(1);
    }
    for(int i=0; i<NPIXELS_SKY; ++i)
    {
        outdata << skyI[i] << " " << skyQ[i] << " " << skyU[i] << std::endl;
    }
    outdata.close();
    
	free(phi); free(theta); free(psi);
	free(skyI); free(skyQ); free(skyU); free(skyV);
	free(data);

	return 0;
}
