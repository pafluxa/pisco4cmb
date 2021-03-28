#include <omp.h>

#include <math.h>

#include <pointing.h>

#include "polbeam.hpp"
#include "convolver.hpp"

// for high_resolution_clock
#include <chrono>  

#include <array>

using namespace Convolution;
using namespace PolarizedBeam;

#define NSAMPLES (1179648/2)

int main( void )
{
	double rhomax = (5.0/180.0 * M_PI);
	PolBeam beam( 512, rhomax );
	beam.make_unpol_gaussian_elliptical_beam( 1.5, 1.5, 0.0 );
		
	// setup sky
	int NSIDE = 128;
	int NPIX  = 12*(NSIDE*NSIDE);
	arr32 sky( NPIX );
	sky.fill( 0.0 );
	
	// add a single non zero pixel at dec=0 RA=90
	Healpix_Base hpx;
	hpx.SetNside( NSIDE , RING );
	int pix = hpx.zphi2pix( 0.0, M_PI_2 );
	sky[ pix ] = 1.0;
	
	std::array< double, 16*NSAMPLES> gpu_result;
	std::array< double, 16*NSAMPLES> cpu_result;
	
	auto start  = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed;
	start  = std::chrono::high_resolution_clock::now();
	
	// create dummy scanning
	double phiMin = M_PI/2.0 - 0.1;
	double phiMax = M_PI/2.0 + 0.1;
	
	int ns = 4;
	omp_set_num_threads( ns );

	Convolver conv( &beam, NSIDE, sky, true, ns );
	start  = std::chrono::high_resolution_clock::now();
	
	start  = std::chrono::high_resolution_clock::now();
	#pragma omp parallel default( shared )
    {
		double phi;
		
	    int pix;
	    int tid    = omp_get_thread_num();
	    int stride = omp_get_num_threads();
	    
	    int chunkSize  = (NSAMPLES / stride );
	    int reminder   = (NSAMPLES % stride );
	    
	    int end;
	    int start = tid * chunkSize;
	    if( tid + 1 == stride )
	    {
	        end = start + chunkSize + reminder; 
	    }
	    else
	    {
	        end = start + chunkSize;
	    }
	    
//#pragma omp critical
//	    std::cout << tid << ":" << start << " " << end << std::endl;
		
		double res = 0.0;
		for( int i=start; i < end; i++ )
		{
			phi = phiMin + (0.2 * i)/NSAMPLES;
			res = conv.cuda_convolve( phi, 0.0, 0.0, tid );
			gpu_result[ 16*i + tid ] = res;
		}
	
	} //end parallel region
	
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	std::cout << "#Convolution took " << elapsed.count() << " s on GPU\n";
	
	ns = 1;
	omp_set_num_threads( ns );
	start  = std::chrono::high_resolution_clock::now();
	#pragma omp parallel
    {
		Convolver conv( &beam, NSIDE, sky );
		
	    int pix;
	    int tid    = omp_get_thread_num();
	    int stride = omp_get_num_threads();
	    
	    int chunkSize  = (NSAMPLES / stride );
	    int reminder   = (NSAMPLES % stride );
	    
	    int end;
	    int start = tid * chunkSize;
	    if( tid + 1 == stride )
	    {
	        end = start + chunkSize + reminder; 
	    }
	    else
	    {
	        end = start + chunkSize;
	    }
	    
	    //#pragma omp critical
	    //std::cout << tid << ":" << start << " " << end << std::endl;
	
		double res = 0.0;
		for( int i=start; i < end; i++ )
		{
//#pragma omp critical
//			std::cout << i << std::endl;

			double phi = phiMin + (0.2 * i)/NSAMPLES;
			res = conv.convolve( phi, 0.0, 0.0 );
			cpu_result[ 16*i + tid ] = d;
		}
		
//#pragma omp critical
//		std::cout << "out of loop." << std::endl;
	
	} //end parallel region
	
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	std::cout << "#Convolution took " << elapsed.count() << " s on CPU\n";
	
	for( int i=0; i < NSAMPLES; i++ )
	{
		double resc = 0.0;
		double resg = 0.0;
		for( int j=0; j < 1; j++ ) {
			resc += cpu_result[ 16*i + j ];
			resg += gpu_result[ 16*i + j ];	
		}
		std::cout << i << " " << cpu_result[i] << " " << gpu_result[i] << std::endl;
	}
	return 0;
}
