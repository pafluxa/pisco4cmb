#include "convolver.hpp"
#include "sphtrigo.hpp"

#include <omp.h>
#include <cstring>

#include <pointing.h>
#include <arr.h>
#include <rangeset.h>

Convolver::Convolver( unsigned long nsamples, unsigned int nthreads )
{
	_nsamples = nsamples;
	_nthreads = nthreads;
	
	// take no risk and make all chunks as large as needed
	int chunkSize = _nsamples/_nthreads;
	int reminder  = _nsamples%_nthreads;
	int samps = chunkSize + reminder;
	
	size_t perThreadBuffSize = sizeof(float)*(samps);
	
	int i;
	for( i=0; i<_nthreads; i++ ) {
		float *tb = (float*)malloc( perThreadBuffSize );
		_tempBuffers.push_back( tb );
	}
}

Convolver::~Convolver()
{
	int i;
	for( i=0; i<_nthreads; i++ ) {
		float *tb = _tempBuffers.back();
		free( tb );
		_tempBuffers.pop_back();
	}
}

void Convolver::exec_convolution(
	float* convData,
	Scan& scan, 
	Sky& sky,
	PolBeam& beam ) {

    float res;
    unsigned long s;
    unsigned long local;
    
    float phi; 
    float theta; 
    float psi;
    float* tempConvData;

	int t;
	long start; 
    long end; 
    long delta;
	size_t buffsize;
    
    // convenience pointers
	const float* phicrd = scan.get_phi_ptr();
	const float* thetacrd = scan.get_theta_ptr();
	const float* psicrd  = scan.get_psi_ptr();
	
	// shared vector to allow reconstruction of data
	// even entries correspond to the start sample of
	// thread T, odd entries correspond to the end sample
	int startEnd[2*_nthreads];
	
	// set number of threads (1 by default)
	omp_set_num_threads( _nthreads );
	
    // begin parallel region
	#pragma omp parallel default(shared)
	{ 
        // these variables must live inside the parallel block
        // in order to make them private by default
        int start, end;
        int thrId  = omp_get_thread_num();
        int stride = omp_get_num_threads();
        int chunkSize  = (scan.get_nsamples() / stride );
        int reminder   = (scan.get_nsamples() % stride );
    
        start = thrId * chunkSize;
        if( thrId + 1 == stride ) 
        {
            end = start + chunkSize + reminder; 
        }
        else 
        {
            end = start + chunkSize;
        }
        // directive to ensure atomic operation
        #pragma omp critical
        startEnd[2*thrId]   = start;
        // directive to ensure atomic operation
        #pragma omp critical
        startEnd[2*thrId+1] =   end;
        
        tempConvData = _tempBuffers[thrId];

        local = 0;
        for( s=start; s<end; s++, local++) {
            phi   =   phicrd[s]; 
            theta = thetacrd[s]; 
            psi   =   psicrd[s];
            res = _beam_times_sky( sky, beam, phi, theta, psi );
            tempConvData[local] = res;
        }

	} 
	// end parallel region
    
	// rebuild convolution buffer
	for( t=0; t<_nthreads; t++) 
    {
		start = startEnd[2*t];
		end   = startEnd[2*t+1];
		delta = end - start;
		buffsize = sizeof(float)*delta;
		
		const float* td = _tempBuffers.at(t);
		memcpy(static_cast<void*>(convData + start), 
		       static_cast<const void*>(td),
		       buffsize);
	}
}

void Convolver::_beam_times_sky( 
	Sky& sky, 
	PolBeam& pb, 
	float phi0, float theta0, float psi0,
    float da, double db)
{
	double data_a;
    double data_b;
  
	double ww;
    double rmax;
    
    double ra_bc;
    double dec_bc;
    double pa_bc;
    
	double ra_pix; 
    double dec_pix;
    
    double rho;
    double sigma;
    double chi;
	double c2chi;
    double s2chi;
    
    double ia;
    double qacos;
    double qasin;
    double uacos;
    double uasin;

    double ib;
    double qbcos;
    double qbsin;
    double ubcos;
    double ubsin;
    
    int range_begin;
    int range_end; 
    int rn;
    
    long i;
    long ni;
    long skyPix;

	fix_arr< int,    4 > neigh;
	fix_arr< double, 4 >   wgh;
	rangeset< int > intraBeamRanges;
	
	// re-named variables for compatilibity with original PISCO code.
	ra_bc  = phi0;
	dec_bc = M_PI/2.0 - theta0;
	pa_bc  = psi0;

	// find sky pixels around beam center up to rhoMax	
	rmax = beam.get_rho_max();
	pointing sc( theta0, ra_bc );
	sky.hpxBase.query_disc( sc, rmax, intraBeamRanges );
		
	// sky times beam multiplication loop
	data_a = 0.0;
	data_b = 0.0;
	for( rn=0; rn < intraBeamRanges.nranges(); rn++ )
	{
		range_begin = intraBeamRanges.ivbegin( rn );
		range_end   = intraBeamRanges.ivend  ( rn );
		
		for( skyPix=range_begin; skyPix < range_end; skyPix++ )
		{	
			pointing sp = sky.hpxBase.pix2ang( skyPix );
			
			ra_pix = sp.phi;
			dec_pix = M_PI/2.0 - sp.theta;
			
			// safety initializers
			rho=0.0; sigma=0.0; chi=0.0;
			SphericalTransformations::rho_sigma_chi_pix( 
				&rho,&sigma,&chi,
				ra_bc , dec_bc, pa_bc,
				ra_pix, dec_pix );
			
			// interpolate beam at (rho,sigma)
			pointing bp( rho, sigma );
			beam.hpxBase.get_interpol( bp, neigh, wgh );
			
			ia = 0.0; 
            qa = 0.0; 
            ua = 0.0;
            va = 0.0;
			for( i=0; i<4; i++ ) {
                n = neigh[i];
                ww = wgh[i];
				
                ia += pb.Da_I[ni] * ;
				qacos += Da_Qcos[ni] * ww;
				qasin += Da_Qsin[ni] * ww;
				uacos += Da_Ucos[ni] * ww;
				uasin += Da_Usin[ni] * ww;

				ib += pb.Db_I[ni] * ;
				qbcos += Db_Qcos[ni] * ww;
				qbsin += Db_Qsin[ni] * ww;
				ubcos += Db_Ucos[ni] * ww;
				ubsin += Db_Usin[ni] * ww;
			}
			
            c2chi = cos(2*chi);
            s2chi = sin(2*chi);
			
            data_a = data_a 
                  + sI[skyPix]*ia 
                  + sQ[skyPix]*qacos*c2chi + qasin*s2chi)
                  + sU[skyPix]*uacos*c2chi + uasin*s2chi);
			data_b = data_b
                  + sI[skyPix]*ib 
                  + sQ[skyPix]*qbcos*c2chi + qbsin*s2chi)
                  + sU[skyPix]*ubcos*c2chi + ubsin*s2chi);
		}
	}
	
	(*da) = (float)(data_a);
    (*db) = (float)(data_b);
}
