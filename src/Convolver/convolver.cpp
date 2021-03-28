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
		float *tba = (float*)malloc( perThreadBuffSize );
		tempBuffer.push_back(tba);
	}
}

Convolver::~Convolver()
{
	int i;
	for( i=0; i<_nthreads; i++ ) {
		float *tba = tempBuffer.back();
		free(tba);
		tempBuffer.pop_back();
	}
}

void Convolver::exec_convolution(
	float* tod_data,
	Scan& scan, 
	Sky& sky,
	PolBeam& beam, char polFlag) {

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
        float data;
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
        
        tempConvData = tempBuffer[thrId];

        local = 0;
        for( s=start; s<end; s++, local++) {
            phi   =   phicrd[s]; 
            theta = thetacrd[s]; 
            psi   =   psicrd[s];
            beam_times_sky(sky, beam, polFlag, phi, theta, psi, &data);
            tempConvData[local] = data;
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
		
		const float* td = tempBuffer.at(t);
		memcpy(static_cast<void*>(tod_data + start), 
		       static_cast<const void*>(td),
		       buffsize);
	}
}

void Convolver::beam_times_sky( 
	Sky& sky, 
	PolBeam& beam, char polFlag, 
	float phi0, float theta0, float psi0,
    float* tod_data)
{
	double data;
  
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
	data = 0.0;
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
            qacos = 0.0; 
            qasin = 0.0;
            uacos = 0.0;
            uasin = 0.0;
            
			ib = 0.0; 
            qbcos = 0.0; 
            qbsin = 0.0;
            ubcos = 0.0;
            ubsin = 0.0;
			for( i=0; i<4; i++ ) {
                ni = neigh[i];
                ww = wgh[i];
				
                if(polFlag == 'a')
                {
                    ia += beam.Da_I[ni] * ww;
                    qacos += beam.Da_Qcos[ni] * ww;
                    qasin += beam.Da_Qsin[ni] * ww;
                    uacos += beam.Da_Ucos[ni] * ww;
                    uasin += beam.Da_Usin[ni] * ww;
                }
                
                if(polFlag == 'b')
                {
                    ib += beam.Db_I[ni] * ww;
                    qbcos += beam.Db_Qcos[ni] * ww;
                    qbsin += beam.Db_Qsin[ni] * ww;
                    ubcos += beam.Db_Ucos[ni] * ww;
                    ubsin += beam.Db_Usin[ni] * ww;
                }
			}
			
            c2chi = cos(2*chi);
            s2chi = sin(2*chi);
			
            if(polFlag == 'a')
            {
                data = data
                    + sky.sI[skyPix]*ia 
                    + sky.sQ[skyPix]*(qacos*c2chi + qasin*s2chi)
                    + sky.sU[skyPix]*(uacos*c2chi + uasin*s2chi);
            }
            
            if(polFlag == 'b')
            {
                data = data
                      + sky.sI[skyPix]*ib 
                      + sky.sQ[skyPix]*(qbcos*c2chi + qbsin*s2chi)
                      + sky.sU[skyPix]*(ubcos*c2chi + ubsin*s2chi);
            }
		}
	}
	
	(*tod_data) = (float)(data);
}
