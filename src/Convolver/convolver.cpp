#include "Convolver/convolver.hpp"
#include "Sphtrigo/sphtrigo.hpp"

#include <omp.h>
#include <cstring>

#include <pointing.h>
#include <arr.h>
#include <rangeset.h>

Convolver::Convolver(unsigned long _nsamples, unsigned int _nthreads)
{
	nsamples = _nsamples;
	nthreads = _nthreads;	
    masterBuffer = (float*)malloc(sizeof(float)*nsamples);
	
    int chunkSize = nsamples/nthreads;
    int it = 0;	
    int s = 0;
    int e = chunkSize;
    while(it < nthreads - 1)
    {
        bufferStart.push_back(s);
        bufferEnd.push_back(e);
        s += chunkSize;
        e += chunkSize;        
        it += 1;
	}
    chunkSize = nsamples - nthreads*chunkSize;
    s += chunkSize;
    e += chunkSize;
    bufferStart.push_back(s);
    bufferEnd.push_back(e);    
}

Convolver::~Convolver()
{
    //free(masterBuffer);
    bufferStart.clear();
    bufferEnd.clear();
}

void Convolver::exec_convolution(
	float* tod_data,
	Scan& scan, 
	Sky& sky,
	PolBeam& beam, char polFlag) {

    float res;
    unsigned long s;
    unsigned long local;

    // convenience pointers
	const double* phicrd = scan.get_phi_ptr();
	const double* thetacrd = scan.get_theta_ptr();
	const double* psicrd  = scan.get_psi_ptr();

    // begin parallel region
	//#pragma omp parallel for
    #pragma omp parallel for
    for(long i = 0; i < nsamples; i++) 
    {
        float data = 0.0;
        double phi   =   phicrd[i]; 
        double theta = thetacrd[i]; 
        double psi   =   psicrd[i];
        beam_times_sky(sky, beam, polFlag, phi, theta, psi, &data);
        tod_data[i] = data;
    }    
}

#pragma omp declare simd
void Convolver::beam_times_sky( 
	Sky& sky, 
	PolBeam& beam, char polFlag, 
	float phi0, float theta0, float psi0,
    float* tod_data)
{
	double data;
  
	double ww;
    double rmax;
    
    double ra_bc, dec_bc, pa_bc;
	double ra_pix, dec_pix;
    
    double rho, sigma, chi, c2chi, s2chi;
    
    double ia, qacos, qasin, uacos, uasin;
    double ib, qbcos, qbsin, ubcos, ubsin;
    
    int range_begin, range_end, rn;
    
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
