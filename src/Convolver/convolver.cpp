#include "Convolver/convolver.hpp"
#include "Sphtrigo/sphtrigo.hpp"

#include <omp.h>
#include <cstring>

#include <pointing.h>
#include <arr.h>
#include <rangeset.h>

Convolver::Convolver(unsigned long _nsamples)
{
    // get active number of threads
    nthreads = omp_get_num_threads();	
    nsamples = _nsamples;	
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
    bufferStart.clear();
    bufferEnd.clear();
}

void Convolver::exec_convolution(
	float* data_a, float* data_b, 
    char polFlag,
	Scan& scan, 
	Sky& sky,
	PolBeam& beam) 
{
    // convenience pointers
	const double* ra_coords = scan.get_ra_ptr();
	const double* dec_coords = scan.get_dec_ptr();
	const double* pa_coords  = scan.get_pa_ptr();

    // begin parallel region
    #pragma omp parallel for
    for(long i = 0; i < nsamples; i++) 
    {
        // declare these internally because they only
        // live inside the parallel region
        float da = 0.0;
        float db = 0.0;
        double ra_bc   = ra_coords[i]; 
        double dec_bc = dec_coords[i]; 
        double pa_bc   = pa_coords[i];
        
        beam_times_sky(
            sky, beam, 
            ra_bc, dec_bc, pa_bc, 
            &da, &db);
        
        if(polFlag == 'a')
        {   
            data_a[i] = da;
        }
        else if (polFlag == 'b')
        {
            data_b[i] = db;
        }
        else if (polFlag == 'p')
        {
            data_a[i] = da;
            data_b[i] = db;
        }
        else
        {
            throw std::invalid_argument(\
                "valid polFlags are 'a', 'b' and 'p'" );
        }
    }    
}

void Convolver::beam_times_sky( 
	Sky& sky, 
    PolBeam& beam, 
	float ra_bc, 
    float dec_bc, \
    float pa_bc,
    float* da, float* db)
{
	double ww;
    double rmax;
 	double data_a;
 	double data_b;
    
	double ra_pix, dec_pix;
    double rho, sigma, chi, c2chi, s2chi;
    
    double beam_a[5]; 
    double beam_b[5];

    long i, ni, skyPix;
    int range_begin, range_end, rn;
    
	fix_arr< int,    4 > neigh;
	fix_arr< double, 4 >   wgh;
	rangeset<int> intraBeamRanges;
	
    // find sky pixels around beam center, up to beam.rhoMax	
	rmax = beam.get_rho_max();
	pointing sc(M_PI_2 - dec_bc, ra_bc);
	sky.hpxBase.query_disc(sc, rmax, intraBeamRanges);
		
	// sky times beam multiplication loop
	data_a = 0.0;
	data_b = 0.0;
    // for every sky pixel in the beam
	for( rn=0; rn < intraBeamRanges.nranges(); rn++ )
	{
		range_begin = intraBeamRanges.ivbegin( rn );
		range_end   = intraBeamRanges.ivend  ( rn );
		for( skyPix=range_begin; skyPix < range_end; skyPix++ )
		{	
            // get pointing of sky pixel
			pointing sp = sky.hpxBase.pix2ang(skyPix);
			ra_pix = sp.phi;
			dec_pix = M_PI/2.0 - sp.theta;
			
			// safety initializers
			rho=0.0; sigma=0.0; chi=0.0;
            // compute rho sigma and chi at beam pixel
			SphericalTransformations::rho_sigma_chi_pix( 
				&rho, &sigma, &chi,
				ra_bc , dec_bc, pa_bc,
				ra_pix, dec_pix );
            c2chi = cos(2*chi);
            s2chi = sin(2*chi);
            // safety initializers
            std::memset(beam_a, 0.0, sizeof(double)*5);
            std::memset(beam_b, 0.0, sizeof(double)*5);
            // interpolate beam at (rho,sigma)
			pointing bp(rho, sigma);
			beam.hpxBase.get_interpol(bp, neigh, wgh);
			for(int b=0; b < 5; b++)
            {
                for(int i=0; i < 4; i++) 
                {
                    ni = neigh[i];
                    if(ni < beam.size())
                    {
                        ww = wgh[i];
                        beam_a[b] += double(beam.aBeams[b][ni])*ww;
                        beam_b[b] += double(beam.bBeams[b][ni])*ww;
                    }
                }
            }
            // data = beam x sky
            data_a = data_a 
              + sky.sI[skyPix]*(beam_a[0])
              + sky.sQ[skyPix]*(beam_a[1]*c2chi - beam_a[2]*s2chi)
              + sky.sU[skyPix]*(beam_a[3]*c2chi + beam_a[4]*s2chi);
            data_b = data_b
              + sky.sI[skyPix]*(beam_b[0])
              + sky.sQ[skyPix]*(-beam_b[1]*c2chi + beam_b[2]*s2chi)
              + sky.sU[skyPix]*(-beam_b[3]*c2chi - beam_b[4]*s2chi);
		}
	}
	(*da) = (float)(data_a);
	(*db) = (float)(data_b);
}
