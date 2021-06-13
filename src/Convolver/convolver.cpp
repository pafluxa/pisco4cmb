#include "Convolver/convolver.hpp"
#include "Sphtrigo/sphtrigo.hpp"

#include <omp.h>
#include <cstring>

#include <pointing.h>
#include <arr.h>
#include <rangeset.h>

Convolver::Convolver(const Scan* _scan, const Sky* _sky, const PolBeam* _beam)
{
    scan = _scan;
    sky = _sky;
    beam = _beam;
}

Convolver::~Convolver()
{
}

void Convolver::exec_convolution(
    char polFlag, float* data_a, float* data_b)
{
    // convenience pointers
    const float* ra_coords = scan->get_ra_ptr();
    const float* dec_coords = scan->get_dec_ptr();
    const float* pa_coords  = scan->get_pa_ptr();

    #ifdef CONVOLVER_DISABLECHI
    std::cerr << "using chi = pa_bc!" << std::endl;
    #endif

    // this is done so that OpenMP doesn't go bananas on calling the
    // function from Scan()
    long nsamples = scan->get_nsamples();
    // begin parallel region
    #ifdef CONVOLVER_OPENMPENABLED
    #pragma omp parallel for
    #endif
    for(long i = 0; i < nsamples; i++)
    {
        // declare these internally because they only
        // live inside the parallel region
        double da = 0.0;
        double db = 0.0;
        double ra_bc   = ra_coords[i];
        double dec_bc = dec_coords[i];
        // Passed arguments are counterclockwise on the sky, while
        // CMB requires clockwise arguments.
        double pa_bc = -pa_coords[i];
        beam_times_sky(ra_bc, dec_bc, pa_bc, &da, &db);
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
    double ra_bc, double dec_bc, double pa_bc,
    double* da, double* db)
{
    double ww;
    double rmax;
    double data_a;
    double data_b;

    double ra_pix, dec_pix;
    double rho, sigma, chi, c2chi, s2chi;

    double beam_a[3];
    double beam_b[3];

    long i, ni, skyPix;
    int range_begin, range_end, rn;

    fix_arr< int,    4 > neigh;
    fix_arr< double, 4 >   wgh;
    rangeset<int> intraBeamRanges;

    // find sky pixels around beam center, up to beam.rhoMax
    rmax = beam->get_rho_max();
    pointing sc(M_PI_2 - dec_bc, ra_bc);
    sky->hpxBase.query_disc(sc, rmax, intraBeamRanges);
    // sky times beam multiplication loop
    data_a = 0.0;
    data_b = 0.0;
    // for every sky pixel in the beam
    for( rn=0; rn < intraBeamRanges.nranges(); rn++ )
    {
        range_begin = intraBeamRanges.ivbegin(rn);
        range_end   = intraBeamRanges.ivend(rn);
        for(skyPix = range_begin; skyPix < range_end; skyPix++)
        {
            // get pointing of sky pixel
            pointing sp = sky->hpxBase.pix2ang(skyPix);
            ra_pix = sp.phi;
            dec_pix = M_PI/2.0 - sp.theta;
            // safety initializers
            rho=0.0; sigma=0.0; chi=0.0;
            // compute rho sigma and chi at beam pixel
            SphericalTransformations::rho_sigma_chi_pix(
                &rho, &sigma, &chi,
                ra_bc, dec_bc, pa_bc,
                ra_pix, dec_pix);
            #ifdef CONVOLVER_DISABLECHI
            chi = pa_bc;
            #endif
            c2chi = cos(2.0*chi);
            s2chi = sin(2.0*chi);
            // safety initializers
            std::memset(beam_a, 0.0, sizeof(double)*3);
            std::memset(beam_b, 0.0, sizeof(double)*3);
            // interpolate beam at (rho,sigma)
            pointing bp(rho, sigma);
            beam->hpxBase.get_interpol(bp, neigh, wgh);
            for(int b=0; b < 3; b++)
            {
                // keep track of the sum of weigths
                double ws = 0.0;
                for(int i=0; i < 4; i++)
                {
                    ni = neigh[i];
                    if(ni < beam->size())
                    {
                        ww = wgh[i];
                        ws += ww;
                        beam_a[b] += double(beam->aBeams[b][ni])*ww;
                        beam_b[b] += double(beam->bBeams[b][ni])*ww;
                    }
                }
                if(ws > 0.0)
                {
                    beam_a[b] /= ws;
                    beam_b[b] /= ws;
                }
            }
            /**
            int beampix = beam->hpxBase->ang2pix(bp);
            beam_a[0] = beam->aBeams[0][beampix];
            beam_a[1] = beam->aBeams[1][beampix];
            beam_a[2] = beam->aBeams[2][beampix];
            beam_b[0] = beam->bBeams[0][beampix];
            beam_b[1] = beam->bBeams[1][beampix];
            beam_b[2] = beam->bBeams[2][beampix];
            **/
            data_a = data_a
              + sky->sI[skyPix]*(beam_a[0])
              + sky->sQ[skyPix]*(beam_a[1]*c2chi - beam_a[2]*s2chi)
              + sky->sU[skyPix]*(beam_a[2]*c2chi + beam_a[1]*s2chi);

            data_b = data_b
              + sky->sI[skyPix]*(beam_b[0])
              + sky->sQ[skyPix]*(-beam_b[1]*c2chi + beam_b[2]*s2chi)
              + sky->sU[skyPix]*(-beam_b[2]*c2chi - beam_b[1]*s2chi);

            #ifdef CONVOLVER_DEBUG
            if((abs(ra_bc - M_PI + 0.3 / 32.0) < 1.0e-4) && (abs(dec_bc - 0.3 / 32.0) < 1.0e-4) && (sky->sI[skyPix] != 0.0)) 
            {
                printf("Pol    %lf %lf  %lf\n", sky->sI[skyPix], sky->sQ[skyPix], sky->sU[skyPix]);
                printf("Angles %lf %lf \n", c2chi, s2chi);
                double degr = 180.0 / M_PI;
                printf("ra_bc %lf dec_bc %lf ra_pix %lf dec_pix %lf \n", ra_bc * degr, dec_bc * degr, ra_pix* degr, dec_pix* degr);
                printf("rho %lf sigma %lf \n", rho * degr, sigma * degr);
                printf("BeamA  %.9le %.9le %.9le \n", beam_a[0], beam_a[1], beam_a[2]);  
                printf("BeamB  %.9le %.9le %.9le \n", beam_b[0], beam_b[1], beam_b[2]); 
                printf("Data   %.9le %.9le \n\n", data_a, data_b);
            }
            #endif            
        }
    }
    (*da) = data_a;
    (*db) = data_b;
}
