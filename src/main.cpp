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

#define NSIDE_SKY 128
#define NPIXELS_SKY (12*NSIDE_SKY*NSIDE_SKY)
// this number needs to be a perfect square for the test to work out!
// this is the number of pixels in an nside=128 healpix map
#define NSAMPLES_GRID 128
#define NSAMPLES NPIXELS_SKY
//#define NSAMPLES NSAMPLES_GRID*NSAMPLES_GRID


#define NSIDE_BEAM 512
// healpy.query_disc( 512, (0,0,1), numpy.radians(5) ).size
#define NPIXELS_BEAM (5940)

void init_point_source_scan(
    double ra0, double dec0, double pa0,
    double dra, double ddec,
    double* ra, double* dec, double* psi)
{
    double ra_bc;
    double dec_bc;
    double ndec = double(NSAMPLES_GRID);
    double nra = double(NSAMPLES_GRID);

    for(int i=0; i < NSAMPLES_GRID; i++)
    {
        dec_bc = (dec0 - ddec) + 2*ddec*(double(i)/ndec);
        //std::cout << dec_bc << std::endl;
        for(int j=0; j < NSAMPLES_GRID; j++)
        {
            ra_bc = (ra0 - dra) + 2*dra*(double(j)/nra);

            ra[i*NSAMPLES_GRID + j] = ra_bc;
            dec[i*NSAMPLES_GRID + j] = dec_bc;
            psi[i*NSAMPLES_GRID + j] = pa0;
        }
    }
}


void init_scan_whole_sky(
    double psi0,
    double* ra, double* dec, double* psi)
{
    Healpix_Base hpxBase;
    hpxBase.SetNside(NSIDE_SKY, RING);

    for(int pix=0; pix < NPIXELS_SKY; pix++)
    {
        pointing bp = hpxBase.pix2ang(pix);
        dec[pix] = M_PI_2 - bp.theta;
        ra[pix] = bp.phi;
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

    // setup sky
    float *skyI, *skyQ, *skyU, *skyV;
    skyI = (float*)malloc(NPIXELS_SKY*sizeof(float));
    skyQ = (float*)malloc(NPIXELS_SKY*sizeof(float));
    skyU = (float*)malloc(NPIXELS_SKY*sizeof(float));
    skyV = (float*)malloc(NPIXELS_SKY*sizeof(float));
    // read in sky from maps_input.txt
    int i = 0;
    std::ifstream in("maps_input.txt");
    while
    (
        in >> skyI[i] >> skyQ[i] >> skyU[i] >> skyV[i] && \
        !in.eof()
    ) { i++; }
    in.close();
    Sky sky(NSIDE_SKY, skyI, skyQ, skyU, skyV);

    // setup TOD
    float *data;
    data = (float*)malloc(sizeof(float)*NSAMPLES);

    // setup beam
    PolBeam beam(NSIDE_BEAM, NPIXELS_BEAM);
    beam.make_unpol_gaussian_elliptical_beams(2.0, 2.0, 0.0);
    beam.build_beams();

    // setup pointing
    double *ra;
    double *dec;
    double *psi;
    ra  = (double*)malloc(sizeof(double)*NSAMPLES);
    dec = (double*)malloc(sizeof(double)*NSAMPLES);
    psi = (double*)malloc(sizeof(double)*NSAMPLES);
    Scan scan( NSAMPLES, ra, dec, psi);

    // convolver object
    Convolver* cconv = new Convolver(NSAMPLES);

    // setup mapping stuff
    int* scanMask = (int*)malloc(sizeof(int)*NSAMPLES);
    std::memset(scanMask, 0, sizeof(int)*NSAMPLES);
    long mapNpixels = NPIXELS_SKY;
    int* mapPixels = (int*)malloc(sizeof(int)*mapNpixels);
    for(long i=0; i < mapNpixels; i++)
    {
        mapPixels[i] = i;
    }
    double* AtA = (double*)malloc(sizeof(double)*9*mapNpixels);
    double* AtD = (double*)malloc(sizeof(double)*3*mapNpixels);
    int detuids[1] = {0};

    // execute convolution at different position angles
    double angles[3] = {-M_PI_4, 0.0, M_PI_4};
    for(double polangle: angles)
    {
        init_scan_whole_sky(polangle, ra, dec, psi);
        //init_point_source_scan(M_PI, 0.0, polangle, 0.2, 0.2, ra, dec, psi);
        start  = std::chrono::high_resolution_clock::now();
        cconv->exec_convolution(
            data, nullptr,
            'a',
            scan, sky, beam);
        finish = std::chrono::high_resolution_clock::now();
        elapsed = finish - start;
        std::cerr << "#CPU Convolution took " << elapsed.count() << " sec\n";

        // project to map-making matrices
        double det_polangles[1];
        det_polangles[0] = 0.0;
        libmapping_project_data_to_matrices
        (
            NSAMPLES, 1,
            ra, dec, psi,
            det_polangles,
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

    //free(ra);
    //free(dec);
    //free(psi);

    //free(skyI);
    //free(skyQ);
    //free(skyU);
    //free(skyV);

    //free(data);

    return 0;
}
