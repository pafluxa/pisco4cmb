/*
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

#define DUMP_POL_BEAMS true
#define NPSB 1 
#define NDETS (2*NPSB)
#define NSIDE_SKY 512
#define NPIXELS_SKY (12*NSIDE_SKY*NSIDE_SKY)
#define NSAMPLES_GRID 512
// uncomment for point source scanning
#define NSAMPLES NSAMPLES_GRID*NSAMPLES_GRID
// uncomment for whole sky convolution
//#define NSAMPLES NPIXELS_SKY

#define NSIDE_BEAM 2048
/* how to calculate the number of pixels below
import healpy
import numpy
print(healpy.query_disc(2048, (0,0,1), numpy.radians(5)).size)
#95484
*/
#define NPIXELS_BEAM (95484)

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

int main(void)
{
    std::ofstream outdata;
    // timing
    auto start  = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed;

    // load sky from disk
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
        in >> skyI[i] >> skyQ[i] >> skyU[i] >> skyV[i] && !in.eof()
    ) { i++;}
    in.close();
    Sky sky(NSIDE_SKY, skyI, skyQ, skyU, skyV);

    // setup PSB beams
    PolBeam beam(NSIDE_BEAM, NPIXELS_BEAM);
    // load beam from disk
    i = 0;
    std::string line;
    std::ifstream detAFields("data/beams/hpx_tilde_beams_detector_x.txt");
    while(std::getline(detAFields, line) && i < NPIXELS_BEAM)
    {
        std::istringstream iss(line);
        // stop if an error while parsing ocurrs
        if(!(iss 
             >> beam.Ia[i] 
             >> beam.Qa[i] 
             >> beam.Ua[i] 
             >> beam.Va[i]))
        { 
            break; 
        } 
        i++;
    }
    detAFields.close();
    // load and buld beam for detector B
    i = 0;
    //std::cout << "running with no cross-polarization!" << std::endl;
    std::ifstream detBFields("data/beams/hpx_tilde_beams_detector_x.txt");
    while(std::getline(detBFields, line) && i < NPIXELS_BEAM)
    {
        std::istringstream iss(line);
        // stop if an error while parsing ocurrs
        if(!(iss 
             >> beam.Ib[i] 
             >> beam.Qb[i] 
             >> beam.Ub[i] 
             >> beam.Vb[i]))
        { 
            break; 
        } 
        i++;
    }
    detBFields.close();
    // execute build of polarized beams
    beam.build_beams();
    // setup pointing
    double *ra;
    double *dec;
    double *psi;
    ra  = (double*)malloc(sizeof(double)*NSAMPLES);
    dec = (double*)malloc(sizeof(double)*NSAMPLES);
    psi = (double*)malloc(sizeof(double)*NSAMPLES);
    Scan scan(NSAMPLES, ra, dec, psi);
    // convolver object
    Convolver cconv(NSAMPLES);
    // mapping stuff, don't mind this
    int detector_mask[1] = {0};
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
    std::memset(AtA, 0, sizeof(double)*9*mapNpixels);
    std::memset(AtD, 0, sizeof(double)*3*mapNpixels);
    // execute convolution
    double detector_angle[1]; 
    double position_angles[3] = {-M_PI_4, 0.0, M_PI_4};
    // setup buffer to store time ordered data
    float *data_a;
    float *data_b;
    data_a = (float*)malloc(sizeof(float)*NSAMPLES);
    data_b = (float*)malloc(sizeof(float)*NSAMPLES);
	// every PSB scans the sky 3 times at three different angles
	for(double bcpa: position_angles)
	{
	    // zero-out data buffer
	    std::memset(data_a, 0, sizeof(float)*NSAMPLES);
	    std::memset(data_b, 0, sizeof(float)*NSAMPLES);
	    // initialize sky with psi = bcpa + detangle
        // uncomment below for whole sky convolution
	    //init_scan_whole_sky(bcpa, ra, dec, psi);
	    // uncomment for point source scanning
        init_point_source_scan(M_PI, 0.0, bcpa, 0.2, 0.2, ra, dec, psi);
	    start  = std::chrono::high_resolution_clock::now();
	    // compute convolution for detector A of PSB
	    cconv.exec_convolution(
		data_a, data_b,
		'p',
		scan, sky, beam);
	    finish = std::chrono::high_resolution_clock::now();
	    elapsed = finish - start;
	    std::cerr << "#GPU Convolution took "
		      << elapsed.count() << " sec\n";
	    // project detector data to map-making matrices
        detector_angle[0] = 0.0;
        libmapping_project_data_to_matrices
	    (
            NSAMPLES, 1,
            ra, dec, psi,
            detector_angle,
            data_a, scanMask, detector_mask,
            NSIDE_SKY, mapNpixels, mapPixels,
            AtA, AtD
	    );
        
        detector_angle[0] = M_PI_2;
	    libmapping_project_data_to_matrices
	    (
            NSAMPLES, 1,
            ra, dec, psi,
            detector_angle,
            data_b, scanMask, detector_mask,
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
    for(int i=0; i<NPIXELS_SKY; i++)
    {
        outdata << skyI[i] << " " << 
                   skyQ[i] << " " << 
                   skyU[i] << std::endl;
    }
    outdata.close();

    return 0;
}
