/**
 * This program simulates receives the coordinates and polarization of
 * a point source, beams for a PSB and other parameters relative to
 * map resolution and input/output to produce another map with the
 * result of the PSB "raster scanning" that point source at three
 * different position angles.
 *
 */
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <unistd.h>

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Bpoint/BPoint.h"
#include "Polbeam/polbeam.hpp"
#include "Convolver/convolver.hpp"
#include "Mapping/mapping_routines.h"

// needed to create the point source map
#include <healpix_base.h>
#include <pointing.h>

#define NSIDE_SKY (512)
#define NPIXELS_SKY (12*(NSIDE_SKY)*(NSIDE_SKY))
#define NSAMPLES (NPIXELS_SKY)
#define NSIDE_BEAM 2048
/** how to calculate the number of pixels below
import healpy
import numpy
print(healpy.query_disc(2048, (0,0,1), numpy.radians(5)).size)
#95484
**/
#define NPIXELS_BEAM (95484)

/** buffers to store the sky **/
float* skyI;
float* skyQ;
float* skyU;
float* skyV;
/** buffers so store PSB pointing **/
double* ra;
double* dec;
double* psi;
/** map-making accumulation matrix (vector) **/
double* AtA;
double* AtD;
/** buffers to simulated data **/
// detector A
double* psbDataA;
// detector B
double* psbDataB;
/** buffers needed for mapping. */
int* scanMask;
int* mapPixels;
/**
 * routine to initialize the scan to a raster scan around a location
 * defined by ra0, dec0. the scan is executed at position angle psi0.
 **/
void init_wholesky_scan(
    int nside, double pa,
    // output
    double *ra, double *dec, double *psi);
/**routine to load sky from disk
 **/
void load_sky_data(std::string path, 
    float* skyI, float* skyQ, float* skyU, float* skyV);
/** allocate all memory needed for this program to work. **/
void allocate_everything(void);
/** deallocates everything. **/
void free_everything(void);
/** load beam data from disk **/
void load_beam_data(char polFlag, std::string path, PolBeam& beam);

int main(int argc, char** argv )
{
    // path to write the output map
    std::string outfilePath;
    // which detectors get projected
    char projDets[1];
    // polarization of input source
    float pI = 1.0;
    float pQ = 0.0;
    float pU = 0.0;
    bool beamsFromGauss = true;
    double bfwhmx = 0.0;
    double bfwhmy = 0.0;
    std::string inputSkyPath;
    std::string beamAPath;
    std::string beamBPath;
    int opt;
    while((opt = getopt(argc, argv, "t:m:o:p:a:b:")) != -1 ) {
        switch(opt)
        {
            case 'p':
                projDets[0] = char(optarg[0]);
                continue;
            case 'o':
                outfilePath.assign(optarg);
                continue;
            case 't':
                if(char(optarg[0]) == 'f'){
                    beamsFromGauss = false;
                }
                else if(char(optarg[0]) == 'g'){
                    beamsFromGauss = true;
                }
                else {
                    throw std::invalid_argument("error");
                }
            case 'a':
                beamAPath.assign(optarg);
                continue;
            case 'b':
                beamBPath.assign(optarg);
                continue;
            case 'm':
                inputSkyPath.assign(optarg);
                continue;
        }
    }
    // opens output file
    std::ofstream outdata;
    outdata.open(outfilePath);
    std::cout << inputSkyPath << std::endl;
    std::cout << projDets[0] << std::endl;
    std::cout << beamAPath << std::endl;
    std::cout << beamBPath << std::endl;
    std::cout << outfilePath << std::endl;
    allocate_everything();
    // initialize sky object
    load_sky_data(inputSkyPath, skyI, skyQ, skyU, skyV);
    Sky sky(NSIDE_SKY, skyI, skyQ, skyU, skyV);
    // initialize polarized beam object. read beam data from disk
    PolBeam beam(NSIDE_BEAM, NPIXELS_BEAM);
    if(beamsFromGauss) {
        bfwhmx = std::stod(beamAPath.c_str());
        bfwhmy = std::stod(beamBPath.c_str());
        beam.make_unpol_gaussian_elliptical_beams(bfwhmx, bfwhmy, 0.0);
    }
    else {
        load_beam_data('a', beamAPath, beam);
        load_beam_data('b', beamBPath, beam);
    }
    beam.build_beams();
    // initialize scan
    Scan scan(NSAMPLES, ra, dec, psi);
    // initialize convolver object
    Convolver cconv(&scan, &sky, &beam);
    // setup detetctor angle arays
    int detectorMask[] = {0};
    double detectorAngle[1];
    //double positionAngles[] = {-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90};
    double positionAngles[] = {-45, 0, 45};
    std::cout << positionAngles[0] << " ";
    std::cout << positionAngles[1] << " ";
    std::cout << positionAngles[2] << " ";
    std::cout << positionAngles[3] << " ";
    std::cout << positionAngles[4] << " " << std::endl;
    // every PSB scans the sky 3 times at three different angles
    for(double bcpa_deg: positionAngles)
    {
        // angles are in degrees. convert.
        double bcpa = M_PI*(bcpa_deg / 180.0);
        // zero-out data buffer
        std::memset(psbDataA, 0, sizeof(double)*NSAMPLES);
        std::memset(psbDataB, 0, sizeof(double)*NSAMPLES);
        init_wholesky_scan(NSIDE_SKY, bcpa, ra, dec, psi);
        // compute convolution for detector A and B of PSB
        cconv.exec_convolution('p', psbDataA, psbDataB);
        // project detector data to map-making matrices
        if(projDets[0] == 'a' || projDets[0] == 'p') {
            // project detector a
            detectorAngle[0] = 0;
            libmapping_project_data_to_matrices
            (
                NSAMPLES, 1,
                ra, dec, psi,
                detectorAngle,
                psbDataA, scanMask, detectorMask,
                NSIDE_SKY, NPIXELS_SKY, mapPixels,
                AtA, AtD
            );
        }
        if(projDets[0] == 'b' || projDets[0] == 'p') {
            // project detector b
            detectorAngle[0] = M_PI_2;
            libmapping_project_data_to_matrices
            (
                NSAMPLES, 1,
                ra, dec, psi,
                detectorAngle,
                psbDataB, scanMask, detectorMask,
                NSIDE_SKY, NPIXELS_SKY, mapPixels,
                AtA, AtD
            );
        }
    }
    std::memset(skyI, 0, sizeof(float)*NPIXELS_SKY);
    std::memset(skyQ, 0, sizeof(float)*NPIXELS_SKY);
    std::memset(skyU, 0, sizeof(float)*NPIXELS_SKY);
    std::memset(skyV, 0, sizeof(float)*NPIXELS_SKY);
    libmapping_get_IQU_from_matrices
    (
        NSIDE_SKY, NPIXELS_SKY,
        AtA, AtD, mapPixels,
        skyI, skyQ, skyU, skyV
    );
    for(int i=0; i<NPIXELS_SKY; i++)
    {
        outdata << skyI[i] << " " <<
                   skyQ[i] << " " <<
                   skyU[i] << " " <<
                   skyV[i] << std::endl;
    }
    outdata.close();
    free_everything();

    return 0;
}

void init_wholesky_scan(
    int nside, 
    double pa,
    double *ra, double *dec, double *psi) {
    
    int i;
    int npixels = 12 * nside * nside;
    Healpix_Base hpx(nside, RING, SET_NSIDE);
    
    for(i = 0; i < npixels; i++)
    {
        pointing ptg = hpx.pix2ang(i);
        ra[i] = ptg.phi;
        dec[i] = M_PI_2 - ptg.theta;
        psi[i] = pa;
    }
}

void allocate_everything(void){

    skyI = (float*)malloc(NPIXELS_SKY*sizeof(float));
    skyQ = (float*)malloc(NPIXELS_SKY*sizeof(float));
    skyU = (float*)malloc(NPIXELS_SKY*sizeof(float));
    skyV = (float*)malloc(NPIXELS_SKY*sizeof(float));

    ra  = (double*)malloc(sizeof(double)*NSAMPLES);
    dec = (double*)malloc(sizeof(double)*NSAMPLES);
    psi = (double*)malloc(sizeof(double)*NSAMPLES);

    AtA = (double*)malloc(sizeof(double)*9*NPIXELS_SKY);
    AtD = (double*)malloc(sizeof(double)*3*NPIXELS_SKY);
    // zero out map-making accumulation matrices
    std::memset(AtA, 0, sizeof(double)*9*NPIXELS_SKY);
    std::memset(AtD, 0, sizeof(double)*3*NPIXELS_SKY);
    scanMask = (int*)malloc(sizeof(int)*NSAMPLES);
    // no samples are masked
    std::memset(scanMask, 0, sizeof(int)*NSAMPLES);
    mapPixels = (int*)malloc(sizeof(int)*NPIXELS_SKY);
    // all pixels are mapped
    for(int i=0; i < NPIXELS_SKY; i++){ mapPixels[i] = i; }

    psbDataA = (double*)malloc(sizeof(double)*NSAMPLES);
    psbDataB = (double*)malloc(sizeof(double)*NSAMPLES);

}

void free_everything(void) {

    free(skyI);
    free(skyQ);
    free(skyU);
    free(skyV);
    std::cerr << "sky buffers freed" << std::endl;
    free(ra);
    free(dec);
    free(psi);
    std::cerr << "pointing buffers freed" << std::endl;
    free(AtA);
    free(AtD);
    std::cerr << "AtA and AtD buffers freed" << std::endl;
    free(scanMask);
    std::cerr << "scanMask buffers freed" << std::endl;
    free(mapPixels);
    std::cerr << "mapPixels buffers freed" << std::endl;
    free(psbDataA);
    free(psbDataB);
    std::cerr << "data buffers freed" << std::endl;
}

// this function is more robust in the case the wrong file is used
void load_beam_data(char polFlag, std::string path, PolBeam& beam) {
    int i = 0;
    std::ifstream detBeams(path);
    if(!detBeams.is_open()){
        std::cout << "File not found" << std::endl;
        throw std::invalid_argument("cannot open beam data files.");
    }
    std::string line;
    while(std::getline(detBeams, line) && i < NPIXELS_BEAM) {
        std::istringstream iss(line);
        // stop if an error while parsing occurs
        if(polFlag == 'a') {
            if(!(iss
                 >> beam.Ia[i]
                 >> beam.Qa[i]
                 >> beam.Ua[i]
                 >> beam.Va[i]))
            {
                throw std::length_error("not enough data.");
            }
            //beam.Qa[i] = beam.Ia[i];
            //beam.Ua[i] = 0.0;
            //beam.Va[i] = 0.0;
        }
        else if(polFlag == 'b') {
            if(!(iss
                 >> beam.Ib[i]
                 >> beam.Qb[i]
                 >> beam.Ub[i]
                 >> beam.Vb[i]))
            {
                throw std::length_error("not enough data.");
            }
            //beam.Qb[i] = beam.Ib[i];
            //beam.Ub[i] = 0.0;
            //beam.Vb[i] = 0.0;
        }
        else {
            throw std::invalid_argument("fail");
        }
        i++;
    }
    detBeams.close();
}

void load_sky_data(
    std::string path, 
    float* skyI, float* skyQ, float* skyU, float* skyV)
{
    int i = 0;
    std::ifstream skydata(path);
    if(!skydata.is_open()){
        std::cout << "File not found" << std::endl;
        throw std::invalid_argument("cannot open sky data files.");
    }
    std::string line;
    while(std::getline(skydata, line) && i < NPIXELS_SKY) {
        std::istringstream iss(line);
        // stop if an error while parsing occurs
        if(!(iss >> skyI[i] >> skyQ[i] >> skyU[i] >> skyV[i]))
        {
            throw std::length_error("not enough data.");
        }
        i++;
    }
    skydata.close();
}
