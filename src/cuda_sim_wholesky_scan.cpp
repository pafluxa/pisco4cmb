/**/
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <unistd.h>

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Bpoint/BPoint.h"
#include "Polbeam/polbeam.hpp"
#include "GpuConvolver/gpuconvolver.hpp"
#include "Mapping/mapping_routines.h"

// needed to create the point source map
#include <healpix_base.h>
#include <pointing.h>

#define NSIDE_SKY (128)
#define NPIXELS_SKY (12*(NSIDE_SKY)*(NSIDE_SKY))
#define NSAMPLES (NPIXELS_SKY)
/** how to calculate the number of pixels below
import healpy
import numpy
nside = 2048
print(healpy.query_disc(nside, (0,0,1), numpy.radians(5)).size)
#95484
**/
#define NSIDE_BEAM 2048
#define NPIXELS_BEAM (95484)

/** buffers to store the sky **/
float* skyI;
float* skyQ;
float* skyU;
float* skyV;
/** buffers so store PSB pointing **/
float* ra;
float* dec;
float* psi;
/** map-making accumulation matrix (vector) **/
double* AtA;
double* AtD;
/** buffers to simulated data **/
// detector A
float* psbDataA;
// detector B
float* psbDataB;
/**
 * routine to initialize the scan to a raster scan around a location
 * defined by ra0, dec0. the scan is executed at position angle psi0.
 **/
void init_wholesky_scan(
    int nside, double pa, float *ra, float *dec, float *psi);
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
    char projDets;
    bool beamsFromGauss = true;
    double bfwhmx = 0.0;
    double bfwhmy = 0.0;
    std::string inputSkyPath;
    std::string beamAPath;
    std::string beamBPath;
    int opt;
    while((opt = getopt(argc, argv, "t:m:o:p:a:b:")) != -1 ) 
    {
        switch(opt)
        {
            case 'p':
                projDets = char(optarg[0]);
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
    std::cout << projDets << std::endl;
    std::cout << beamAPath << std::endl;
    std::cout << beamBPath << std::endl;
    std::cout << outfilePath << std::endl;
    allocate_everything();
    // initialize sky object
    load_sky_data(inputSkyPath, skyI, skyQ, skyU, skyV);
    Sky sky(NSIDE_SKY, skyI, skyQ, skyU, skyV);
    // initialize polarized beam object. read beam data from disk
    PolBeam beam(NSIDE_BEAM, NPIXELS_BEAM, 0.0, projDets);
    if(beamsFromGauss) 
    {
        bfwhmx = std::stod(beamAPath.c_str());
        bfwhmy = std::stod(beamBPath.c_str());
        beam.make_unpol_gaussian_elliptical_beams(bfwhmx, bfwhmy, 0.0);
    }
    else if (projDets == 'a')// || projDets == 'p') 
    {
        load_beam_data('a', beamAPath, beam);
    }
    else if (projDets == 'b')// || projDets == 'p') 
    {
        load_beam_data('b', beamBPath, beam);
        std::cout << "POLBEAM OK" << std::endl;
    }
    else if (projDets == 'p')// || projDets == 'p') 
    {
        load_beam_data('a', beamBPath, beam);
        load_beam_data('b', beamBPath, beam);
        std::cout << "POLBEAM OK" << std::endl;
    }
    beam.build(NSIDE_SKY);
    std::cout << "POLBEAM OK" << std::endl;
    // initialize scan
    Scan scan(NSAMPLES, ra, dec, psi);
    // initialize convolver object
    CUDACONV::RunConfig cfg;
    cfg.gridSizeX = 512;
    cfg.gridSizeY = 1;
    cfg.blockSizeX = 32;
    cfg.blockSizeY = 1;
    cfg.ptgPerConv = 8192 * 8;
    cfg.pixelsPerDisc = 2000;
    GPUConvolver cconv(scan, cfg);
    cconv.update_sky(&sky);
    cconv.update_beam(&beam);
    // setup detetctor angle arays
    double detectorAngle[1];
    double positionAngles[] = {-45, 0, 45};
    // every PSB scans the sky at different angles
    for(double bcpa_deg: positionAngles)
    {
        // angles are in degrees. convert.
        double bcpa = M_PI * (bcpa_deg / 180.0);
        init_wholesky_scan(NSIDE_SKY, bcpa, ra, dec, psi);
        // compute convolution for detector A and B of PSB
        cconv.exec_convolution(
            cfg, 
            psbDataA, psbDataB, 
            &scan, &sky, &beam);
        // project detector data to map-making matrices
        if(projDets == 'a' || projDets == 'p') 
        {
            // project detector a
            detectorAngle[0] = 0;
            libmapping_project_data_to_matrices
            (
                NSAMPLES, 
                ra, dec, psi,
                1, detectorAngle,
                psbDataA,
                NSIDE_SKY,
                AtA, AtD
            );
        }
        if(projDets == 'b' || projDets == 'p') 
        {
            // project detector b
            detectorAngle[0] = M_PI_2;
            libmapping_project_data_to_matrices
            (
                NSAMPLES,
                ra, dec, psi,
                1, detectorAngle,
                psbDataB,
                NSIDE_SKY,
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
        NSIDE_SKY,
        AtA, AtD,
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
    float* ra, float* dec, float* psi) 
{
    
    int i;
    Healpix_Base hpx(nside, RING, SET_NSIDE);
    
    for(i = 0; i < NSAMPLES; i++)
    {
        pointing ptg = hpx.pix2ang(i);
        ra[i] = ptg.phi;
        dec[i] = M_PI_2 - ptg.theta;
        psi[i] = pa;
    }
}

void allocate_everything(void) 
{
    skyI = (float*)malloc(NPIXELS_SKY*sizeof(float));
    skyQ = (float*)malloc(NPIXELS_SKY*sizeof(float));
    skyU = (float*)malloc(NPIXELS_SKY*sizeof(float));
    skyV = (float*)malloc(NPIXELS_SKY*sizeof(float));
    
    ra = (float*)malloc(sizeof(float)*NSAMPLES);
    dec = (float*)malloc(sizeof(float)*NSAMPLES);
    psi = (float*)malloc(sizeof(float)*NSAMPLES);
    
    psbDataA = (float*)malloc(sizeof(float)*NSAMPLES);
    psbDataB = (float*)malloc(sizeof(float)*NSAMPLES);
    
    // zero out map-making accumulation matrices
    AtA = (double*)malloc(sizeof(double)*9*NPIXELS_SKY);
    AtD = (double*)malloc(sizeof(double)*3*NPIXELS_SKY);
    std::memset(AtA, 0, sizeof(double)*9*NPIXELS_SKY);
    std::memset(AtD, 0, sizeof(double)*3*NPIXELS_SKY);
}

void free_everything(void) 
{
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
    free(psbDataA);
    free(psbDataB);
    std::cerr << "data buffers freed" << std::endl;
}

// this function is more robust in the case the wrong file is used
void load_beam_data(char polFlag, std::string path, PolBeam& beam) 
{
    int i = 0;
    std::ifstream detBeams(path);
    if(!detBeams.is_open())
    {
        std::cout << "File not found" << std::endl;
        throw std::invalid_argument("cannot open beam data files.");
    }
    std::string line;
    while(std::getline(detBeams, line) && i < NPIXELS_BEAM) 
    {
        std::istringstream iss(line);
        // stop if an error occurs while parsing the file
        if(polFlag == 'a') 
        {
            if(!(iss
                 >> beam.Ia[i]
                 >> beam.Qa[i]
                 >> beam.Ua[i]
                 >> beam.Va[i]))
            {
                throw std::length_error("not enough data.");
            }
        }
        else if(polFlag == 'b') 
        {
            if(!(iss
                 >> beam.Ib[i]
                 >> beam.Qb[i]
                 >> beam.Ub[i]
                 >> beam.Vb[i]))
            {
                throw std::length_error("not enough data.");
            }
        }
        else 
        {
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
    if(!skydata.is_open())
    {
        std::cout << "File not found" << std::endl;
        throw std::invalid_argument("cannot open sky data files.");
    }
    std::string line;
    while(std::getline(skydata, line) && i < NPIXELS_SKY) 
    {
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
