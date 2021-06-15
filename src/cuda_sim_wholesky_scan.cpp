/**/
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <unistd.h>

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Bpoint/BPoint.h"
#include "Polbeam/polbeam.hpp"
#include "GpuConvolver/gpuconvolver.hpp"
#include "SkyMap/skymap.hpp"

// needed to create the point source map
#include <healpix_base.h>
#include <pointing.h>

#define NSIDE_SKY (128)
#define NSIDE_SKY_OUT (128)
#define NPIXELS_SKY (12 * (NSIDE_SKY) * (NSIDE_SKY))
#define NPIXELS_SKY_OUT (12 * (NSIDE_SKY_OUT) * (NSIDE_SKY_OUT))
#define NSIDE_BEAM 512
/** how to calculate the number of pixels below
import healpy
import numpy
nside = 512
print(healpy.query_disc(nside, (0,0,1), numpy.radians(5)).size)
#5940
**/
#define NPIXELS_BEAM (5940)
/** number of time samples = number of pixels because this is a 
 *  whole sky simulation.
 **/
#define NSAMPLES (NPIXELS_SKY)
/** pointers to buffers storing the sky **/
float* skyI;
float* skyQ;
float* skyU;
float* skyV;
/** pointers to buffers storing output maps. **/
const int* hits;
const float* mapI;
const float* mapQ;
const float* mapU;
/** buffers so store PSB pointing **/
float* ra;
float* dec;
float* psi;
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
    int nside, float pa, float *ra, float *dec, float *psi);
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
    auto start = std::chrono::steady_clock::now();
    auto stop = std::chrono::steady_clock::now();
    
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
    // initialize SkyMap object 
    SkyMap smap(NSIDE_SKY_OUT);
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
    cfg.nStreams = 2;
    cfg.maxMemUsage = size_t(11 * 1e9);
    cfg.deviceId = 0;
    cfg.gridSizeX = 512;
    cfg.gridSizeY = 1;
    cfg.blockSizeX = 64;
    cfg.blockSizeY = 1;
    cfg.ptgPerConv = 8192;
    cfg.pixelsPerDisc = 6200;
    GPUConvolver cconv(cfg);
    cconv.update_sky(&sky);
    cconv.update_beam(&beam);
    // setup detetctor angle arays
    float detectorAngle[1];
    float positionAngles[] = {-45, 0, 45};
    start = std::chrono::steady_clock::now();
    // every PSB scans the sky at different angles
    for(float bcpa_deg: positionAngles)
    {
        // angles are in degrees. convert.
        float bcpa = M_PI * (bcpa_deg / 180.0);
        init_wholesky_scan(NSIDE_SKY, bcpa, ra, dec, psi);
        // compute convolution for detector A and B of PSB
        cconv.exec_convolution(
            psbDataA, psbDataB, 
            &scan, &sky, &beam);
        // project detector data to map-making matrices
        if(projDets == 'a' || projDets == 'p') 
        {
            // project detector a
            detectorAngle[0] = 0;
            smap.accumulate_data
            (   NSAMPLES, ra, dec, psi, 
                1, detectorAngle,
                psbDataA
            );
        }
        if(projDets == 'b' || projDets == 'p') 
        {
            // project detector b
            detectorAngle[0] = M_PI_2;
            smap.accumulate_data
            (   NSAMPLES, ra, dec, psi, 
                1, detectorAngle,
                psbDataB
            );
        }
    }
    smap.solve_map();
    stop = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cerr << "time taken to convolve beam and sky: " << elapsed << "ms" << std::endl;
    outdata.close();
    free_everything();
    /*
    hits = smap.get_hitmap();
    mapI = smap.get_stokes_I();
    mapQ = smap.get_stokes_Q();
    mapU = smap.get_stokes_U();
    for(int i=0; i < NPIXELS_SKY_OUT; i++)
    {
        outdata << mapI[i] << " " <<
                   mapQ[i] << " " <<
                   mapU[i] << " " <<
                   hits[i] << std::endl;
    }

    outdata.close();
    free_everything();
    */
    return 0;
}

void init_wholesky_scan(
    int nside, 
    float pa,
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
