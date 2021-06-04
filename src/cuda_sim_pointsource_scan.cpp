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

#define NSIDE_SKY (512)
#define NPIXELS_SKY (12*(NSIDE_SKY)*(NSIDE_SKY))
#define NSAMPLES_GRID 384
#define NSAMPLES (NSAMPLES_GRID*NSAMPLES_GRID)
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
void init_point_source_scan(
    double ra0, double dec0, double psi0,
    double deltaRA, double deltaDEC,
    int nRA, int nDEC,
    // output
    float *ra, float *dec, float *psi);
/**routine to initialize sky to have a single non-zero pixel with
 * polarization defined by pwrI,Q,U.
 **/
void init_point_source_sky(
    double ra0, double dec0,
    double pwrI, double pwrQ, double pwrU,
    // output
    float *I, float *Q, float *U, float *V);
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
    // polarization of input source
    float pI = 1.0;
    float pQ = 0.0;
    float pU = 0.0;
    bool beamsFromGauss = true;
    double bfwhmx = 0.0;
    double bfwhmy = 0.0;
    std::string beamAPath;
    std::string beamBPath;
    int opt;
    while((opt = getopt(argc, argv, "t:s:o:p:a:b:")) != -1 ) {
        switch(opt)
        {
            case 's':
                if(char(optarg[0]) == 'Q'){pQ = 1.0;}
                if(char(optarg[0]) == 'U'){pU = 1.0;}
                continue;
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
        }
    }
    // opens output file
    std::ofstream outdata;
    outdata.open(outfilePath);
    std::cout << pI << " " << pQ << " " << pU << std::endl;
    std::cout << projDets << std::endl;
    std::cout << beamAPath << std::endl;
    std::cout << beamBPath << std::endl;
    std::cout << outfilePath << std::endl;
    allocate_everything();
    // initialize sky object
    Sky sky(NSIDE_SKY, skyI, skyQ, skyU, skyV);
    init_point_source_sky(M_PI, 0.0, pI, pQ, pU, skyI, skyQ, skyU, skyV);
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
    cfg.ptgPerConv = 1024;
    cfg.pixelsPerDisc = 10000;
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
        init_point_source_scan(
            M_PI, 0.0, bcpa,
            0.2, 0.2,
            NSAMPLES_GRID, NSAMPLES_GRID,
            ra, dec, psi);
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

void init_point_source_scan(
    double ra0, double dec0, double psiScan,
    double deltaRa, double deltaDec,
    int gridSizeRa, int gridSizeDec,
    float* ra, float* dec, float* psi) {

    double ra_bc;
    double dec_bc;
    double ndec = double(gridSizeDec);
    double nra = double(gridSizeRa);
    for(int i=0; i < gridSizeDec; i++) {
        dec_bc = dec0 - deltaDec + 2.0*deltaDec*(double(i)/ndec);
        for(int j=0; j < gridSizeRa; j++) {
            ra_bc = ra0 - deltaRa + 2.0*deltaRa*(double(j)/nra);
            ra[i*gridSizeDec + j] = ra_bc;
            dec[i*gridSizeDec + j] = dec_bc;
            psi[i*gridSizeDec + j] = psiScan;
        }
    }
}

void init_point_source_sky(
    double ra0, double dec0,
    double pwrI, double pwrQ, double pwrU,
    // output
    float *I, float *Q, float *U, float* V) {

    int sourcePixel;
    Healpix_Base hpx;
    hpx.SetNside(NSIDE_SKY, RING);

    std::memset(I, 0, sizeof(float)*NPIXELS_SKY);
    std::memset(Q, 0, sizeof(float)*NPIXELS_SKY);
    std::memset(U, 0, sizeof(float)*NPIXELS_SKY);
    std::memset(V, 0, sizeof(float)*NPIXELS_SKY);

    pointing psptg(M_PI_2 - dec0, ra0);
    sourcePixel = hpx.ang2pix(psptg);
    I[sourcePixel] = pwrI;
    Q[sourcePixel] = pwrQ;
    U[sourcePixel] = pwrU;
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
