#include <complex>
#include <cstdlib>
#include <cstring>
// to make use of Healpix_base pointing
// include healpix related routines
#include <healpix_base.h>
#include <pointing.h>

#include "polbeam.hpp"
#ifdef USA_CUDA
#include <cuda.h>
#include "Convolution/cuda/cuda_error_check.h"
#endif

/* deprecated code
#ifdef POLBEAM_DUMPBEAMS
#include <fstream>
#endif
*/

PolBeam::PolBeam
(
    int _nside, long _nPixels, 
    double _epsilon, char _enabledDets 
)
/**
 * PolBeam constructor.
 * 
 * \param _nside: NSIDE parameter of internal Healpix maps.
 * \param _nPixels: number of pixels in every map.
 * \param _epsilon: 1 - polarization efficiency of detectors.
 * \param _enabledDets: set to 'a' or 'b' to use a single detector. Use
 * 'p' to use both detectors in the PSB.
 */
{
    nside = _nside;
    nPixels = _nPixels;
    epsilon = _epsilon;
    enabledDets = _enabledDets;
    buffersOK = false;
    
    Healpix_Base hpxBase(_nside, RING, SET_NSIDE)
    pointing p = hpxBase.pix2ang(nPixels);
    rhoMax = p.theta;
    
    beamBufferSize = sizeof(float) * nPixels;
    alloc_buffers();
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::constructor: ";
    std::cerr << "detectors enabled = " << enabledDets << std::endl;
    std::cerr << "beam extension = " << rhoMax * (180.0 / M_PI) << " ";
    std::cerr << "degrees." << std::endl;
    std::cerr << "number of pixels = " << nPixels << std::endl;
    #endif 
}

PolBeam::~PolBeam()
/** Destructor.
 * 
 * \note: frees all buffers before destruction.
 */
{
    free_buffers();
}

int PolBeam::get_nside(void) const
{
    return nside;
};

int PolBeam::get_npixels(void) const
{
    return nPixels;
};

int PolBeam::get_enabled_detectors(void) const
{
    return enabledDets;
}

double PolBeam::get_rho_max() const
{
    return rhoMax;
};


void PolBeam::alloc_buffers()
/** 
 * Allocate buffers to store beams.
 *
 */
{
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::alloc_buffers: ";
    std::cerr << "buffer size is " << beamBufferSize << " bytes (";
    std::cerr << nPixels << " pixels)" << std::endl;
    std::cerr << "PolBeam::alloc_buffers: ";
    std::cerr << "allocating beam buffers for detector a." << std::endl;
    #endif
    /* allocate buffers for detector a. */
    aBeams = (float*)malloc(NPOLBEAMS * beamBufferSize);
    Ia = (float*)malloc(beamBufferSize);
    Qa = (float*)malloc(beamBufferSize);
    Ua = (float*)malloc(beamBufferSize);
    Va = (float*)malloc(beamBufferSize); 
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::alloc_buffers: ";
    std::cerr << "allocating beam buffers for detector b." << std::endl;
    #endif
    /* allocate buffers for detector b. */
    bBeams = (float*)malloc(NPOLBEAMS * beamBufferSize);
    Ib = (float*)malloc(beamBufferSize);
    Qb = (float*)malloc(beamBufferSize);
    Ub = (float*)malloc(beamBufferSize);
    Vb = (float*)malloc(beamBufferSize);
    /* set beams to zero by default. */
    std::memset(aBeams, 0, NPOLBEAMS * buffSize);
    std::memset(bBeams, 0, NPOLBEAMS * buffSize);
    std::memset(Ia, 0, buffSize);
    std::memset(Ib, 0, buffSize);
    std::memset(Qa, 0, buffSize);
    std::memset(Qb, 0, buffSize);
    std::memset(Ua, 0, buffSize);
    std::memset(Ub, 0, buffSize);
    std::memset(Va, 0, buffSize);
    std::memset(Vb, 0, buffSize);
    
    /* setup GPU buffers if CUDA is enabled. */
    #ifdef USE_CUDA
    /* allocate buffers in the GPU. */
    std::cerr << "PolBeam::alloc_buffers: ";
    std::cerr << "allocating beam buffers in the GPU.";
    std::cerr << std::endl;
    CUDA_ERROR_CHECK(
        cudaMalloc((void **)&cuda_aBeams, NPOLBEAMS * beamBufferSize)
    );
    CUDA_ERROR_CHECK(
        cudaMalloc((void **)&cuda_bBeams, NPOLBEAMS * beamBufferSize)
    );
    #endif
    buffersOK = true;
}

void PolBeam::free_buffers()
/**
 * Routine to free all buffers before deleting the object.
 */
{
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::free_buffers: ";
    std::cerr << "freeing beam buffers for detector a." << std::endl;
    #endif
    free(Ia);
    free(Qa);
    free(Ua);
    free(Va);
    free(aBeams);
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::free_buffers: ";
    std::cerr << "freeing beam buffers for detector b." << std::endl;
    #endif
    free(Ib);
    free(Qb);
    free(Ub);
    free(Vb);
    free(bBeams);
    #ifdef USE_CUDA
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::free_buffers: ";
    std::cerr << "freeing beam gpu buffers for detector a." << std::endl;
    #endif // end DEBUG_MESSAGES
    CUDA_ERROR_CHECK(cudaFree(cuda_aBeams));
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::free_buffers: ";
    std::cerr << "freeing beam gpu buffers for detector b." << std::endl;
    #endif // end DEBUG_MESSAGES
    CUDA_ERROR_CHECK(cudaFree(cuda_bBeams));
    #endif // end USE_CUDA
}

void
PolBeam::load_beam_data_from_txt(std::string path)
/**
 * Loads I, Q, U and V beam data from a text file specified by the path
 * argument. File must contain 8 columns: 4 for each of the I, Q, U, V
 * beams of each detector.
 */
{
    int i;
    std::string line;
    std::ifstream beamDataFile(path);
    
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::load_beam_data_from_txt: ";
    std::cerr << "reading beam data from file " << path << std::endl;
    #endif
    if(!buffersOK)
    {
        std::cerr << 
            "[ERROR] Buffers have not been successfully allocated.";
        std::cerr << std::endl;
        throw std::runtime_error("Critical error. Aborting.");
    }
    if(!beamDataFile.is_open())
    {
        std::cerr << "[ERROR] File not found." << std::endl;
        throw std::invalid_argument("Critical error. Aborting.");
    }
    i = 0;
    while(std::getline(beamDataFile, line) && i < nPixels) 
    {
        std::istringstream iss(line);
        // stop if an error occurs while parsing the file. 
        // also, this is a rather hacky way of reading a file.
        if(!(iss 
            >> Ia[i] >> Qa[i] >> Ua[i] >> Va[i]
            >> Ib[i] >> Qb[i] >> Ub[i] >> Vb[i]))
        {
            beamDataFile.close();
            std:cerr << 
                "[ERROR] Not enough data in the file." << std::endl;
            throw std::length_error("Critical error. Aborting.");
        }
        i++;
    }
    beamDataFile.close();
}    

void
PolBeam::beam_from_fields
(
    char polFlag,
    double* magEco, double* phaseEco,
    double* magEcx, double* phaseEcx
)
/**
 * Computes the polarized beam of a single PSB detector (a or b) using 
 * the electric field power density (EFPD) of an antenna. This routine
 * is based on the formalism found in Rosset el at. 2010, with heavy
 * contributions from Michael K. Brewer.
 *
 * The field must be specified as 4 arrays containing the magnitude and
 * phase of the EFPD along the co and cross polar directions of the
 * antenna basis.
 *
 * \param polFlag: set to 'a' ('b') on the particular detector beam of
 *        the PSB that is being loaded.
 * \param magEco: array with co-polar EFPD magnitude.
 * \param phaseEco: array with the phase of co-polar EFPD.
 * \param magEcx: array with cross-polar EFPD magnitude along.
 * \param phaseEcx: array with the phase of cross-polar EFPD.
 *
 */
{
    float *I;
    float *Q;
    float *U;
    float *V;
    std::complex<double> Eco;
    std::complex<double> Ecx;
    if(polFlag == 'a')
    {
        #ifdef DEBUG_MESSAGES
        std::cerr << "PolBeam::beam_from_fields: ";
        std::cerr << "building (I,Q,U,V) beams from fields for detector a.";
        std::cerr << std::endl;
        #endif
        I = Ia;
        Q = Qa;
        U = Ua;
        V = Va;
    }
    if(polFlag == 'b')
    {
        #ifdef DEBUG_MESSAGES
        std::cerr << "PolBeam::beam_from_fields: ";
        std::cerr << "building (I,Q,U,V) beams from fields for detector b.";
        std::cerr << std::endl;
        #endif
        I = Ib;
        Q = Qb;
        U = Ub;
        V = Vb;
    }
    for(int i = 0; i < nPixels; i++)
    {
        // component along x basis vector
        Eco = std::polar(magEco[i], phaseEco[i]);
        // component along y basis vector
        Ecx = std::polar(magEcx[i], phaseEcx[i]);
        // compute \tilde{I}
        I[i] = float(std::norm(Eco) + std::norm(Ecx));
        // compute \tilde{Q}
        Q[i] = float(std::norm(Eco) - std::norm(Ecx));
        // compute \tilde{U}
        U[i] = float(2*std::real(Eco*std::conj(Ecx)));
        // compute \tilde{V}
        V[i] = float(-2*std::imag(Eco*std::conj(Ecx)));
    }
}

void PolBeam::build(int nsideSky)
/**
 * Build the actual polarized beams from tilde I, Q, U and V. These 
 * beams take into account possible cross-talk produced by finite 
 * polarization efficiency of detectors.
 * 
 * The NSIDE parameter of the sky is needed to calculate a normalization 
 * factor such that the time ordered data generated by the convolution
 * process keeps being in brightness temperature units (Kelvin).
 * 
 * \param nsideSky: NSIDE parameter of the sky map that will be used
 * in the convolution process.
 *
 */
{
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::build: ";
    std::cerr << "building polarized beams." << std::endl;
    #endif
    // compute solid angle
    double sumIa = 0;
    double sumIb = 0;
    for(int i = 0; i < nPixels; i++) 
    {
        if(enabledDets == 'a' || enabledDets == 'p')
        {
            sumIa += Ia[i];
        }
        if(enabledDets == 'b' || enabledDets == 'p')
        {        
            sumIb += Ib[i];
        }
    }   
    // compute normalization factor
    double ocmp = nsideSky / double(nside);
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::build: ";
    std::cerr << "solid angle of detector a = ";
    std::cerr << sumIa * ocmp << " strad" << std::endl;
    std::cerr << "PolBeam::build: ";
    std::cerr << "solid angle of detector b = "; 
    std::cerr << sumIb * ocmp << " strad" << std::endl;
    #endif
    ocmp = ocmp * ocmp;
    double aNorm = sumIa * ocmp;
    double bNorm = sumIb * ocmp;
    // hack to avoid the beam from blowing up
    if(enabledDets == 'a')
    {
        bNorm += 1e-8;
    }
    if(enabledDets == 'b')
    {
        aNorm += 1e-8;
    }
    /* deprecated code.
    #ifdef POLBEAM_DUMPBEAMS
    std::string dumpfilepatha = "dump_detector_a.txt";
    std::string dumpfilepathb = "dump_detector_b.txt";
    std::ofstream dumpfilea;
    std::ofstream dumpfileb;
    if(enabledDets == 'a' || enabledDets == 'p')
    {
        dumpfilea.open(dumpfilepatha);
    }
    if(enabledDets == 'b' || enabledDets == 'p')
    {        
        dumpfileb.open(dumpfilepathb);
    }
    std::cerr << "PolBeam::build: ";
    std::cerr << "dumping beams to ./dump_detector_a.txt";
    std::cerr << "and ./dump_detector_b.txt";
    std::cerr << std::endl;
    #endif
    */
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::build: ";
    std::cerr << "building polarized beams.";
    std::cerr << std::endl;
    #endif
    for(int i = 0; i < nPixels; i++)
    {
        aBeams[NPOLBEAMS * i + 0] = (Ia[i] + epsilon * Ib[i]) / aNorm;
        aBeams[NPOLBEAMS * i + 1] = (Qa[i] - epsilon * Qb[i]) / aNorm;
        aBeams[NPOLBEAMS * i + 2] = (Ua[i] - epsilon * Ub[i]) / aNorm;
        aBeams[NPOLBEAMS * i + 3] = (Va[i] + epsilon * Vb[i]) / aNorm;
        
        bBeams[NPOLBEAMS * i + 0] = (Ib[i] + epsilon * Ia[i]) / bNorm;
        bBeams[NPOLBEAMS * i + 1] = (Qb[i] - epsilon * Qa[i]) / bNorm;
        bBeams[NPOLBEAMS * i + 2] = (Ub[i] - epsilon * Ua[i]) / bNorm;
        bBeams[NPOLBEAMS * i + 3] = (Vb[i] + epsilon * Va[i]) / bNorm;
        /* deprecated code. 
        #ifdef POLBEAM_DUMPBEAMS
        dumpfilea
        << aBeams[0][i] << " "
        << aBeams[1][i] << " "
        << aBeams[2][i] << " "
        << aBeams[3][i] << std::endl;
        dumpfileb
        << bBeams[0][i] << " "
        << bBeams[1][i] << " "
        << bBeams[2][i] << " "
        << bBeams[3][i] << std::endl;
        #endif
        */
    }
    /* deprecated code
    #ifdef POLBEAM_DUMPBEAMS
    dumpfilea.close();
    dumpfileb.close();
    #endif
    */
}

const float* get_beam_a() const;
{
    #ifdef CUDA
    const float* b = cuda_aBeams;
    #else
    const float* b = aBeams;
    #endif
    return b;
}
const float* get_beam_b() const
{
    #ifdef CUDA
    const float* b = cuda_bBeams;
    #else
    const float* b = bBeams;
    #endif
    return b;
}

#ifdef USE_CUDA
void set_gpu_device(int deviceId)
{
    cudaSetDevice(devideId);
}

void transfer_to_gpu(void)
{
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::build: ";
    std::cerr << "copying beam of detector a to GPU device.";
    std::cerr << std::endl;
    #endif
    CUDA_ERROR_CHECK
    (
        cudaMemcpy(
            cuda_aBeams, aBeams,
            NPOLBEAMS * beamBufferSize,
            cudaMemcpyHostToDevice)
    );
    #ifdef DEBUG_MESSAGES
    std::cerr << "PolBeam::build: ";
    std::cerr << "copying beam of detector b to GPU device.";
    std::cerr << std::endl;
    #endif
    CUDA_ERROR_CHECK
    (
        cudaMemcpy(
            cuda_bBeams, bBeams,
            NPOLBEAMS * beamBufferSize,
            cudaMemcpyHostToDevice)
    );    
}
#endif


void PolBeam::make_unpol_gaussian_elliptical_beams
(
    double fwhmx,
    double fwhmy,
    double phi0
)
/**
 *  Creates a Gaussian Elliptical Beam as a HEALPix grid.
 *
 *  Authors: Michael K. Brewer (CLASS collaboration, 2018)
 *           C++ adaptation by Pedro Fluxa (PUC, 2019)
 *  input:
 *
 *      fwhm_x  : Full Width at Half Maximum, in the x (South-North)
 *                direction if phi_0=0 (degrees)
 *      fwhm_y  : Full Width at Half Maximum, in the y (East-West)
 *                direction if phi=0, (degrees)
 *      phi_0   : Angle between Noth-South direction and x-axis of the
 *                ellipse (degrees). phi_0 increases clockwise towards
 *                East.
 *
 *  In this routine, the same elliptical gaussian is assigned to
 *  I,Q, U and V beams. All beams are normalized so that
 *
 *  \int_{4\pi} b(\rho,\sigma) d\Omega = 1.0
 *
 */
{
    // temporal buffers to store fields
    size_t buffSize = sizeof(double)*nPixels;
    double* magEco = (double*)malloc(buffSize);
    double* magEcx = (double*)malloc(buffSize);
    double* phsEco = (double*)malloc(buffSize);
    double* phsEcx = (double*)malloc(buffSize);
    // Convert FWHM in degres to sigma, in radians
    double deg2rad = M_PI/180.0;
    // From https://en.wikipedia.org/wiki/Full_width_at_half_maximum:
    //
    // FWHM = 2 \sqrt{2 \ln{2}} \sigma
    //
    // where \sigma is the standard deviation and 
    // 2 \sqrt{2 \ln{2}} ~ 2.35482
    double sigma_x = (deg2rad*fwhmx)/2.35482;
    double sigma_y = (deg2rad*fwhmy)/2.35482;
    //Convert phi_0 to radians
    double phi_0 = deg2rad * phi0;
    // Compute coefficients to rotate the ellipse.
    double a =  (cos(phi_0)*cos(phi_0))/(2*sigma_x*sigma_x) +
                (sin(phi_0)*sin(phi_0))/(2*sigma_y*sigma_y);
    double b = -(sin(2*phi_0) )/(4*sigma_x*sigma_x) +
                (sin(2*phi_0 ))/(4*sigma_y*sigma_y);
    double c =  (sin(phi_0)*sin(phi_0))/(2*sigma_x*sigma_x) +
                (cos(phi_0)*cos(phi_0))/(2*sigma_y*sigma_y);
    double rho, sig, val;
    for(int bpix = 0; bpix < nPixels; bpix++)
    {
        pointing bp = hpxBase.pix2ang(bpix);
        rho = bp.theta;
        sig = bp.phi;
        val = exp(-(a*cos(sig)*cos(sig)
                  + 2.0*b*cos(sig)*sin(sig)
                  + c*sin(sig)*sin(sig))*rho*rho);
        // build fields for a perfectly co-polarized beams
        magEco[bpix] = sqrt(val);
        magEcx[bpix] = 0.0;
        phsEco[bpix] = 0.0;
        phsEcx[bpix] = 0.0;
    }
    // build beams from dummy fields
    if(enabledDets == 'a' || enabledDets == 'p')
    {
        beam_from_fields('a', magEco, phsEco, magEcx, phsEcx);
    }
    if(enabledDets == 'b' || enabledDets == 'p')
    {
        beam_from_fields('b', magEco, phsEco, magEcx, phsEcx);
    }
    // free temporal storage
    free(magEco);
    free(phsEco);
    free(magEcx);
    free(phsEcx);
}
