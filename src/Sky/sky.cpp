#include "sky.hpp"

#include <cstdlib>
#include <fstream>
#include <string.h>

#include <healpix_base.h>
#include <pointing.h>

Sky::Sky(int _nside) : hpxBase(_nside, RING, SET_NSIDE)
{
    nside = _nside;
    nPixels = 12 * nside * nside;
    skyBufferSize = nPixels * sizeof(float);
} 

Sky::~Sky(void)
{
    free_buffers();
}

int Sky::get_nside (void) const 
{ 
    return nside; 
}

int Sky::get_npixels (void) const 
{ 
    return nPixels; 
}

const float* Sky::get_I(void) const {
    const float* x = I;
    return x;
}

const float* Sky::get_Q(void) const {
    const float* x = Q;
    return x;
}

const float* Sky::get_U(void) const {
    const float* x = U;
    return x;
}

const float* Sky::get_V(void) const {
    const float* x = V;
    return x;
}

void Sky::allocate_buffers(void)
{
    #ifdef SKY_DEBUG
    std::cerr << "Sky::alloc_buffers." << std::endl;
    std::cerr << "  allocating sky data buffer.";
    std::cerr << std::endl;
    #endif
    I = (float*)malloc(skyBufferSize);
    Q = (float*)malloc(skyBufferSize);
    U = (float*)malloc(skyBufferSize);
    V = (float*)malloc(skyBufferSize);
    
    buffersOK = true;
}

void Sky::free_buffers(void)
{
    #ifdef SKY_DEBUG
    std::cerr << "Sky::free_buffers. " << std::endl;
    std::cerr << "  deallocating sky data buffer." << std::endl;
    #endif
    free(I);
    free(Q);
    free(U);
    free(V);
}

void Sky::load_sky_data_from_txt(std::string path)
/**
 * Loads sky's I, Q, U and V data from a text file specified by the path
 * argument. File must contain 4 columns, one for each I, Q, U, V value.
 */
{
    int i;
    std::string line;
    std::ifstream skyDataFile(path);
    
    #ifdef SKY_DEBUG
    std::cerr << "Sky::load_sky_data_from_txt." << std::endl;
    std::cerr << "  reading beam data from file " << path << std::endl;
    #endif
    if(!buffersOK)
    {
        std::cerr << 
            "[ERROR] Buffers have not been allocated.";
        std::cerr << std::endl;
        throw std::runtime_error("Critical error. Aborting.");
    }
    if(!skyDataFile.is_open())
    {
        std::cerr << "[ERROR] Could not open file." << std::endl;
        throw std::invalid_argument("Critical error. Aborting.");
    }
    i = 0;
    while(std::getline(skyDataFile, line) && i < nPixels) 
    {
        std::istringstream iss(line);
        // stop if an error occurs while parsing the file. 
        // also, this is a rather hacky way of reading a file.
        if(!(iss >> I[i] >> Q[i] >> U[i] >> V[i]))
        {
            skyDataFile.close();
            std::cerr << 
                "[ERROR] Not enough data in the file." << std::endl;
            throw std::length_error("Critical error. Aborting.");
        }
        i++;
    }
    skyDataFile.close();
}    

void Sky::make_point_source_sky(float ra0, float dec0, 
    float I0, float Q0, float U0, float V0)
{
    pointing ptgpix;
    int pix;

    // check buffers are allocated
    if(!buffersOK) {
        std::cerr << "[ERROR] Sky object does not have buffers allocated." << std::endl;
        throw std::runtime_error("ERROR.");
    }
    
    // set the map to zero everywhere
    memset(I, 0.0, skyBufferSize);
    memset(Q, 0.0, skyBufferSize);
    memset(U, 0.0, skyBufferSize);
    memset(V, 0.0, skyBufferSize);

    // this is only to ensure correct casting from float to double
    ptgpix.theta = M_PI_2 - double(dec0);
    ptgpix.phi = double(ra0);
    
    // get pixel of point source
    pix = hpxBase.ang2pix(ptgpix);

    // fill in values
    I[pix] = I0;
    Q[pix] = Q0;
    U[pix] = U0;
    V[pix] = V0;
}