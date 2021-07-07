#include "sky.hpp"
#include <cstdlib>
// include healpix related routines
#include <healpix_base.h>

Sky::Sky(int _nside) 
{
    nside = _nside;
    nPixels = 12 * nside * nside;
    skyBufferSize = nPixels * sizeof(float);
}

Sky::~Sky(void)
{
    free_buffers();
}

int get_nside (void) const 
{ 
    return nside; 
}

int get_npixels (void) const 
{ 
    return nPixels; 
}

void Sky::alloc_buffers(void)
{
    #ifdef DEBUG_MESSAGES
    std::cerr << "Sky::alloc_buffers: ";
    std::cerr << "allocating sky data buffer.";
    std::cerr << std::endl;
    #endif
    skyData = (float*)malloc(NPOLSKY * skyBufferSize);
    /* setup GPU buffers if CUDA is enabled. */
    #ifdef USE_CUDA
    /* allocate buffers in the GPU. */
    #ifdef DEBUG_MESSAGES
    std::cerr << "Sky::alloc_buffers: ";
    std::cerr << "allocating sky data buffer in the GPU.";
    std::cerr << std::endl;
    #endif
    CUDA_ERROR_CHECK(
        cudaMalloc((void **)&cuda_skyData, NPOLSKY * skyBufferSize)
    );
    #endif
}

void Sky::free_buffers(void)
{
    #ifdef DEBUG_MESSAGES
    std::cerr << "Sky::free_buffers: ";
    std::cerr << "deallocating sky data buffer.";
    std::cerr << std::endl;
    #endif
    free(skyData);
    /* free GPU buffers if CUDA is enabled. */
    #ifdef USE_CUDA
    /* deallocate buffers in the GPU. */
    #ifdef DEBUG_MESSAGES
    std::cerr << "Sky::free_buffers: ";
    std::cerr << "deallocating sky data buffer in the GPU.";
    std::cerr << std::endl;
    #endif
    CUDA_ERROR_CHECK(cudaFree(cuda_skyData));
    #endif
}

void
Sky::load_sky_data_from_txt(std::string path)
/**
 * Loads sky's I, Q, U and V data from a text file specified by the path
 * argument. File must contain 4 columns, one for each I, Q, U, V value.
 */
{
    int i;
    std::string line;
    std::ifstream skyDataFile(path);
    
    #ifdef DEBUG_MESSAGES
    std::cerr << "Sky::load_sky_data_from_txt: ";
    std::cerr << "reading beam data from file " << path << std::endl;
    #endif
    if(!buffersOK)
    {
        std::cerr << 
            "[ERROR] Buffers have not been successfully allocated.";
        std::cerr << std::endl;
        throw std::runtime_error("Critical error. Aborting.");
    }
    if(!skyDataFile.is_open())
    {
        std::cerr << "[ERROR] File not found." << std::endl;
        throw std::invalid_argument("Critical error. Aborting.");
    }
    i = 0;
    while(std::getline(skyDataFile, line) && i < nPixels) 
    {
        std::istringstream iss(line);
        // stop if an error occurs while parsing the file. 
        // also, this is a rather hacky way of reading a file.
        if(!(iss 
            >> skyData[NPOLSKY*i + 0] // I
            >> skyData[NPOLSKY*i + 1] // Q
            >> skyData[NPOLSKY*i + 2] // U
            >> skyData[NPOLSKY*i + 3] /* V*/))
        {
            skyDataFile.close();
            std:cerr << 
                "[ERROR] Not enough data in the file." << std::endl;
            throw std::length_error("Critical error. Aborting.");
        }
        i++;
    }
    skyDataFile.close();
}    

const float* get_buffer() const
{
    #ifdef CUDA
    const float* b = cuda_skyData;
    #else
    const float* b = skyData;
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
    std::cerr << "Sky::transfer_to_gpu: ";
    std::cerr << "copying sky data to GPU device.";
    std::cerr << std::endl;
    #endif
    CUDA_ERROR_CHECK
    (
        cudaMemcpy(
            cuda_skyData, skyData,
            NPOLSKY * skyBufferSize,
            cudaMemcpyHostToDevice)
    );
}
#endif


