#include <cuda.h>
#include <omp.h>
#include <cstring>
#include <cuda_runtime.h>

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Polbeam/polbeam.hpp"
#include "GpuConvolver/gpuconvolver.hpp"
#include "GpuConvolver/cuda/cuda_error_check.h"
#include "GpuConvolver/cuda/beam_times_sky.h"

#define N_POINTING_COORDS 4

GPUConvolver::GPUConvolver(int _nsamples, CUDACONV::RunConfig cfg)
{
    nsamples = _nsamples;
    hasSky = false;
    hasBeam = false;
    
    chunkSize = cfg.chunkSize;
   
    resultBufferSize = 2*chunkSize*sizeof(float);
    nPixelInDiscBufferSize = chunkSize*sizeof(int);
    ptgBufferSize = N_POINTING_COORDS*chunkSize*sizeof(double);
    skyPixelsInBeamBufferSize = \
        chunkSize*cfg.MAX_PIXELS_PER_DISC*sizeof(int);
    
    // allocate buffers to store pointing
    ptgBuffer = (double*)malloc(ptgBufferSize);
    CUDA_ERROR_CHECK(
        cudaMalloc((void **)&ptgBufferGPU, 
            ptgBufferSize)
    );
    // to store sky pixels
    skyPixelsInBeam = (int*)malloc(skyPixelsInBeamBufferSize);
    CUDA_ERROR_CHECK(
        cudaMalloc(
            (void **)&skyPixelsInBeamGPU, 
            skyPixelsInBeamBufferSize)
    );
    // to store the largest pixel in the disc
    nPixelsInDisc = (int*)malloc(nPixelInDiscBufferSize);
    CUDA_ERROR_CHECK(
        cudaMalloc(
            (void **)&nPixelsInDiscGPU, 
            nPixelInDiscBufferSize)
    );
    // to store the result from the GPU
    result = (float *)malloc(resultBufferSize);
    CUDA_ERROR_CHECK(
        cudaMalloc(
            (void **)&resultGPU, 
            resultBufferSize)
    );
}

GPUConvolver::~GPUConvolver()
{
}

void GPUConvolver::update_sky(CUDACONV::RunConfig cfg, Sky sky)
{
    /* Allocate buffers if no beam has been added yet.*/
    long nspixels = sky.size();
    skyBufferSize = 4*sizeof(float)*nspixels;
    float* skyTransponsed;
    skyTransponsed = (float*)malloc(skyBufferSize);
    //if(!hasSky)
    //{
    CUDA_ERROR_CHECK(
        cudaMalloc(
            (void **)&skyGPU, 
            skyBufferSize)
    );
    //    hasSky = true;
    //}
    
    for(long i = 0; i < sky.size(); i++)
    {
        skyTransponsed[4*i + 0] = sky.sI[i];
        skyTransponsed[4*i + 1] = sky.sQ[i];
        skyTransponsed[4*i + 2] = sky.sU[i];
        skyTransponsed[4*i + 3] = sky.sV[i];
    }
    CUDA_ERROR_CHECK(
        cudaMemcpy(
            skyGPU,
            skyTransponsed, 
            skyBufferSize,
            cudaMemcpyHostToDevice)
    );
    free(skyTransponsed);
}

void GPUConvolver::update_beam(PolBeam beam)
{
    /* Allocate buffers if no beam has been added yet.*/
    long nbpixels = beam.size();
    beamBufferSize = 6*nbpixels*sizeof(float);
    /* Allocate buffers if no beam has been added yet.*/
    //if(!hasBeam)
    //{
    CUDA_ERROR_CHECK(
        cudaMalloc(
            (void **)&aBeamsGPU, 
            beamBufferSize)
    );
    CUDA_ERROR_CHECK(
        cudaMalloc(
            (void **)&bBeamsGPU, 
            beamBufferSize)
    );
    //}
    /* Update flag to signal that beam was allocated. */
    hasBeam = true;
    /* Transpose and copy to GPU . */
    float* transposedBeams;
    transposedBeams = (float*)malloc(beamBufferSize);
    for(long i = 0; i < nbpixels; i++)
    {
        transposedBeams[6*i + 0] = beam.Da_I[i];
        transposedBeams[6*i + 1] = beam.Da_Qcos[i];
        transposedBeams[6*i + 2] = beam.Da_Qsin[i];
        transposedBeams[6*i + 3] = beam.Da_Ucos[i];
        transposedBeams[6*i + 4] = beam.Da_Usin[i];
        transposedBeams[6*i + 5] = beam.Da_V[i];
    }
    CUDA_ERROR_CHECK(
        cudaMemcpy(
            aBeamsGPU,
            transposedBeams, 
            beamBufferSize,
            cudaMemcpyHostToDevice)
    );
    free(transposedBeams);
    
    transposedBeams = (float*)malloc(beamBufferSize);
    for(long i = 0; i < nbpixels; i++)
    {
        transposedBeams[6*i + 0] = beam.Db_I[i];
        transposedBeams[6*i + 1] = beam.Db_Qcos[i];
        transposedBeams[6*i + 2] = beam.Db_Qsin[i];
        transposedBeams[6*i + 3] = beam.Db_Ucos[i];
        transposedBeams[6*i + 4] = beam.Db_Usin[i];
        transposedBeams[6*i + 5] = beam.Db_V[i];
    }
    CUDA_ERROR_CHECK(
        cudaMemcpy(
            bBeamsGPU,
            transposedBeams, 
            beamBufferSize,
            cudaMemcpyHostToDevice)
    );
    free(transposedBeams);
}

void GPUConvolver::exec_convolution(
    CUDACONV::RunConfig cfg,
    float* data_a, 
    float* data_b,
    char polFlag,
    Scan scan, 
    Sky sky, 
    PolBeam beam)
{
    int p;
    int idx;
    double rmax = beam.get_rho_max();
    
	const double* ra_coords = scan.get_ra_ptr();
	const double* dec_coords = scan.get_dec_ptr();
	const double* pa_coords = scan.get_pa_ptr();
    for(long s = 0; s < nsamples; s += chunkSize) 
    {
        for(int lid = 0; lid < chunkSize; lid++)
        {
            ptgBuffer[4*lid+0] = ra_coords[s+lid]; 
            ptgBuffer[4*lid+1] = dec_coords[s+lid]; 
            ptgBuffer[4*lid+2] = pa_coords[s+lid];
            
            /* Locate all sky pixels inside the beam for 
             * every pointing direction in the Scan. */
            rangeset<int> intraBeamRanges;
            rmax = beam.get_rho_max();
            pointing sc(M_PI_2 - dec_coords[s+lid], ra_coords[s+lid]);
            sky.hpxBase.query_disc(sc, rmax, intraBeamRanges);
            /* Flatten the pixel range list. */
            idx = 0;
            p = lid*cfg.MAX_PIXELS_PER_DISC;
            for(int r=0; r < intraBeamRanges.nranges(); r++) {
                int ss = intraBeamRanges.ivbegin(r);
                int ee = intraBeamRanges.ivend(r);
                for(int ii = ss; ii < ee; ii++) {
                    skyPixelsInBeam[p+idx] = ii;
                    idx++;
                }
            }
            nPixelsInDisc[lid] = idx;
        }
        
        // send pointing to gpu.
        CUDA_ERROR_CHECK(
            cudaMemcpy(
                ptgBufferGPU,
                ptgBuffer,
                ptgBufferSize,
                cudaMemcpyHostToDevice)
        );
        
        // send sky pixels in beam to gpu.
        CUDA_ERROR_CHECK(
            cudaMemcpy(
                skyPixelsInBeamGPU,
                skyPixelsInBeam,
                skyPixelsInBeamBufferSize,
                cudaMemcpyHostToDevice)
        );
        
        //// send max pixel number to gpu.
        CUDA_ERROR_CHECK(
            cudaMemcpy(
                nPixelsInDiscGPU,
                nPixelsInDisc,
                nPixelInDiscBufferSize,
                cudaMemcpyHostToDevice)
        );
        
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        
        // "Multiply" sky times the beam. */
        beam_times_sky2(
            cfg,
            chunkSize, 
            ptgBufferGPU,
            beam.get_nside(), 
            beam.size(), 
            aBeamsGPU, 
            bBeamsGPU,
            sky.get_nside(),
            skyGPU, 
            skyPixelsInBeamGPU, 
            nPixelsInDiscGPU,
            resultGPU
        );
        // bring data back from gpu. 
        CUDA_ERROR_CHECK(
            cudaMemcpy(
                result,
                resultGPU,
                resultBufferSize,
                cudaMemcpyDeviceToHost)
        );
        
         /* bring the data from buffer to useful format. */
        for(int lid = 0; lid < chunkSize; lid++)
        {
            if(polFlag == 'a')
                data_a[s+lid] += result[lid];
            //if(polFlag == 'b')
            //    data_b[s+lid] += result[2*lid+1];
        } 
       //std::cout << s << " " << nsamples << std::endl;
    }
}
