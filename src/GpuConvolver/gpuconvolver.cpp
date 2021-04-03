#include <cuda.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Polbeam/polbeam.hpp"
#include "GpuConvolver/gpuconvolver.hpp"
#include "GpuConvolver/cuda/cuda_error_check.h"

#include "cuda/beam_times_sky.h"

GPUConvolver::GPUConvolver(long _nsamples)
{
    nsamples = _nsamples;
    hasSky = false;
    hasBeam = false;
    #pragma omp parallel
        parallelLaunches = omp_get_num_threads();
    result = (float**)malloc(sizeof(float*)*parallelLaunches);
    resultGPU = (float**)malloc(sizeof(float*)*parallelLaunches);
    for(int i = 0; i < parallelLaunches; i++)
    {
        CUDA_ERROR_CHECK(
                cudaMalloc((void **)&(resultGPU[i]),
                sizeof(float)*2*CUDACONV::GRID_SIZE)
        );
        result[i] = (float*)malloc(sizeof(float)*2*CUDACONV::GRID_SIZE);
    }
}

GPUConvolver::~GPUConvolver()
{
    if(hasSky)
    {
        CUDA_ERROR_CHECK(cudaFreeHost(skyGPU));
    }
    if(hasBeam)
    {
        CUDA_ERROR_CHECK(cudaFreeHost(aBeamsGPU));
        CUDA_ERROR_CHECK(cudaFreeHost(bBeamsGPU));
    }
}

void GPUConvolver::update_sky(Sky sky)
{
    /* Allocate buffers if no beam has been added yet.*/
    long nspixels = sky.size();
    skySize = sizeof(float)*nspixels;
    float *skyTransponsed = (float*)malloc(4*skySize);
    if(!hasSky)
    {
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&skyGPU, 4*skySize)
        );
        skyPixelsInBeamSize = sizeof(int)*sky.size();
        for(int p = 0; p < parallelLaunches; p++)
        {
            int* temp;
            temp = nullptr;            
            CUDA_ERROR_CHECK(
                cudaHostAlloc(
                    (void **)&temp,
                    skyPixelsInBeamSize,
                    cudaHostAllocWriteCombined)
            );
            skyPixelsInBeamGPU.push_back(temp);
            
            temp = (int*)malloc(skyPixelsInBeamSize);
            skyPixelsInBeam.push_back(temp);
        }
        
        hasSky = true;
    }
    
    for(long i = 0; i < sky.size(); i++)
    {
        skyTransponsed[4*i+0] = sky.sI[i];
        skyTransponsed[4*i+1] = sky.sQ[i];
        skyTransponsed[4*i+2] = sky.sU[i];
        skyTransponsed[4*i+3] = sky.sV[i];
    }
    CUDA_ERROR_CHECK(
        cudaMemcpy(
            skyGPU,
            skyTransponsed, 
            4*skySize,
            cudaMemcpyHostToDevice)
    );

    free(skyTransponsed);
}

void GPUConvolver::update_beam(PolBeam beam)
{
    /* Allocate buffers if no beam has been added yet.*/
    long nbpixels = beam.size();
    beamSize = sizeof(float)*nbpixels;
    /* Allocate buffers if no beam has been added yet.*/
    if(!hasBeam)
    {
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&aBeamsGPU, 6*beamSize)
        );
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&bBeamsGPU, 6*beamSize)
        );
    }
    /* Update flag to signal that beam was allocated. */
    hasBeam = true;
    /* Transpose and copy to GPU . */
    float* transposedBeams = (float*)malloc(6*beamSize);
    for(long i = 0; i < nbpixels; i++)
    {
        transposedBeams[6*i+0] = beam.Da_I[i];
        transposedBeams[6*i+1] = beam.Da_Qcos[i];
        transposedBeams[6*i+2] = beam.Da_Qsin[i];
        transposedBeams[6*i+3] = beam.Da_Ucos[i];
        transposedBeams[6*i+4] = beam.Da_Usin[i];
        transposedBeams[6*i+5] = beam.Da_V[i];
    }
    CUDA_ERROR_CHECK(
        cudaMemcpy(
            aBeamsGPU,
            transposedBeams, 
            6*beamSize,
            cudaMemcpyHostToDevice)
    );

    for(long i = 0; i < beam.size(); i++)
    {
        transposedBeams[6*i+0] = beam.Db_I[i];
        transposedBeams[6*i+1] = beam.Db_Qcos[i];
        transposedBeams[6*i+2] = beam.Db_Qsin[i];
        transposedBeams[6*i+3] = beam.Db_Ucos[i];
        transposedBeams[6*i+4] = beam.Db_Usin[i];
        transposedBeams[6*i+5] = beam.Db_V[i];
    }
    CUDA_ERROR_CHECK(
        cudaMemcpy(
            bBeamsGPU,
            transposedBeams, 
            6*beamSize,
            cudaMemcpyHostToDevice)
    );

    free(transposedBeams);
}

void GPUConvolver::exec_convolution(
    float* data_a, float* data_b,
    char polFlag,
    Scan scan, Sky sky, PolBeam beam)
{
	const double* ra_coords = scan.get_ra_ptr();
	const double* dec_coords = scan.get_dec_ptr();
	const double* pa_coords = scan.get_pa_ptr();

    cudaStream_t streams[parallelLaunches];
    for(int i=0; i < parallelLaunches; i++)
    {
        cudaStreamCreateWithFlags(
            &(streams[i]), cudaStreamNonBlocking);
    }

    #pragma omp parallel default(shared)
    {   
        int idx;
        int npixDisc;
    
        double ra_bc;
        double dec_bc;
        double pa_bc;
    
        int lid = omp_get_thread_num();
        for(long s = lid; s < nsamples; s += parallelLaunches) 
        {
            int lid = omp_get_thread_num();
            double rmax;
            rangeset<int> intraBeamRanges;
            ra_bc = ra_coords[s]; 
            dec_bc = dec_coords[s]; 
            pa_bc = pa_coords[s];
            
            /* Locate all sky pixels inside the beam for 
             * every pointing direction in the Scan. */
            rmax = beam.get_rho_max();
            pointing sc(M_PI_2 - dec_bc, ra_bc);
            sky.hpxBase.query_disc(sc, rmax, intraBeamRanges);
            /* Flatten the pixel range list. */
            idx = 0;
            for(int r=0; r < intraBeamRanges.nranges(); r++) 
            {
                int s = intraBeamRanges.ivbegin(r);
                int e = intraBeamRanges.ivend(r);
                for(int i=s; i < e; i++) 
                {
                    skyPixelsInBeam[lid][idx] = i;
                    idx++;
                }
            }
            npixDisc = idx;
            CUDA_ERROR_CHECK(
                cudaMemcpyAsync(
                    skyPixelsInBeamGPU[lid],
                    skyPixelsInBeam[lid],
                    npixDisc*sizeof(int),
                    cudaMemcpyHostToDevice,
                    streams[lid])
            );
            /* "Multiply" sky times the beam. */
            CUDACONV::beam_times_sky(
                ra_bc, dec_bc, pa_bc, 
                beam.get_nside(), beam.size(), 
                aBeamsGPU, bBeamsGPU,
                sky.get_nside(), skyGPU, 
                npixDisc, skyPixelsInBeamGPU[lid],
                streams[lid],
                resultGPU[lid]);
            
            cudaStreamSynchronize(streams[lid]);
            
            CUDA_ERROR_CHECK(
                cudaMemcpy(
                    result[lid], 
                    resultGPU[lid], 
                    sizeof(float)*2*CUDACONV::GRID_SIZE,
                    cudaMemcpyDeviceToHost)
            );
            float resultA = 0.0;
            float resultB = 0.0;
            /* bring the buffer back. */
            for(int i = 0; i < CUDACONV::GRID_SIZE; i++)
            {
                resultA += result[lid][2*i];
                //resultB += result[lid][2*i+1];
            }
            data_a[s] = resultA;
        }
    }
    
    for(int i = 0; i < parallelLaunches; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
}
