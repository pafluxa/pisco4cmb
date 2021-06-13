#include <omp.h>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

#include "GpuConvolver/gpuconvolver.hpp"

// macro-function combo to catch GPU errors
#define CUDA_ERROR_CHECK(ans) {gpuAssert((ans), __FILE__, __LINE__);}   
inline void 
gpuAssert(cudaError_t code, const char file[], int line, bool abort=true)                         
{                                                                                                             
    if (code != cudaSuccess)                                                                                   
    {                                                                                                          
        fprintf(stderr,
        "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);                          
        if(abort)
        { 
            exit(-1); 
        }                                                                                     
    }                                                                                                          
}                                                                                                             

GPUConvolver::GPUConvolver(Scan& scan, CUDACONV::RunConfig cfg)
{
    nsamples = scan.get_nsamples();
    hasSky = false;
    hasBeam = false;
    
    pixPerDisc = cfg.pixelsPerDisc;
    if(pixPerDisc > cfg.MAX_PIXELS_PER_DISC)
    {
        std::cerr 
          << "WARNING: too many pixels per disc. Using : " 
          << cfg.MAX_PIXELS_PER_DISC << " pixels per disk." 
          << std::endl;
        pixPerDisc = cfg.MAX_PIXELS_PER_DISC;
    }

    ptgPerConv = cfg.ptgPerConv;
    if(ptgPerConv > cfg.MAX_PTGS_PER_CONV)
    {
        std::cerr 
          << "WARNING: too many pointings per convolution. Using : " 
          << cfg.MAX_PTGS_PER_CONV << " pointings per convolution." 
          << std::endl;
        ptgPerConv = cfg.MAX_PTGS_PER_CONV;
    }
    // 2 x size because there are two detectors. This is a small array
    // anyways, not worth to go cheap on memory
    resultBufferSize = 2 * ptgPerConv * sizeof(float);
    ptgBufferSize = N_POINTING_COORDS * ptgPerConv * sizeof(float);
    // buffer holding all pixels seen by a particular pointing.
    // this array can get pretty large so be careful
    nPixelInDiscBufferSize = ptgPerConv * sizeof(int);
    skyPixelsInBeamBufferSize = ptgPerConv * pixPerDisc * sizeof(int);
    
    // allocate buffers to store partial pointing
    CUDA_ERROR_CHECK(
        cudaMallocHost(
            (void **)&ptgBuffer, 
            ptgBufferSize)
    );
    CUDA_ERROR_CHECK(
        cudaMalloc(
            (void **)&ptgBufferGPU, 
            ptgBufferSize)
    );
    // to store intra-beam sky pixels (host)
    CUDA_ERROR_CHECK(
        cudaMallocHost((void **)&skyPixelsInBeam, 
            skyPixelsInBeamBufferSize)
    );
    // to store intra-beam sky pixels (GPU)
    CUDA_ERROR_CHECK(
        cudaMalloc((void **)&skyPixelsInBeamGPU, 
            skyPixelsInBeamBufferSize)
    );
    // to store the number of pixels in every disc (host)
    CUDA_ERROR_CHECK(
        cudaMallocHost((void **)&nPixelsInDisc, nPixelInDiscBufferSize)
    );
    // to store the number of pixels in every disc (GPU)
    CUDA_ERROR_CHECK(
        cudaMalloc((void **)&nPixelsInDiscGPU, nPixelInDiscBufferSize)
    );
    // to store the result of a partial convolution (GPU)
    CUDA_ERROR_CHECK(
        cudaMallocHost((void **)&result, resultBufferSize)
    );
    CUDA_ERROR_CHECK(
        cudaMalloc((void **)&resultGPU, resultBufferSize)
    );
    
    // set device to prefer L1 cache
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}

GPUConvolver::~GPUConvolver()
{
    cudaFree(resultGPU);
    cudaFreeHost(result);
    cudaFree(nPixelsInDiscGPU);
    cudaFreeHost(nPixelsInDisc);
    cudaFree(skyPixelsInBeamGPU);
    cudaFreeHost(skyPixelsInBeamGPU);
    cudaFreeHost(ptgBuffer);
    cudaFree(ptgBufferGPU);
    
    if(hasBeam)
    {
        cudaFree(aBeamsGPU);
    }
    if(hasSky)
    {
        cudaFree(skyGPU);
    }
}

void GPUConvolver::update_sky(Sky* sky)
{
    /* Allocate temporal buffers to hold the transposed sky*/
    long nspixels = sky->size();
    skyBufferSize = N_SKY_COMP * sizeof(float) * nspixels;
    float* skyTransponsed;
    skyTransponsed = (float*)malloc(skyBufferSize);
    /* Allocate buffers if no sky has been allocated yet.*/
    if(!hasSky)
    {
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&skyGPU, skyBufferSize));
    }
    hasSky = true;
    for(long i = 0; i < sky->size(); i++)
    {
        skyTransponsed[4 * i + 0] = sky->sI[i];
        skyTransponsed[4 * i + 1] = sky->sQ[i];
        skyTransponsed[4 * i + 2] = sky->sU[i];
        skyTransponsed[4 * i + 3] = sky->sV[i];
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

void GPUConvolver::update_beam(PolBeam* beam)
{
    /* Allocate buffers if no beam has been added yet.*/
    long nbpixels = beam->size();
    
    beamBufferSize = N_POLBEAMS * nbpixels * sizeof(float);
    /* Allocate buffers if no beam has been added yet.*/
    if(!hasBeam && 
      (beam->enabledDets == 'a' || beam->enabledDets == 'p'))
    {
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&aBeamsGPU, beamBufferSize)
        );
    }
    if(!hasBeam && 
      (beam->enabledDets == 'b' || beam->enabledDets == 'p'))
    {
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&bBeamsGPU, beamBufferSize)
        );
    }
    /* Update flag to signal that beam was allocated. */
    hasBeam = true;
    /* Transpose and copy to GPU . */
    float* transposedBeams;
    transposedBeams = (float*)malloc(N_POLBEAMS * beamBufferSize);
    if(beam->enabledDets == 'a' || beam->enabledDets == 'p')
    {
        for(long i = 0; i < nbpixels; i++)
        {
            transposedBeams[N_POLBEAMS * i + 0] = beam->aBeams[0][i];
            transposedBeams[N_POLBEAMS * i + 1] = beam->aBeams[1][i];
            transposedBeams[N_POLBEAMS * i + 2] = beam->aBeams[2][i];
            transposedBeams[N_POLBEAMS * i + 3] = beam->aBeams[3][i];
        }
        CUDA_ERROR_CHECK(
            cudaMemcpy(
                aBeamsGPU,
                transposedBeams, 
                beamBufferSize,
                cudaMemcpyHostToDevice)
        );
    }
    if(beam->enabledDets == 'b' || beam->enabledDets == 'p')
    {
        for(long i = 0; i < nbpixels; i++)
        {
            transposedBeams[N_POLBEAMS*i + 0] = beam->bBeams[0][i];
            transposedBeams[N_POLBEAMS*i + 1] = beam->bBeams[1][i];
            transposedBeams[N_POLBEAMS*i + 2] = beam->bBeams[2][i];
            transposedBeams[N_POLBEAMS*i + 3] = beam->bBeams[3][i];
        }
        CUDA_ERROR_CHECK(
            cudaMemcpy(
                bBeamsGPU,
                transposedBeams, 
                beamBufferSize,
                cudaMemcpyHostToDevice)
        );
    }
    free(transposedBeams);
}

void GPUConvolver::exec_convolution(
    CUDACONV::RunConfig cfg,
    float* data_a, 
    float* data_b,
    Scan* scan, Sky* sky, PolBeam* beam)
{
    int idx;
    long nsamples = scan->get_nsamples();
    double rmax = beam->get_rho_max();
    
	const float* ra_coords = scan->get_ra_ptr();
	const float* dec_coords = scan->get_dec_ptr();
	const float* pa_coords = scan->get_pa_ptr();
    for(long s = 0; s < nsamples; s += ptgPerConv) 
    { 
        for(int k = 0; k < ptgPerConv; k++)
        {
            // Locate all sky pixels inside the beam for every 
            // pointing direction in the Scan
            rangeset<int> intraBeamRanges;
            pointing sc(M_PI_2 - dec_coords[s + k], ra_coords[s + k]);
            sky->hpxBase.query_disc(sc, rmax, intraBeamRanges);
            // Flatten the pixel range list. 
            idx = 0;
            for(int r = 0; r < intraBeamRanges.nranges(); r++) 
            {
                int ss = intraBeamRanges.ivbegin(r);
                int ee = intraBeamRanges.ivend(r);
                for(int ii = ss; ii < ee; ii++) 
                {
                    skyPixelsInBeam[k * pixPerDisc + idx] = ii;
                    idx++;
                }
            }
            nPixelsInDisc[k] = idx;
        } 
        // send sky pixels in beam to gpu.
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(
                skyPixelsInBeamGPU,
                skyPixelsInBeam,
                skyPixelsInBeamBufferSize,
                cudaMemcpyHostToDevice)
        );
        // send max pixel number per-disc to gpu.
        CUDA_ERROR_CHECK
        (
            cudaMemcpyAsync(
                nPixelsInDiscGPU,
                nPixelsInDisc,
                nPixelInDiscBufferSize,
                cudaMemcpyHostToDevice)
        );
        for(int k = 0; k < ptgPerConv; k++)
        {
            // it is simpler to just send dummy data 
            if(s + k > nsamples)
            { 
                ptgBuffer[N_POINTING_COORDS * k + 0] = 0.0;
                ptgBuffer[N_POINTING_COORDS * k + 1] = 0.0;
                ptgBuffer[N_POINTING_COORDS * k + 2] = 0.0;
                // this is a dummy entry
                ptgBuffer[N_POINTING_COORDS * k + 3] = 0.0;
                idx = 0;
                nPixelsInDisc[k] = 0;
                skyPixelsInBeam[k * pixPerDisc + idx] = 0;
            }
            else
            {
                ptgBuffer[N_POINTING_COORDS * k + 0] = ra_coords[s + k]; 
                ptgBuffer[N_POINTING_COORDS * k + 1] = dec_coords[s + k]; 
                // Passed arguments are counterclockwise on the sky, while
                // CMB requires clockwise arguments.
                ptgBuffer[N_POINTING_COORDS * k + 2] = -pa_coords[s + k];
                // this is a dummy entry
                ptgBuffer[N_POINTING_COORDS * k + 3] = 0.0;
            }
        }
        // send pointing to gpu.
        CUDA_ERROR_CHECK(
            cudaMemcpy(
                ptgBufferGPU,
                ptgBuffer,
                ptgBufferSize,
                cudaMemcpyHostToDevice)
        );

        // calculate partial convolution of beam and sky
        CUDACONV::beam_times_sky(
            cfg,
            ptgBufferGPU,
            beam->get_nside(), beam->size(), 
            aBeamsGPU, bBeamsGPU,
            sky->get_nside(), skyGPU, 
            skyPixelsInBeamGPU, nPixelsInDiscGPU,
            resultGPU);
        // bring the data from GPU back to the host buffer.
        CUDA_ERROR_CHECK
        (
            cudaMemcpy(
                result,
                resultGPU,
                resultBufferSize,
                cudaMemcpyDeviceToHost)
        );
        for(int k = 0; k < ptgPerConv; k++)
        {   
            if(beam->enabledDets == 'a' || beam->enabledDets == 'p')
            {
                data_a[s + k] = result[2*k + 0];
            }
            if(beam->enabledDets == 'b' || beam->enabledDets == 'p')
            {
                data_b[s + k] = result[2*k + 1];
            }
        }
    }
}
