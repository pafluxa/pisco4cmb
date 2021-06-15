#include <omp.h>
#include <stdexcept>
#include <string.h>

#include "GpuConvolver/gpuconvolver.hpp"

// macro-function combo to catch GPU errors
#define CUDA_ERROR_CHECK(ans) {gpuAssert((ans), __FILE__, __LINE__);}   
inline void 
gpuAssert(cudaError_t code, const char file[], int line, bool abort=true)                         
{                                                                                                             
    if (code != cudaSuccess)                                                                                   
    {                                                                                                          
        std::cerr << "ERROR (GPUassert): " 
            << "A CUDA related error was encountered. "
            << "Error is: " << cudaGetErrorString(code) << ", triggered "
            << "from file " << file 
            << " at line " << line << std::endl;                          
        if(abort)
        { 
            std::cerr << "FATAL (GPUassert): " 
                << "program was aborted." << std::endl;
            exit(-1); 
        }  
        else
        {
            std::cerr << "WARNING (GPUassert):"
                << "press any key to continue." << std::endl;
            getchar();
        }
    }                                                                                                          
}                                                                                                             

GPUConvolver::GPUConvolver(CUDACONV::RunConfig _cfg)
{
    cfg = _cfg;
    hasSky = false;
    hasBeam = false;
    hasValidConfig = false;
    /* Calculate size of all buffers that will be used. */
    pixPerDisc = cfg.pixelsPerDisc;
    ptgPerConv = cfg.ptgPerConv;
    // 2 x size because there are two detectors. This is a small array
    // anyways, not worth to go cheap on memory
    resultBufferSize = 2 * ptgPerConv * sizeof(float);
    ptgBufferSize = N_POINTING_COORDS * ptgPerConv * sizeof(float);
    // buffer holding all pixels seen by a particular pointing.
    // this array can get pretty large so be careful
    skyPixelsInBeamBufferSize = ptgPerConv * pixPerDisc;
    skyPixelsInBeamBufferSize *= sizeof(int);
    // array with the size of seen sky pixels
    nPixelInDiscBufferSize = ptgPerConv * sizeof(int);
    /* Check that required memory is below the user limit. */
    const int ns = cfg.nStreams;
    size_t reqMemory = ns * (resultBufferSize 
      + ptgBufferSize + nPixelInDiscBufferSize 
      + skyPixelsInBeamBufferSize);
    // check required memory is within limits set by user.
    hasValidConfig = (reqMemory < cfg.maxMemUsage) && (cfg.nStreams > 0);
    std::cerr << reqMemory << std::endl;
    std::cerr << cfg.maxMemUsage << std::endl;
    std::cerr << cfg.nStreams << std::endl;
    configure_execution();
    allocate_buffers();
}

void GPUConvolver::configure_execution(void)
{
    if(cfg.nStreams > CUDACONV_MAXSTREAMS)
    {
        hasValidConfig = false;
        std::cerr << "MESSAGE (cudaconv): " 
            << cfg.nStreams << " must be lower than " 
            << CUDACONV_MAXSTREAMS << std::endl;
    }
    if(!hasValidConfig)
    {
        throw std::invalid_argument(
            "ERROR (GPUConvolver): invalid configuration.");
    }
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, cfg.deviceId));
    //#ifdef CUDACONV_DEBUG
    std::cerr << "MESSAGE (GPUConvolver): "
        << "Executing on device id: " << prop.name << std::endl;
    //#endif
    /* Set device to the one specified by the user. */
    CUDA_ERROR_CHECK(cudaSetDevice(cfg.deviceId));
    /* Create events and streams for pipelined convolution. */
    for(int i = 0; i < cfg.nStreams; i++)
    {
        CUDA_ERROR_CHECK
        (  
            cudaStreamCreate(&(streams[i]))
        );
        CUDA_ERROR_CHECK(
            cudaEventCreate(&(streamWait[i]))
        );
    }
}

void GPUConvolver::allocate_buffers(void)
{
    for(int i = 0; i < cfg.nStreams; i++)
    {
    // allocate buffers to store partial pointing
        CUDA_ERROR_CHECK(
            cudaMallocHost(
                (void **)&(ptgBuffer[i]), ptgBufferSize)
        );
        CUDA_ERROR_CHECK(
            cudaMalloc(
                (void **)&(ptgBufferGPU[i]), ptgBufferSize)
        );
        // to store intra-beam sky pixels (host)
        CUDA_ERROR_CHECK(
            cudaMallocHost((void **)&(skyPixelsInBeam[i]), 
                skyPixelsInBeamBufferSize)
        );
        // to store intra-beam sky pixels (GPU)
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&(skyPixelsInBeamGPU[i]), 
                skyPixelsInBeamBufferSize)
        );
        // to store the number of pixels in every disc (host)
        CUDA_ERROR_CHECK(
            cudaMallocHost((void **)&(nPixelsInDisc[i]), 
                nPixelInDiscBufferSize)
        );
        // to store the number of pixels in every disc (GPU)
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&(nPixelsInDiscGPU[i]), 
                nPixelInDiscBufferSize)
        );
        // to store the result of a partial convolution (GPU)
        CUDA_ERROR_CHECK(
            cudaMallocHost((void **)&(result[i]), 
                resultBufferSize)
        );
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&(resultGPU[i]), 
                resultBufferSize)
        );
    }
    std::cerr << "memory allocated." << std::endl;
}

GPUConvolver::~GPUConvolver()
{
    if(hasValidConfig)
    {
        for(int i = 0; i < cfg.nStreams; ++i)
        {
            CUDA_ERROR_CHECK
            (
                cudaStreamDestroy(streams[i])
            );
            CUDA_ERROR_CHECK
            (
                cudaEventDestroy(streamWait[i])
            );
        }
        for(int i = 0; i < cfg.nStreams; ++i)
        {
            CUDA_ERROR_CHECK(cudaFreeHost(result[i]));
            CUDA_ERROR_CHECK(cudaFree(resultGPU[i]));
            CUDA_ERROR_CHECK(cudaFreeHost(nPixelsInDisc[i]));
            CUDA_ERROR_CHECK(cudaFree(nPixelsInDiscGPU[i]));
            CUDA_ERROR_CHECK(cudaFreeHost(skyPixelsInBeam[i]));
            CUDA_ERROR_CHECK(cudaFree(skyPixelsInBeamGPU[i]));
            CUDA_ERROR_CHECK(cudaFreeHost(ptgBuffer[i]));
            CUDA_ERROR_CHECK(cudaFree(ptgBufferGPU[i]));
        }
    }
    if(hasBeam)
    {
        CUDA_ERROR_CHECK(cudaFree(aBeamsGPU));
    }
    if(hasSky)
    {
        CUDA_ERROR_CHECK(cudaFree(skyGPU));
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
    std::cerr << "sky updated." << std::endl;
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
    std::cerr << "beams updated." << std::endl;
}

void GPUConvolver::exec_convolution(
    float* data_a, float* data_b,
    Scan* scan, Sky* sky, PolBeam* beam)
{
    int samp;
    int strm;
    long nsamples;
    double rmax; 

    nsamples = scan->get_nsamples();
    rmax = beam->get_rho_max();
	
    const float* ra_coords = scan->get_ra_ptr();
	const float* dec_coords = scan->get_dec_ptr();
	const float* pa_coords = scan->get_pa_ptr();
    for(samp = 0; samp < nsamples; samp += cfg.nStreams * ptgPerConv) 
    { 
        std::cerr << samp << "/" << nsamples << std::endl;
        
        for(strm = 0; strm < cfg.nStreams; strm++)
        { 
            std::cerr 
                << "stream: " << strm 
                << " filling buffers for pointing and pixels-in-beam." 
                << std::endl;
            #pragma omp parallel default(shared)
            // begin parallel region
            {
            float ra;
            float dec;
            float pa;
            
            int idx;
            int sidx;
            
            #pragma omp for schedule(static)
            for(int k = 0; k < ptgPerConv; k++)
            {
                int aidx = samp + strm * ptgPerConv + k;
                // send dummy data
                if(aidx >= nsamples)
                { 
                    ptgBuffer[strm][N_POINTING_COORDS * k + 0] = 0.0;
                    ptgBuffer[strm][N_POINTING_COORDS * k + 1] = 0.0;
                    ptgBuffer[strm][N_POINTING_COORDS * k + 2] = 0.0;
                    nPixelsInDisc[strm][k] = 0;
                    skyPixelsInBeam[strm][k * pixPerDisc + 0] = 0;
                }
                else
                {
                    ra = ra_coords[aidx];
                    dec = dec_coords[aidx];
                    // Passed arguments are counterclockwise on the sky
                    // while CMB requires clockwise arguments.
                    pa = -pa_coords[aidx];
                    ptgBuffer[strm][N_POINTING_COORDS * k + 0] = ra; 
                    ptgBuffer[strm][N_POINTING_COORDS * k + 1] = dec; 
                    ptgBuffer[strm][N_POINTING_COORDS * k + 2] = pa;
                    // Locate all sky pixels inside the beam for every 
                    // pointing direction in the scan
                    rangeset<int> intraBeamRanges;
                    pointing sc(
                        M_PI_2 - dec_coords[aidx], ra_coords[aidx]);
                    sky->hpxBase.query_disc(sc, rmax, intraBeamRanges);
                    // convert ranges to pixel list 
                    idx = 0;
                    for(int r = 0; r < intraBeamRanges.nranges(); r++) 
                    {
                        int ss = intraBeamRanges.ivbegin(r);
                        int ee = intraBeamRanges.ivend(r);
                        for(int ii = ss; ii < ee; ii++) 
                        {
                            sidx = k * pixPerDisc + idx;
                            skyPixelsInBeam[strm][sidx] = ii;
                            idx++;
                        }
                    }
                    nPixelsInDisc[strm][k] = idx;
                }
            }
            // end parallel region
            }
        }
        // do all the gpu stuff
        for(strm = 0; strm < cfg.nStreams; strm++)
        {
            std::cerr 
                << "stream: " << strm 
                << " copying ptgBuffer to gpu" << std::endl;
            // send pointing to gpu asynchronously
            //CUDA_ERROR_CHECK(
                cudaMemcpyAsync(
                    ptgBufferGPU[strm],
                    ptgBuffer[strm],
                    ptgBufferSize,
                    cudaMemcpyHostToDevice, streams[strm]);
            //);
        }
        for(strm = 0; strm < cfg.nStreams; strm++)
        {
            std::cerr
                << "stream: " << strm 
                << " copying nPixelsInDisc to gpu" << std::endl;
            // send max pixel number per-disc to gpu.
            //CUDA_ERROR_CHECK(
                cudaMemcpyAsync(
                    nPixelsInDiscGPU[strm],
                    nPixelsInDisc[strm],
                    nPixelInDiscBufferSize,
                    cudaMemcpyHostToDevice, streams[strm]);
            //);
        }
        for(strm = 0; strm < cfg.nStreams; strm++)
        {
            std::cerr 
                << "stream: " << strm 
                << " copying skyPixelsInBeam to gpu" << std::endl;
            // send sky pixels in beam to gpu.
            //CUDA_ERROR_CHECK(
                cudaMemcpyAsync(
                    skyPixelsInBeamGPU[strm],
                    skyPixelsInBeam[strm],
                    skyPixelsInBeamBufferSize,
                    cudaMemcpyHostToDevice, streams[strm]);
            //);
        }
        std::cerr 
            << "stream: " << strm 
            << " executing gpu kernel on stream " << strm 
            << "." << std::endl;
        // and run all the kernels
        // calculate partial convolution of beam and sky
        for(strm = 0; strm < cfg.nStreams; strm++)
        {
            CUDACONV::streamed_beam_times_sky(
                cfg,
                ptgBufferGPU[strm],
                beam->get_nside(), beam->size(), 
                aBeamsGPU, bBeamsGPU,
                sky->get_nside(), skyGPU, 
                skyPixelsInBeamGPU[strm], nPixelsInDiscGPU[strm],
                streams[strm],
                resultGPU[strm]);
        }
        // bring the data from GPU back to the host buffer.
        /** why so serial! **/
        // explicitly disable dynamic teams!
        //omp_set_dynamic(0);     
        // Use nStreams threads for all consecutive parallel regions
        //omp_set_num_threads(cfg.nStreams); 
        //#pragma omp parallel default(shared) private(strm)
       // {
        for(strm = 0; strm < cfg.nStreams; strm++)
        {
            //strm = omp_get_thread_num();
            std::cerr 
                << "stream: " << strm 
                << " copying data back." << std::endl;
            if(beam->enabledDets == 'a' || beam->enabledDets == 'p')
            {
                CUDA_ERROR_CHECK
                (
                    cudaMemcpyAsync(
                        &(data_a[samp + strm * ptgPerConv]),
                        &(resultGPU[strm][0]),
                        resultBufferSize / 2,
                        cudaMemcpyDeviceToHost, 
                        streams[strm])
                );
            }
            if(beam->enabledDets == 'b' || beam->enabledDets == 'p')
            {
                CUDA_ERROR_CHECK
                (
                    cudaMemcpyAsync(
                        &(data_b[samp + strm * ptgPerConv]),
                        &(resultGPU[strm][ptgPerConv]),
                        resultBufferSize / 2,
                        cudaMemcpyDeviceToHost, 
                        streams[strm])
                );
            }
        }
    }
}
