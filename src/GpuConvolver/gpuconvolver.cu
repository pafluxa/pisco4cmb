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
        std::cerr << "GPUassert: " 
            << "A CUDA related error was encountered. "
            << "Error is: " << cudaGetErrorString(code) << ", triggered "
            << "from file " << file 
            << " at line " << line << std::endl;                          
        if(abort)
        { 
            std::cerr << "FATAL: " 
                << "program was aborted." << std::endl;
            exit(-1); 
        }  
        else
        {
            std::cerr << "WARNING:"
                << "press any key to continue." << std::endl;
            getchar();
        }
    }                                                                                                          
}                                                                                                             

GPUConvolver::GPUConvolver(CUDACONV::RunConfig _cfg)
{
    cfg = _cfg;
    
    // todo: fix this!!!
    cfg.nStreams = 1;
    
    hasSky = false;
    hasBeam = false;
    hasScan = false;
    hasValidConfig = false;
    /* Calculate size of all buffers that will be used. */
    pixPerDisc = cfg.pixelsPerDisc;
    ptgPerConv = cfg.ptgPerConv;
    // 2 x size because there are two detectors. This is a small array
    // anyways, not worth to go cheap on memory
    resultBufferSize = ptgPerConv * sizeof(float);
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
    std::cerr << "GPUConvolver::constructor: ";
    std::cerr << "is configuration is valid? " << hasValidConfig << std::endl;
    std::cerr << "GPUConvolver::constructor: ";
    std::cerr << "requested memory is " << reqMemory << " bytes and ";
    std::cerr << "available memory is " << cfg.maxMemUsage << " bytes. " << std::endl;
}

void GPUConvolver::configure_execution(void)
{
    if(!hasValidConfig)
    {
        throw std::invalid_argument(
            "FATAL: GPUConvolver has an invalid configuration.");
    }
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, cfg.deviceId));
    std::cerr << "GPUConvolver::configure_execution: ";
    std::cerr << "Executing on device id: " << prop.name << std::endl;
    /* Set device to the one specified by the user. */
    CUDA_ERROR_CHECK(cudaSetDevice(cfg.deviceId));
    
    allocate_buffers();
}

void GPUConvolver::allocate_buffers(void)
{
    std::cerr << "GPUConvolver::allocate_buffers: ";
    std::cerr << "allocating GPU pointing buffer" << std::endl;
    CUDA_ERROR_CHECK
    (
        cudaMalloc(
            (void **)&(ptgBufferGPU), ptgBufferSize)
    );
    // to store intra-beam sky pixels (host)
    std::cerr << "GPUConvolver::allocate_buffers: ";
    std::cerr << "allocating host intra-beam pixel buffer" << std::endl;
    CUDA_ERROR_CHECK
    (
        cudaMallocHost((void **)&(skyPixelsInBeam), 
            skyPixelsInBeamBufferSize)
    );
    // to store intra-beam sky pixels (GPU)
    std::cerr << "GPUConvolver::allocate_buffers: ";
    std::cerr << "allocating GPU intra-beam pixel buffer" << std::endl;
    CUDA_ERROR_CHECK
    (
        cudaMalloc((void **)&(skyPixelsInBeamGPU), 
            skyPixelsInBeamBufferSize)
    );
    CUDA_ERROR_CHECK
    (
        cudaMemset(skyPixelsInBeamGPU, 0, skyPixelsInBeamBufferSize)
    );
    
    //allocate storage for matchingBeamPixels
    std::cerr << "GPUConvolver::allocate_buffers: ";
    std::cerr << "allocating GPU intra-beam pixel buffer (beam pixels)" << std::endl;
    CUDA_ERROR_CHECK
    (
        cudaMalloc((void **)&(matchingBeamPixelsGPU), 
            skyPixelsInBeamBufferSize)
    );
    CUDA_ERROR_CHECK
    (
        cudaMemset(matchingBeamPixelsGPU, 0, skyPixelsInBeamBufferSize)
    );

    //allocate storage for chiAngles
    std::cerr << "GPUConvolver::allocate_buffers: ";
    std::cerr << "allocating GPU chi angles." << std::endl;
    CUDA_ERROR_CHECK
    (
        cudaMalloc((void **)&(chiAnglesGPU), 
            skyPixelsInBeamBufferSize)
    );
    CUDA_ERROR_CHECK
    (
        cudaMemset(chiAnglesGPU, 0, skyPixelsInBeamBufferSize)
    );

    // to store the number of pixels in every disc (host)
    std::cerr << "GPUConvolver::allocate_buffers: ";
    std::cerr << "allocating host npixels in intra-beam buffer" << std::endl;
    CUDA_ERROR_CHECK
    (
        cudaMallocHost((void **)&(nPixelsInDisc), 
            nPixelInDiscBufferSize)
    );
    std::cerr << "GPUConvolver::allocate_buffers: ";
    std::cerr << "allocating GPU npixels in intra-beam buffer" << std::endl;
    // to store the number of pixels in every disc (GPU)
    CUDA_ERROR_CHECK(
        cudaMalloc((void **)&(nPixelsInDiscGPU), 
            nPixelInDiscBufferSize)
    );
    CUDA_ERROR_CHECK
    (
        cudaMemset(nPixelsInDiscGPU, 0, nPixelInDiscBufferSize)
    );
    std::cerr << "GPUConvolver::allocate_buffers: ";
    std::cerr << "allocating GPU data buffer" << std::endl;
    CUDA_ERROR_CHECK
    (
        cudaMalloc((void **)&(resultGPU), 2 * resultBufferSize)
    );
}

GPUConvolver::~GPUConvolver()
{
    if(hasValidConfig)
    {
        std::cerr << "GPUConvolver::destructor: ";
        std::cerr << "deallocating host data buffer" << std::endl;
        CUDA_ERROR_CHECK(cudaFree(resultGPU));
        
        std::cerr << "GPUConvolver::destructor: ";
        std::cerr << "deallocating host npixels in intra-beam buffer" << std::endl;
        CUDA_ERROR_CHECK(cudaFreeHost(nPixelsInDisc));
        
        std::cerr << "GPUConvolver::destructor: ";
        std::cerr << "deallocating GPU npixels in intra-beam buffer" << std::endl;
        CUDA_ERROR_CHECK(cudaFree(nPixelsInDiscGPU));
        
        std::cerr << "GPUConvolver::destructor: ";
        std::cerr << "deallocating host intra-beam pixel buffer" << std::endl;
        CUDA_ERROR_CHECK(cudaFreeHost(skyPixelsInBeam));
        
        std::cerr << "GPUConvolver::destructor: ";
        std::cerr << "deallocating GPU intra-beam pixel buffer" << std::endl;
        CUDA_ERROR_CHECK(cudaFree(skyPixelsInBeamGPU));
        
        std::cerr << "GPUConvolver::destructor: ";
        std::cerr << "deallocating GPU pointing buffer" << std::endl;
        CUDA_ERROR_CHECK(cudaFree(ptgBufferGPU));
        
        std::cerr << "GPUConvolver::destructor: ";
        std::cerr << "deallocating chi angles buffer" << std::endl;
        CUDA_ERROR_CHECK(cudaFree(chiAnglesGPU));

        std::cerr << "GPUConvolver::destructor: ";
        std::cerr << "deallocating beam matching pixels buffer" << std::endl;
        CUDA_ERROR_CHECK(cudaFree(matchingBeamPixelsGPU));
    }
    if(hasBeam)
    {
        std::cerr << "GPUConvolver::desctructor: ";
        std::cerr << "deallocating GPU beam buffers." << std::endl;
        CUDA_ERROR_CHECK(cudaFree(aBeamsGPU));
        CUDA_ERROR_CHECK(cudaFree(bBeamsGPU));
    }
    if(hasSky)
    {
        std::cerr << "GPUConvolver::desctructor: ";
        std::cerr << "deallocating GPU sky buffer." << std::endl;
        CUDA_ERROR_CHECK(cudaFree(skyGPU));
    }
}

void GPUConvolver::update_sky(Sky* sky)
{
    /* Allocate temporal buffers to hold the transposed sky*/
    std::cerr << "GPUConvolver::update_sky: ";
    std::cerr << "allocating temporary transposed sky buffer.";
    std::cerr << std::endl;
    npixSky = sky->size();
    nsideSky = sky->get_nside();
    
    skyBufferSize = N_SKY_COMP * sizeof(float) * npixSky;
    float* skyTransponsed;
    skyTransponsed = (float*)malloc(skyBufferSize);
    /* Allocate buffers if no sky has been allocated yet.*/
    if(!hasSky)
    {
        std::cerr << "GPUConvolver::update_sky: ";
        std::cerr << "allocating GPU sky buffer.";
        std::cerr << std::endl;
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&skyGPU, skyBufferSize));
    }
    hasSky = true;
    std::cerr << "GPUConvolver::update_sky: ";
    std::cerr << "moving data to transposed sky buffer.";
    std::cerr << std::endl;
    for(long i = 0; i < sky->size(); i++)
    {
        skyTransponsed[4 * i + 0] = sky->sI[i];
        skyTransponsed[4 * i + 1] = sky->sQ[i];
        skyTransponsed[4 * i + 2] = sky->sU[i];
        skyTransponsed[4 * i + 3] = sky->sV[i];
    }
    std::cerr << "GPUConvolver::update_sky: ";
    std::cerr << "copying transposed sky buffer to GPU.";
    std::cerr << std::endl;
    CUDA_ERROR_CHECK(
        cudaMemcpy(
            skyGPU,
            skyTransponsed, 
            skyBufferSize,
            cudaMemcpyHostToDevice)
    );
    std::cerr << "GPUConvolver::update_sky: ";
    std::cerr << "free transposed sky buffer.";
    std::cerr << std::endl;
    free(skyTransponsed);
}

void GPUConvolver::update_beam(PolBeam* beam)
{
    /* Allocate buffers if no beam has been added yet.*/
    npixBeam = beam->size();
    beamRhoMax = beam->get_rho_max();
    nsideBeam = beam->get_nside();
    beamBufferSize = N_POLBEAMS * npixBeam * sizeof(float);
    /* Allocate buffers if no beam has been added yet.*/
    if(!hasBeam)
    {
        std::cerr << "GPUConvolver::update_beam: ";
        std::cerr << "allocating GPU beam buffers.";
        std::cerr << std::endl;
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&aBeamsGPU, beamBufferSize)
        );
        CUDA_ERROR_CHECK(
            cudaMalloc((void **)&bBeamsGPU, beamBufferSize)
        );
    }
    /* Update flag to signal that beam was allocated. */
    hasBeam = true;
    /* Transpose and copy to GPU . */
    std::cerr << "GPUConvolver::update_beam: ";
    std::cerr << "allocating transposed beam buffers.";
    std::cerr << std::endl;
    float* transposedBeams;
    transposedBeams = (float*)malloc(N_POLBEAMS * beamBufferSize);
    std::cerr << "GPUConvolver::update_beam: ";
    std::cerr << "copying data from beam_a to transposed beam buffer.";
    std::cerr << std::endl;
    for(long i = 0; i < npixBeam; i++)
    {
        transposedBeams[N_POLBEAMS * i + 0] = beam->aBeams[0][i];
        transposedBeams[N_POLBEAMS * i + 1] = beam->aBeams[1][i];
        transposedBeams[N_POLBEAMS * i + 2] = beam->aBeams[2][i];
        transposedBeams[N_POLBEAMS * i + 3] = beam->aBeams[3][i];
    }
    std::cerr << "GPUConvolver::update_beam: ";
    std::cerr << "copying data from beam_a from transposed beam buffer ";
    std::cerr << "to GPU beam buffer." << std::endl;
    CUDA_ERROR_CHECK
    (
        cudaMemcpy(
            aBeamsGPU,
            transposedBeams, 
            beamBufferSize,
            cudaMemcpyHostToDevice)
    );
    std::cerr << "GPUConvolver::update_beam: ";
    std::cerr << "copying data from beam_b to transposed beam buffer.";
    std::cerr << std::endl;
    for(long i = 0; i < npixBeam; i++)
    {
        transposedBeams[N_POLBEAMS * i + 0] = beam->bBeams[0][i];
        transposedBeams[N_POLBEAMS * i + 1] = beam->bBeams[1][i];
        transposedBeams[N_POLBEAMS * i + 2] = beam->bBeams[2][i];
        transposedBeams[N_POLBEAMS * i + 3] = beam->bBeams[3][i];
    }
    std::cerr << "GPUConvolver::update_beam: ";
    std::cerr << "copying data from beam_b from transposed beam buffer ";
    std::cerr << "to GPU beam buffer." << std::endl;
    CUDA_ERROR_CHECK
    (
        cudaMemcpy(
            bBeamsGPU,
            transposedBeams, 
            beamBufferSize,
            cudaMemcpyHostToDevice)
    );
    std::cerr << "GPUConvolver::update_beam: ";
    std::cerr << "deallocating transposed beam buffer.";
    std::cerr << std::endl;
    free(transposedBeams);
}

void GPUConvolver::update_scan(Scan* scan)
{
    nsamples = scan->get_nsamples();
    float* rabcptg = scan->get_ra_ptr();
    float* decbcptg = scan->get_dec_ptr();
    float* pabcptg = scan->get_pa_ptr();
    
    if(!hasScan)
    {
        std::cerr << "GPUConvolver::update_scan: ";
        std::cerr << "allocating pinned memory to store scan.";
        std::cerr << std::endl;
        CUDA_ERROR_CHECK(
            cudaMallocHost
            (
                (void **)&(bcptg), 
                nsamples * N_POINTING_COORDS * sizeof(float) 
            )
        );
        hasScan = true;
    }
    std::cerr << "GPUConvolver::update_scan: ";
    std::cerr << "copying data to pinned memory.";
    std::cerr << std::endl;
    for(long i = 0; i < nsamples; i++)
    {
        bcptg[N_POINTING_COORDS * i + 0] = rabcptg[i];
        bcptg[N_POINTING_COORDS * i + 1] = decbcptg[i];
        // Passed arguments are counterclockwise on the sky
        // while CMB requires clockwise arguments.
        bcptg[N_POINTING_COORDS * i + 2] = -pabcptg[i];
        bcptg[N_POINTING_COORDS * i + 3] = 0.0;
    }
}

void GPUConvolver::exec_convolution(float* data_a, float* data_b)
{
    int samp;
    // compute convolution in equal chunks of ptgPerConv samples
    int nsamples2 = nsamples - (nsamples % ptgPerConv);
    samp = 0;
    while(samp < nsamples2)
    {
        /* Locate all sky pixels inside the beam for every pointing 
         * direction in the scan chunk. */
        #pragma omp parallel default(shared)
        // begin parallel region
        {
            int idx;
            float ra;
            float dec;
            Healpix_Base hpxBase(nsideSky, RING, SET_NSIDE);
            
            #pragma omp for schedule(static)
            for(int k = 0; k < ptgPerConv; k++)
            {
                int aidx = samp + k;
                ra = bcptg[N_POINTING_COORDS * aidx + 0];
                dec = bcptg[N_POINTING_COORDS * aidx + 1];
                rangeset<int> intraBeamRanges;
                pointing sc(M_PI_2 - dec, ra);
                hpxBase.query_disc(sc, beamRhoMax, intraBeamRanges);
                // convert ranges to pixel list 
                idx = 0;
                for(int r = 0; r < intraBeamRanges.nranges(); r++) 
                {
                    int ss = intraBeamRanges.ivbegin(r);
                    int ee = intraBeamRanges.ivend(r);
                    for(int ii = ss; ii < ee; ii++) 
                    {
                        int sidx = k * pixPerDisc + idx;
                        skyPixelsInBeam[sidx] = ii;
                        idx++;
                    }
                }
                nPixelsInDisc[k] = idx;
            }
        // end parallel region
        }
        // copy pointing information to GPU
        std::cerr 
            << "copying ptgBuffer to gpu" << std::endl;
        CUDA_ERROR_CHECK
        (
            cudaMemcpy(
                ptgBufferGPU,
                &(bcptg[N_POINTING_COORDS * samp]),
                ptgBufferSize,
                cudaMemcpyHostToDevice)
        );
        // copy intra-beam sky pixel buffer to gpu
        std::cerr
            << " copying nPixelsInDisc to gpu" << std::endl;
        CUDA_ERROR_CHECK
        (
            cudaMemcpy(
                nPixelsInDiscGPU,
                nPixelsInDisc,
                nPixelInDiscBufferSize,
                cudaMemcpyHostToDevice)
        );
        // Copy intra-beam sky pixel buffers to gpu
        std::cerr << " copying skyPixelsInBeam to gpu" << std::endl;
        CUDA_ERROR_CHECK
        (
            cudaMemcpy(
                skyPixelsInBeamGPU,
                skyPixelsInBeam,
                skyPixelsInBeamBufferSize,
                cudaMemcpyHostToDevice)
        );
        // kernel execution
        std::cerr << " executing gpu kernel." << std::endl;

        CUDACONV::launch_fill_pixel_matching_matrix_kernel
        (
            ptgPerConv, ptgBufferGPU,
            nsideSky, nsideBeam,
            pixPerDisc, nPixelsInDiscGPU, skyPixelsInBeamGPU,
            matchingBeamPixelsGPU, chiAnglesGPU
        );
        CUDA_ERROR_CHECK
        (
            cudaMemset(resultGPU, 0, 2 * resultBufferSize)
        );
        CUDACONV::launch_partial_polarized_convolution_kernel
        (
           nsideSky, npixSky, skyGPU,
           nsideBeam, npixBeam, aBeamsGPU, bBeamsGPU,
           ptgPerConv, pixPerDisc, 
           nPixelsInDiscGPU, skyPixelsInBeamGPU,
           matchingBeamPixelsGPU, chiAnglesGPU,
           // output
           &(resultGPU[0]), &(resultGPU[ptgPerConv])
        );

        std::cerr 
            << " copying data back to psb data. size = " << resultBufferSize << " bytes." << std::endl;
        CUDA_ERROR_CHECK
        (
            cudaMemcpy(
                &(data_a[samp]),
                resultGPU,
                resultBufferSize,
                cudaMemcpyDeviceToHost)
        );
        CUDA_ERROR_CHECK
        (
            cudaMemcpy(
                &(data_b[samp]),
                &(resultGPU[ptgPerConv]),
                resultBufferSize,
                cudaMemcpyDeviceToHost)
        );
        samp += ptgPerConv;
    }
    // synchronize to guarantee data is actually in place.
    cudaDeviceSynchronize();
}
