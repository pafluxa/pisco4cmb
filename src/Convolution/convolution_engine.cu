#include "convolution_engine.h"

#include "cu_utils.h"
#include "Sphtrigo/sphtrigo.hpp"

// healpix stuff
#include <arr.h>
#include <pointing.h>
#include <healpix_base.h>
// for OpenMP
#include <omp.h>

#include <iostream>
#include <iomanip>

// Constructor
ConvolutionEngine::ConvolutionEngine(int nsideSky, int nsideBeam, int nSamples) {

    int i;    
    this->nSamples = nSamples;
    this->nsideSky = nsideSky;
    this->nsideBeam = nsideBeam;
    nspix = 12 * nsideSky * nsideSky;
    nbpix = 12 * nsideBeam * nsideBeam;
    nnz = 4 * nspix;
    
    // create cuSparse context
    CATCH_CUSPARSE( cusparseCreate(&cuspH) )

    // create cuBLAS context
    CATCH_CUBLAS(cublasCreate(&cublasH))

    // create streams for transfer
    for(i = 0; i < N_TRANSFER_STREAMS; i++)
    {
        CATCH_CUDA(cudaStreamCreate(&transferStreams[i]))
    }
    // create streams for processing
    for(i = 0; i < N_PROC_STREAMS; i++)
    {
        CATCH_CUDA(cudaStreamCreate(&procStreams[i]))
    }
}

// dummy destructor
ConvolutionEngine::~ConvolutionEngine(void)
{
    int i;

    // clean-up CUBLAS context
    CATCH_CUBLAS(cublasDestroy(cublasH))
    
    // clean-up cuSparse context
    CATCH_CUSPARSE( cusparseDestroy(cuspH))

    // destroy transfer streams
    // create streams for transfer
    for(i = 0; i < N_TRANSFER_STREAMS; i++)
    {
        CATCH_CUDA(cudaStreamDestroy(transferStreams[i]))
    }
    
    for(i = 0; i < N_PROC_STREAMS; i++)
    {
        CATCH_CUDA(cudaStreamDestroy(procStreams[i]))
    }
    
}

void ConvolutionEngine::allocate_host_buffers(void) {
    // matrix buffers are allocated using pinned memory 
    // because they are constantly updated
    size_t bytes;
    // allocate buffers for matrix
    bytes = sizeof(float) * nnz;
    CATCH_CUDA(cudaMallocHost((void**)&csrValR, bytes))
    bytes = sizeof(int) * nnz;
    CATCH_CUDA(cudaMallocHost((void**)&csrColIndR, bytes))
    bytes = sizeof(int) * (nspix + 1);
    CATCH_CUDA(cudaMallocHost((void**)&csrRowPtrR, bytes))
    // allocate bufferse for cosine and sine 2 chi
    bytes = sizeof(float) * nspix;
    CATCH_CUDA(cudaMallocHost((void**)&s2chi, bytes))
    CATCH_CUDA(cudaMallocHost((void**)&c2chi, bytes))
    // sky pixel coordinates
    raPixSky = (float *)malloc(sizeof(float) * nspix);
    decPixSky = (float *)malloc(sizeof(float) * nspix);
}

void ConvolutionEngine::allocate_device_buffers(void) {
    
    float alpha; 
    float beta;
    float addme;
    float subme;
    
    size_t bytes;
    
    // allocate buffers for sky
    bytes = sizeof(float) * nspix;
    CATCH_CUDA(cudaMalloc((void**)&dev_stokesI, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_stokesQ, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_stokesU, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_stokesV, bytes))
    // allocate buffers for beams
    bytes = sizeof(float) * nbpix;
    CATCH_CUDA(cudaMalloc((void**)&dev_beamIa, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_beamQa, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_beamUa, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_beamVa, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_beamIb, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_beamQb, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_beamUb, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_beamVb, bytes))
    // allocate buffers for aligned beams
    bytes = sizeof(float) * nspix;
    CATCH_CUDA(cudaMalloc((void**)&dev_AbeamIa, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_AbeamQa, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_AbeamUa, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_AbeamVa, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_AbeamIb, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_AbeamQb, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_AbeamUb, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_AbeamVb, bytes))
    // allocate buffers for cosine and sine 2 chi
    bytes = sizeof(float) * nspix;
    CATCH_CUDA(cudaMalloc((void**)&dev_s2chi, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_c2chi, bytes))
    // allocate buffers for matrix
    bytes = sizeof(float) * nnz;
    CATCH_CUDA(cudaMalloc((void**)&dev_csrValR, bytes))
    bytes = sizeof(int) * nnz;
    CATCH_CUDA(cudaMalloc((void**)&dev_csrColIndR, bytes))
    bytes = sizeof(int) * (nspix + 1);
    CATCH_CUDA(cudaMalloc((void**)&dev_csrRowPtrR, bytes))
    // allocate buffers for calculation of D
    bytes = sizeof(float) * nspix;
    CATCH_CUDA(cudaMalloc((void**)&dev_DQ1a, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_DQ2a, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_DU1a, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_DU2a, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_DQ1b, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_DQ2b, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_DU1b, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_DU2b, bytes))
    /* detector data */
    bytes = sizeof(float) * nSamples;
    CATCH_CUDA(cudaMalloc((void**)&dev_ia, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_ib, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_qa, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_qb, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_ua, bytes))
    CATCH_CUDA(cudaMalloc((void**)&dev_ub, bytes))
    /* cuda uses device memory for EVERYTHING*/

    alpha = 1.0;
    beta = 0.0;
    addme = 1.0;
    subme = -1.0;

    // cuBLAS operates in device mode, so we need to make explicit copies
    // of constants from the host to the device.    
    CATCH_CUDA( cudaMalloc((void **)&dev_alpha, sizeof(float)) )
    CATCH_CUDA( cudaMalloc((void **)&dev_beta, sizeof(float)) )
    CATCH_CUDA( cudaMalloc((void **)&dev_addme, sizeof(float)) )
    CATCH_CUDA( cudaMalloc((void **)&dev_subme, sizeof(float)) )
    CATCH_CUDA( cudaMemcpy(dev_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice))
    CATCH_CUDA( cudaMemcpy(dev_beta, &beta, sizeof(float), cudaMemcpyHostToDevice))
    CATCH_CUDA( cudaMemcpy(dev_addme, &addme, sizeof(float), cudaMemcpyHostToDevice))
    CATCH_CUDA( cudaMemcpy(dev_subme, &subme, sizeof(float), cudaMemcpyHostToDevice))
}

// creates internal cuSparse CSR matrix. 
void ConvolutionEngine::create_matrix(void) {
    
    CATCH_CUSPARSE(
        cusparseCreateCsr(
            &R, nspix, nbpix, nnz,
            dev_csrRowPtrR, dev_csrColIndR, dev_csrValR,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
    )
}

/* creates internal cuSparse object reprensenting the beam. */
void ConvolutionEngine::beam_to_cuspvec(PolBeam* beam) {
    size_t beamBufferSize; 
    /* size (in bytes) of buffer to be copied. */
    beamBufferSize = sizeof(float) * nbpix;
    /* copy data to device for detector a. */
    CATCH_CUDA(
        cudaMemcpy(dev_beamIa, beam->get_I_beam('a'), beamBufferSize, cudaMemcpyHostToDevice))
    CATCH_CUDA(
        cudaMemcpy(dev_beamQa, beam->get_Q_beam('a'), beamBufferSize, cudaMemcpyHostToDevice))
    CATCH_CUDA(
        cudaMemcpy(dev_beamUa, beam->get_U_beam('a'), beamBufferSize, cudaMemcpyHostToDevice))
    CATCH_CUDA(
        cudaMemcpy(dev_beamVa, beam->get_V_beam('a'), beamBufferSize, cudaMemcpyHostToDevice))
    /* copy data to device for detector b. */
    CATCH_CUDA(
        cudaMemcpy(dev_beamIb, beam->get_I_beam('b'), beamBufferSize, cudaMemcpyHostToDevice))
    CATCH_CUDA(
        cudaMemcpy(dev_beamQb, beam->get_Q_beam('b'), beamBufferSize, cudaMemcpyHostToDevice))
    CATCH_CUDA(
        cudaMemcpy(dev_beamUb, beam->get_U_beam('b'), beamBufferSize, cudaMemcpyHostToDevice))
    CATCH_CUDA(
        cudaMemcpy(dev_beamVb, beam->get_V_beam('b'), beamBufferSize, cudaMemcpyHostToDevice))
    
    CATCH_CUSPARSE(cusparseCreateDnVec(&beamIa, nbpix, dev_beamIa, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&beamQa, nbpix, dev_beamQa, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&beamUa, nbpix, dev_beamUa, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&beamVa, nbpix, dev_beamVa, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&beamIb, nbpix, dev_beamIb, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&beamQb, nbpix, dev_beamQb, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&beamUb, nbpix, dev_beamUb, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&beamVb, nbpix, dev_beamVb, CUDA_R_32F))
    
    CATCH_CUSPARSE(cusparseCreateDnVec(&AbeamIa, nspix, dev_AbeamIa, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&AbeamQa, nspix, dev_AbeamQa, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&AbeamUa, nspix, dev_AbeamUa, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&AbeamVa, nspix, dev_AbeamVa, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&AbeamIb, nspix, dev_AbeamIb, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&AbeamQb, nspix, dev_AbeamQb, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&AbeamUb, nspix, dev_AbeamUb, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&AbeamVb, nspix, dev_AbeamVb, CUDA_R_32F))


}

/* creates internal cuSparse object reprensenting the sky. */
void ConvolutionEngine::sky_to_cuspvec(Sky* sky) {
    size_t bufferSize; 
    /* size (in bytes) of buffer to be copied. */
    bufferSize = sizeof(float) * nspix;
    /* copy data to device for detector a. */
    CATCH_CUDA(
        cudaMemcpy(dev_stokesI, sky->get_I(), bufferSize, cudaMemcpyHostToDevice))
    CATCH_CUDA(
        cudaMemcpy(dev_stokesQ, sky->get_Q(), bufferSize, cudaMemcpyHostToDevice))
    CATCH_CUDA(
        cudaMemcpy(dev_stokesU, sky->get_U(), bufferSize, cudaMemcpyHostToDevice))
    CATCH_CUDA(
        cudaMemcpy(dev_stokesV, sky->get_V(), bufferSize, cudaMemcpyHostToDevice))
    // create dense vector descriptors for cuSparse 
    CATCH_CUSPARSE(cusparseCreateDnVec(&stokesI, nspix, dev_stokesI, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&stokesQ, nspix, dev_stokesQ, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&stokesU, nspix, dev_stokesU, CUDA_R_32F))
    CATCH_CUSPARSE(cusparseCreateDnVec(&stokesV, nspix, dev_stokesV, CUDA_R_32F))
}



void ConvolutionEngine::data_to_host(float *a, float *b)
{
    int i;
    float *ia;
    float *qa;
    float *ua;
    float *ib;
    float *qb;
    float *ub;

    ia = (float *)malloc(sizeof(float) * nSamples);
    qa = (float *)malloc(sizeof(float) * nSamples);
    ua = (float *)malloc(sizeof(float) * nSamples);
    ib = (float *)malloc(sizeof(float) * nSamples);
    qb = (float *)malloc(sizeof(float) * nSamples);
    ub = (float *)malloc(sizeof(float) * nSamples);

    CATCH_CUDA(cudaMemcpy(ia, dev_ia, sizeof(float) * nSamples, cudaMemcpyDeviceToHost))
    CATCH_CUDA(cudaMemcpy(qa, dev_qa, sizeof(float) * nSamples, cudaMemcpyDeviceToHost))
    CATCH_CUDA(cudaMemcpy(ua, dev_ua, sizeof(float) * nSamples, cudaMemcpyDeviceToHost))
    CATCH_CUDA(cudaMemcpy(ib, dev_ib, sizeof(float) * nSamples, cudaMemcpyDeviceToHost))
    CATCH_CUDA(cudaMemcpy(qb, dev_qb, sizeof(float) * nSamples, cudaMemcpyDeviceToHost))
    CATCH_CUDA(cudaMemcpy(ub, dev_ub, sizeof(float) * nSamples, cudaMemcpyDeviceToHost))

    for(i = 0; i < nSamples; i++)
    {
        a[i] = ia[i] + qa[i] + ua[i];
        b[i] = ib[i] + qb[i] + ub[i];
    }

    free(ia);
    free(qa);
    free(ua);
    free(ib);
    free(qb);
    free(ub);
}

void ConvolutionEngine::calculate_sky_pixel_coordinates(Sky* sky) {

    int npixels;
    int skyPix;
    pointing sp;
    
    npixels = 12 * nsideSky * nsideSky;
    for(skyPix = 0; skyPix < npixels; skyPix++)
    {
        sp = sky->hpxBase.pix2ang(skyPix);
        
        raPixSky[skyPix] = float(sp.phi);
        decPixSky[skyPix] = float(M_PI_2 - sp.theta);
    }
}

/* creates internal cuSparse object reprensenting the "rotated" beam. */
void ConvolutionEngine::update_matrix(int pix, float* weights, int* neigh) {

    csrValR[4 * pix + 0] = weights[0];    
    csrValR[4 * pix + 1] = weights[1];    
    csrValR[4 * pix + 2] = weights[2];    
    csrValR[4 * pix + 3] = weights[3];    

    csrColIndR[4 * pix + 0] = neigh[0];
    csrColIndR[4 * pix + 1] = neigh[1];
    csrColIndR[4 * pix + 2] = neigh[2];
    csrColIndR[4 * pix + 3] = neigh[3];

    csrRowPtrR[pix] = 4 * pix;
}

void ConvolutionEngine::update_chi(int pix, float chi) {

    s2chi[pix] = sinf(2.0f * chi);
    c2chi[pix] = cosf(2.0f * chi);
}

void ConvolutionEngine::fill_matrix(Sky* sky, PolBeam* beam, 
    float ra_bc, float dec_bc, float psi_bc) {

    #pragma omp parallel
    {
    int i;
    int idx;
    int off;
    int skyPix;
    int chunksize;
    int reminder;
    double ra_pix;
    double dec_pix;
    double rho;
    double sigma;
    double chi;
    pointing bp;
    fix_arr<int, 4> neigh;
    fix_arr<double, 4> wgh;
    float* weights;
    int* neighbors;
    int* pixels;
    
    idx = omp_get_thread_num();
    off = omp_get_num_threads();
    chunksize = nspix / off;
    // last thread gets more work
    reminder = 0;
    if(idx == omp_get_thread_num() - 1) {
        reminder += (nspix - chunksize * off);
    }
    // allocate local (thread-safe) memory space to store results
    pixels = (int *)malloc(sizeof(int) * (chunksize + reminder));
    neighbors = (int *)malloc(sizeof(int) * 4 * (chunksize + reminder));
    weights = (float *)malloc(sizeof(float) * 4 * (chunksize + reminder));
    
    // Calculate equivalent beam coordinates for every sky pixel. 
    i = 0; 
    while(i < chunksize + reminder)
    {
        skyPix = idx * chunksize + i;        
        // coordinates of sky pixel already computed. 
        ra_pix = raPixSky[skyPix];
        dec_pix = decPixSky[skyPix];

        // get equivalent beam coordinates and polarization angle. 
        SphericalTransformations::rho_sigma_chi_pix(
            &rho, &sigma, &chi,
            double(ra_bc), double(dec_bc), double(psi_bc),
            ra_pix, dec_pix);

        // get interpolation information for beam at (rho, sigma). 
        bp.theta = double(rho);
        bp.phi = double(sigma);
        // handle bug in acos
        if(bp.theta >= M_PI) 
        { 
            bp.theta = double(M_PI); 
        }
        beam->hpxBase.get_interpol(bp, neigh, wgh);

        weights[4*i + 0] = float(wgh[0]);    
        weights[4*i + 1] = float(wgh[1]);    
        weights[4*i + 2] = float(wgh[2]);    
        weights[4*i + 3] = float(wgh[3]);
        neighbors[4*i + 0] = neigh[0];    
        neighbors[4*i + 1] = neigh[1];    
        neighbors[4*i + 2] = neigh[2];    
        neighbors[4*i + 3] = neigh[3];   

        pixels[i] = skyPix;

        i++;
    }

    // every thread performs memcpy in parallel
    memcpy(csrValR + idx * 4 * chunksize, weights, sizeof(float) * (chunksize + reminder) * 4);
    memcpy(csrColIndR + idx * 4 * chunksize, neighbors, sizeof(int) * (chunksize + reminder) * 4);
    memcpy(csrRowPtrR + idx * chunksize, pixels, sizeof(int) * (chunksize + reminder));
    
    free(weights);
    free(neighbors);
    free(pixels);
    /*
    {
        // update matrix values 
        update_matrix(skyPix, weights, neighbors);
        update_chi(skyPix, chi);

        i++;
    }
    */
    }
}

/* copies sine and consine chi and matrix to device.*/
void ConvolutionEngine::exec_transfer(void) {

    size_t bytes;

    // synchronize device to avoid writing while computing
    CATCH_CUDA( cudaDeviceSynchronize() )

    // fill last value of csrRowPtrR
    csrRowPtrR[nspix] = 4 * nspix;

    // transfer cosine and sine of 2 chi
    bytes = sizeof(float) * nspix;
    CATCH_CUDA(cudaMemcpy(dev_s2chi, s2chi, bytes, cudaMemcpyHostToDevice))
    CATCH_CUDA(cudaMemcpy(dev_c2chi, c2chi, bytes, cudaMemcpyHostToDevice))
    
    // transfer matrix descriptors (CSR format)
    bytes = sizeof(float) * nnz;
    CATCH_CUDA(cudaMemcpy(dev_csrValR, csrValR, bytes, cudaMemcpyHostToDevice))
    bytes = sizeof(int) * nnz;
    CATCH_CUDA(cudaMemcpy(dev_csrColIndR, csrColIndR, bytes, cudaMemcpyHostToDevice))
    bytes = sizeof(int) * (nspix + 1);
    CATCH_CUDA(cudaMemcpy(dev_csrRowPtrR, csrRowPtrR, bytes, cudaMemcpyHostToDevice))
}

void ConvolutionEngine::exec_single_convolution_step(int s) {

    size_t bufferSize;

    // set cuSparse to run in device mode so all memory operations
    // happen in the GPU (fully async)
    CATCH_CUSPARSE(cusparseSetPointerMode(cuspH, CUSPARSE_POINTER_MODE_DEVICE))

    // set CUBLAS to run in device mode, i.e., all memory operations happen 
    // in the GPU
    CATCH_CUBLAS( cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE) )

    // calculate required extra space and allocate if needed 
    CATCH_CUSPARSE(
        cusparseSpMV_bufferSize(
            cuspH, CUSPARSE_OPERATION_NON_TRANSPOSE,
            dev_alpha, R, beamIa, dev_beta, AbeamIa, CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, &bufferSize)
    )
    
    if(bufferSize != cusparse_buffersize)
    {
        if(cusparse_buffersize != 0)
        {
            CATCH_CUDA( cudaFree(dev_tempBuffer) )
        }
        cusparse_buffersize = bufferSize;
        CATCH_CUDA( cudaMalloc(&dev_tempBuffer, cusparse_buffersize) )
    }

    // capture graph execution
    if(useGraph)
    {
        CATCH_CUDA( cudaGraphLaunch(graph_exec, procStreams[0]) )
    }
    else
    {
        CATCH_CUSPARSE( cusparseSetStream(cuspH, procStreams[0]) )
        CATCH_CUDA( cudaStreamBeginCapture(procStreams[0], cudaStreamCaptureModeGlobal) )
        // Detector a
        // calculate R x beam for Stokes I beam of detector a
        CATCH_CUSPARSE(
            cusparseSpMV(cuspH,
                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                dev_alpha, R, beamIa, dev_beta, AbeamIa,
                CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2, 
                dev_tempBuffer)
        )

        // Stokes Q beam of detector a ...
        CATCH_CUSPARSE(
            cusparseSpMV(cuspH,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                dev_alpha, R, beamQa, dev_beta, AbeamQa,
                CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2, 
                dev_tempBuffer)
        )
        // Stokes U beam of detector a
        CATCH_CUSPARSE(
            cusparseSpMV(cuspH,
                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                dev_alpha, R, beamUa, dev_beta, AbeamUa,
                CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2, 
                dev_tempBuffer)
        )
        // Repeat for detector b 
        // calculate R x beam for:
        // - Stokes I beam of detector b
        CATCH_CUSPARSE(
            cusparseSpMV(cuspH,
                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                dev_alpha, R, beamIb, dev_beta, AbeamIb,
                CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2, 
                dev_tempBuffer)
        )
        // - Stokes Q beam of detector b ...
        CATCH_CUSPARSE(
            cusparseSpMV(cuspH,
                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                dev_alpha, R, beamQb, dev_beta, AbeamQb,
                CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2, 
                dev_tempBuffer)
        )
        // - Stokes U beam of detector b
        CATCH_CUSPARSE(
            cusparseSpMV(cuspH,
                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                dev_alpha, R, beamUb, dev_beta, AbeamUb,
                CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2, 
                dev_tempBuffer)
        )
        CATCH_CUDA(cudaStreamEndCapture(procStreams[0], &graph))
        CATCH_CUDA(cudaDeviceSynchronize())
        CATCH_CUDA(cudaGetLastError())
        CATCH_CUDA(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0))
        useGraph = true;
    }
    //CATCH_CUDA( cudaStreamSynchronize(procStreams[0]) )

    // calculate D_Q for detector a
    // note: the order in which operations below are executed should not be changed
    //       as it was designed to maximize concurrency with cuSparse
    // - multiply cos(2chi) by Q beam
    cudaUtilities::ewxty(nspix, dev_AbeamQa, dev_c2chi, procStreams[0], dev_DQ1a); 
    // - multiply sin(2chi) by Q beam
    cudaUtilities::ewxty(nspix, dev_AbeamQa, dev_s2chi, procStreams[0], dev_DU2a);
    // - multiply sin(2chi) by U beam
    cudaUtilities::ewxty(nspix, dev_AbeamUa, dev_s2chi, procStreams[0], dev_DQ2a);
    // - multiply cos(2chi) by U beam
    cudaUtilities::ewxty(nspix, dev_AbeamUa, dev_c2chi, procStreams[0], dev_DU1a);
    // - multiply cos(2chi) by Q beam
    cudaUtilities::ewxty(nspix, dev_AbeamQb, dev_c2chi, procStreams[0], dev_DQ1b);
    // - multiply sin(2chi) by Q beam
    cudaUtilities::ewxty(nspix, dev_AbeamQb, dev_s2chi, procStreams[0], dev_DU2b);
    // - multiply sin(2chi) by U beam
    cudaUtilities::ewxty(nspix, dev_AbeamUb, dev_s2chi, procStreams[0], dev_DQ2b);
    // - multiply cos(2chi) by U beam
    cudaUtilities::ewxty(nspix, dev_AbeamUb, dev_c2chi, procStreams[0], dev_DU1b);

    // calculate D for detector a 
    // - add and overwrite DQ1
    CATCH_CUBLAS( cublasSetStream(cublasH, procStreams[0]) )
    CATCH_CUBLAS(cublasSaxpy(cublasH, nspix, dev_subme, dev_DQ2a, 1, dev_DQ1a, 1))
    // - add and overwrite DU1
    CATCH_CUBLAS(cublasSaxpy(cublasH, nspix, dev_addme, dev_DU2a, 1, dev_DU1a, 1))

    // calculate D for detector b 
    // - scale DQ1b by -1.0
    CATCH_CUBLAS(cublasSscal(cublasH, nspix, dev_subme, dev_DQ1b, 1))
    // - add and overwrite DQ1b = -(Q c2chi) + (U s2chi)
    CATCH_CUBLAS(cublasSaxpy(cublasH, nspix, dev_addme, dev_DQ2b, 1, dev_DQ1b, 1))
    // - scale DU1b by -1.0 
    CATCH_CUBLAS(cublasSscal(cublasH, nspix, dev_subme, dev_DU1b, 1))
    // - scale DU2 by -1.0 
    CATCH_CUBLAS(cublasSscal(cublasH, nspix, dev_subme, dev_DU2b, 1))
    // - add and overwrite DU1 = - (U c2chi + Q s2chi)
    CATCH_CUBLAS(cublasSaxpy(cublasH, nspix, dev_addme, dev_DU2b, 1, dev_DU1b, 1))
    
    // convolution is now a dot product
    // - Stokes I part for detector a 
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesI, 1, dev_AbeamIa, 1, dev_ia + s));
    // - Stokes Q part for detector a
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesQ, 1, dev_DQ1a, 1, dev_qa + s));
    // - Stokes U part for detector a
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesU, 1, dev_DU1a, 1, dev_ua + s));
    
    // - Stokes I part for detector b 
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesI, 1, dev_AbeamIb, 1, dev_ib + s));
    // - Stokes Q part for detector a
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesQ, 1, dev_DQ1b, 1, dev_qb + s));
    // - Stokes U part for detector a
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesU, 1, dev_DU1b, 1, dev_ub + s));
}