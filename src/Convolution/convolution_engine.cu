#include "convolution_engine.h"

#include "cu_utils.h"
#include "Sphtrigo/sphtrigo.hpp"

// healpix stuff
#include <arr.h>
#include <pointing.h>
// for OpenMP
#include <omp.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include <iostream>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort


template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  std::stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


// Constructor
ConvolutionEngine::ConvolutionEngine(int _nsideSky, int _nsideBeam, int _nSamples) 
{

    int i;    
    
    nSamples = _nSamples;
    nsideSky = _nsideSky;
    nsideBeam = _nsideBeam;
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

// constructor that can handle partial beam coverage
// rho is the beam extension, in radians
ConvolutionEngine::ConvolutionEngine(int _nsideSky, int _nsideBeam, int _nSamples, double rho) 
: hpxSky(_nsideSky, RING, SET_NSIDE) {

    int i;    
    pointing equator;
    rangeset<int> intraBeamPixels;

    nSamples = _nSamples;
    nsideSky = _nsideSky;
    nsideBeam = _nsideBeam;
    nspix = 12 * nsideSky * nsideSky;
    nbpix = 12 * nsideBeam * nsideBeam;

    // the number of non-zero entries in the matrix is set as a function
    // of how large the beam is, i.e., larger beams require calculating
    // stuff for more sky pixels than small beams.
    // as a worst scase scenario, we consider all the pixels around the 
    // equator with a headroom of 10%
    equator.phi = 0.0;
    equator.theta = M_PI_2;
    intraBeamPixels = hpxSky.query_disc(equator, rho * 1.02);
    nzpixels = intraBeamPixels.nval();
    nnz = 4 * nzpixels;
    std::cerr << "Maximum number of sky pixels within beam = " << nzpixels << std::endl;
    std::cerr << "Maximum number of non-zero elements in matrix = " << 4 * nzpixels << std::endl; 
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


void ConvolutionEngine::sync(void) 
{
    CATCH_CUDA(cudaDeviceSynchronize())
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
    raPixSky = (double *)malloc(sizeof(double) * nspix);
    decPixSky = (double *)malloc(sizeof(double) * nspix);
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
    bytes = sizeof(int) * (nnz);
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
    CATCH_CUDA(cudaMallocManaged(&dev_ia, bytes))
    CATCH_CUDA(cudaMallocManaged(&dev_ib, bytes))
    CATCH_CUDA(cudaMallocManaged(&dev_qa, bytes))
    CATCH_CUDA(cudaMallocManaged(&dev_qb, bytes))
    CATCH_CUDA(cudaMallocManaged(&dev_ua, bytes))
    CATCH_CUDA(cudaMallocManaged(&dev_ub, bytes))
    /* uses device memory for EVERYTHING in cuSparse and cuBLAS*/
    alpha = 1.0;
    beta = 0.0;
    addme = 1.0;
    subme = -1.0;
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
void ConvolutionEngine::create_matrix(void) 
{      
    CATCH_CUSPARSE(
        cusparseCreateCoo(&R, 
            nspix, nbpix, nnz,
            dev_csrRowPtrR, dev_csrColIndR, dev_csrValR,
            CUSPARSE_INDEX_32I,
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

void ConvolutionEngine::iqu_to_tod(float *a, float *b)
{
    int i;
    for(i = 0; i < nSamples; i++)
    {
        a[i] = dev_ia[i] + dev_qa[i] + dev_ua[i];
        b[i] = dev_ib[i] + dev_qb[i] + dev_ub[i];
    }
}

void ConvolutionEngine::calculate_sky_pixel_coordinates(Sky* sky) {

    int skyPix;
    pointing sp;
    
    for(skyPix = 0; skyPix < nspix; skyPix++)
    {
        sp = sky->hpxBase.pix2ang(skyPix);
        
        raPixSky[skyPix] = sp.phi;
        decPixSky[skyPix] = M_PI_2 - sp.theta;
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

    pointing bc;
    rangeset<int> intraBeamSkyPixels_rg;
    std::vector<int> intraBeamSkyPixels;
    // build beam centern pointing
    bc.phi = double(ra_bc);
    bc.theta = M_PI_2 - double(dec_bc);
    intraBeamSkyPixels_rg = hpxSky.query_disc(bc, beam->get_rho_max());
    // sky pixels inside the beam are sorted in ascending order, nice!
    intraBeamSkyPixels = intraBeamSkyPixels_rg.toVector();
    
    #pragma omp parallel shared(intraBeamSkyPixels)
    {
    int i;
    int m;
    int loc;
    int idx;
    int skyPix;
    int chunksize;
    int reminder;

    double rho;
    double sigma;
    double chi;
    pointing bp;
    fix_arr<int, 4> neigh;
    fix_arr<double, 4> wgh;


    chunksize = intraBeamSkyPixels.size() / omp_get_num_threads();
    reminder = 0;
    if(omp_get_thread_num() == omp_get_num_threads() - 1)
    {
        reminder += (intraBeamSkyPixels.size() - chunksize * omp_get_num_threads());
    } 

    // Calculate equivalent beam coordinates for every sky pixel inside the beam.
    i = 0;
    idx = omp_get_thread_num();
    while(i < chunksize + reminder)
    {   
        loc = idx * chunksize + i;
        skyPix = intraBeamSkyPixels[loc];
        // get equivalent beam coordinates and polarization angle. 
        SphericalTransformations::rho_sigma_chi_pix(
            &rho, &sigma, &chi,
            double(ra_bc), double(dec_bc), double(psi_bc),
            raPixSky[skyPix], decPixSky[skyPix]);

        // assign to healpix pointig type
        bp.theta = rho;
        bp.phi = sigma;

        // get interpolation information for beam at (rho, sigma). 
        beam->hpxBase.get_interpol(bp, neigh, wgh);

        // as vector, for argsort
        std::vector<int> neighv;
        for(int j = 0; j < 4; j++)
        {
            neighv.push_back(neigh[j]);
        }
        // store sorted!!
        auto ascidx = sort_indexes(neighv);
        for(int k = 0; k < 4; k++) 
        { 
            csrValR[4 * loc + k] = float(wgh[ascidx[k]]);    
            csrColIndR[4 * loc + k] = neigh[ascidx[k]];
            csrRowPtrR[4 * loc + k] = skyPix;
        }

        // chi angles
        s2chi[skyPix] = sinf(2.0f * float(chi));
        c2chi[skyPix] = cosf(2.0f * float(chi));
        
        i++;
    }
    // handle pixels what would still fit into buffer but are outside the beam
    // this usually doesn't for too much extra computation so it is handled
    // by a single thread
    m = 0;
    if(omp_get_thread_num() == omp_get_num_threads() - 1)
    {
        while(m < nzpixels - intraBeamSkyPixels.size())
        {
            skyPix++;
            loc++; 
            // get equivalent beam coordinates and polarization angle. 
            SphericalTransformations::rho_sigma_chi_pix(
                &rho, &sigma, &chi,
                double(ra_bc), double(dec_bc), double(psi_bc),
                raPixSky[skyPix], decPixSky[skyPix]);

            // assign to healpix pointig type
            bp.theta = rho;
            bp.phi = sigma;

            // get interpolation information for beam at (rho, sigma). 
            beam->hpxBase.get_interpol(bp, neigh, wgh);

            // as vector, for argsort
            std::vector<int> neighv;
            for(int j = 0; j < 4; j++)
            {
                neighv.push_back(neigh[j]);
            }
            // store sorted!!
            auto ascidx = sort_indexes(neighv);
            for(int k = 0; k < 4; k++) 
            { 
                csrValR[4 * loc + k] = float(wgh[ascidx[k]]);    
                csrColIndR[4 * loc + k] = neigh[ascidx[k]];
                csrRowPtrR[4 * loc + k] = skyPix;
            }

            // chi angles
            s2chi[skyPix] = sinf(2.0f * float(chi));
            c2chi[skyPix] = cosf(2.0f * float(chi));
            
            m++;
        }
    }

    } // end parallel region
}

/* copies sine and consine chi and matrix to device.*/
void ConvolutionEngine::exec_transfer(void) {

    size_t bytes;

    // transfer cosine and sine of 2 chi
    bytes = sizeof(float) * nspix;
    CATCH_CUDA(cudaMemcpyAsync(dev_s2chi, s2chi, bytes, cudaMemcpyHostToDevice, transferStreams[0]))
    CATCH_CUDA(cudaMemcpyAsync(dev_c2chi, c2chi, bytes, cudaMemcpyHostToDevice, transferStreams[1]))
    
    // transfer matrix descriptors (CSR format)
    bytes = sizeof(float) * nnz;
    CATCH_CUDA(cudaMemcpyAsync(dev_csrValR, csrValR, bytes, cudaMemcpyHostToDevice, transferStreams[0]))
    bytes = sizeof(int) * nnz;
    CATCH_CUDA(cudaMemcpyAsync(dev_csrColIndR, csrColIndR, bytes, cudaMemcpyHostToDevice, transferStreams[1]))
    bytes = sizeof(int) * (nnz);
    CATCH_CUDA(cudaMemcpyAsync(dev_csrRowPtrR, csrRowPtrR, bytes, cudaMemcpyHostToDevice, transferStreams[0]))

    CATCH_CUDA(cudaDeviceSynchronize())
}

void ConvolutionEngine::exec_single_convolution_step(int s) {

    size_t bufferSize1;
    size_t bufferSize2;
    size_t bufferSize3;
    size_t bufferSize4;
    size_t bufferSize5;
    size_t bufferSize6;

    // calculate required extra space and allocate if needed when running the first time
    if(firstRun)
    {
        // set cuSparse to run in device mode so all memory operations
        // happen in the GPU (fully async)
        CATCH_CUSPARSE(cusparseSetPointerMode(cuspH, CUSPARSE_POINTER_MODE_DEVICE))

        // set CUBLAS to run in device mode, i.e., all memory operations happen 
        // in the GPU
        CATCH_CUBLAS( cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE) )
        // set cuSparse to use the same stream as cuBlas
        CATCH_CUSPARSE( cusparseSetStream(cuspH, procStreams[0]) )
        // set cuBLAS to use the stream as cuSparse
        CATCH_CUBLAS( cublasSetStream(cublasH, procStreams[0]) )
        CATCH_CUSPARSE(
            cusparseSpMV_bufferSize(
                cuspH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &dev_alpha, R, beamIa, dev_beta, AbeamIa, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize1)
        )
        CATCH_CUSPARSE(
            cusparseSpMV_bufferSize(
                cuspH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                dev_alpha, R, beamQa, dev_beta, AbeamQa, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize2)
        )
        CATCH_CUSPARSE(
            cusparseSpMV_bufferSize(
                cuspH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                dev_alpha, R, beamUa, dev_beta, AbeamUa, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize3)
        )
        CATCH_CUSPARSE(
            cusparseSpMV_bufferSize(
                cuspH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                dev_alpha, R, beamIb, dev_beta, AbeamIb, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize4)
        )
        CATCH_CUSPARSE(
            cusparseSpMV_bufferSize(
                cuspH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                dev_alpha, R, beamQa, dev_beta, AbeamUb, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize5)
        )
        CATCH_CUSPARSE(
            cusparseSpMV_bufferSize(
                cuspH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                dev_alpha, R, beamQa, dev_beta, AbeamUb, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize6)
        )

        CATCH_CUDA( cudaMalloc(&dev_tempBuffer1, bufferSize1) )
        CATCH_CUDA( cudaMalloc(&dev_tempBuffer2, bufferSize2) )
        CATCH_CUDA( cudaMalloc(&dev_tempBuffer3, bufferSize3) )
        CATCH_CUDA( cudaMalloc(&dev_tempBuffer4, bufferSize4) )
        CATCH_CUDA( cudaMalloc(&dev_tempBuffer5, bufferSize5) )
        CATCH_CUDA( cudaMalloc(&dev_tempBuffer6, bufferSize6) )
        
        CATCH_CUDA(cudaDeviceSynchronize())
        CATCH_CUDA(cudaGetLastError())

        firstRun = false;
    }
    

    // Detector a
    // calculate R x beam for Stokes I beam of detector a
    CATCH_CUSPARSE(
        cusparseSpMV(cuspH,
            CUSPARSE_OPERATION_NON_TRANSPOSE, 
            dev_alpha, R, beamIa, dev_beta, AbeamIa,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, 
            dev_tempBuffer1)
    )

    // Stokes Q beam of detector a ...
    CATCH_CUSPARSE(
        cusparseSpMV(cuspH,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            dev_alpha, R, beamQa, dev_beta, AbeamQa,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, 
            dev_tempBuffer2)
    )
    // Stokes U beam of detector a
    CATCH_CUSPARSE(
        cusparseSpMV(cuspH,
            CUSPARSE_OPERATION_NON_TRANSPOSE, 
            dev_alpha, R, beamUa, dev_beta, AbeamUa,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, 
            dev_tempBuffer3)
    )
    
    // Repeat for detector b 
    // calculate R x beam for:
    // - Stokes I beam of detector b
    CATCH_CUSPARSE(
        cusparseSpMV(cuspH,
            CUSPARSE_OPERATION_NON_TRANSPOSE, 
            dev_alpha, R, beamIb, dev_beta, AbeamIb,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, 
            dev_tempBuffer4)
    )
    // - Stokes Q beam of detector b ...
    CATCH_CUSPARSE(
        cusparseSpMV(cuspH,
            CUSPARSE_OPERATION_NON_TRANSPOSE, 
            dev_alpha, R, beamQb, dev_beta, AbeamQb,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, 
            dev_tempBuffer5)
    )
    // - Stokes U beam of detector b
    CATCH_CUSPARSE(
        cusparseSpMV(cuspH,
            CUSPARSE_OPERATION_NON_TRANSPOSE, 
            dev_alpha, R, beamUb, dev_beta, AbeamUb,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, 
            dev_tempBuffer6)
    )

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
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesI, 1, dev_AbeamIa, 1, dev_ia + s))
    // - Stokes Q part for detector a
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesQ, 1, dev_DQ1a, 1, dev_qa + s))
    // - Stokes U part for detector a
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesU, 1, dev_DU1a, 1, dev_ua + s))

    // - Stokes I part for detector b 
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesI, 1, dev_AbeamIb, 1, dev_ib + s))
    // - Stokes Q part for detector a
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesQ, 1, dev_DQ1b, 1, dev_qb + s))
    // - Stokes U part for detector a
    CATCH_CUBLAS(cublasSdot(cublasH, nspix, dev_stokesU, 1, dev_DU1b, 1, dev_ub + s))
    
} 