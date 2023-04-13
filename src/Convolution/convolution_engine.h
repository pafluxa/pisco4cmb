#ifndef __CONVOLUTIONENGINE__
#define __CONVOLUTIONENGINE__

#include <cublas_v2.h>
#include <cusparse.h>

#include "Sky/sky.hpp"
#include "Polbeam/polbeam.hpp"

#define N_TRANSFER_STREAMS 2
#define N_PROC_STREAMS 5

// catch CUDA errors
#define CATCH_CUDA(stat) { catch_cuda((stat), __FILE__, __LINE__); }
inline void catch_cuda(cudaError_t status, const char *file, int line, bool abort=true) {    
    if(status != cudaSuccess) {
        fprintf(stderr, 
            "[ERROR] catch_cuda reports error %s at %s, line %d\n", 
            cudaGetErrorString(status), file, line);
        if (abort) { exit(status); }
    }                                                                       
}

// catch CUSPARSE errors
#define CATCH_CUSPARSE(stat) { catch_cusparse((stat), __FILE__, __LINE__); }
inline void catch_cusparse(cusparseStatus_t status, const char *file, int line, bool abort=true) {    
    if(status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, 
            "[ERROR] cusparse_catch reports error %s at %s, line %d\n", 
            cusparseGetErrorString(status), file, line);
        if (abort) { exit(status); }
    }                                                                       
}

// catch CUBLAS errors
#define CATCH_CUBLAS(stat) { catch_cublas((stat), __FILE__, __LINE__); }
inline void catch_cublas(cublasStatus_t status, const char *file, int line, bool abort=true) {    
    if(status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, 
            "[ERROR] catch_cublas reports error %s at %s, line %d\n", 
            cublasGetStatusString(status), file, line);
        if (abort) { exit(status); }
    }                                                                       
}

class ConvolutionEngine
{

    public:
    
        ConvolutionEngine(int nsideSky, int nsideBeam, int nSamples);
       ~ConvolutionEngine();
        /* Allocate host memory. */
        void allocate_host_buffers(void);
        /* Free host memory. */
        void free_host_buffers(void);
        /* Allocate device memory. */
        void allocate_device_buffers(void);
        /* Free device memory. */
        void free_device_buffers(void);
        /* Computes and stores coordinates (RA, dec) of all pixels in the sky.*/
        void calculate_sky_pixel_coordinates(Sky* sky);
        /* Creates internal representation of the beam a dense vector. */
        void beam_to_cuspvec(PolBeam* beam);
        /* Creates internal representation of the sky as a dense vectors. */
        void sky_to_cuspvec(Sky* sky);
        /* Creates internal representation of rotation matrix as a sparse matrix in CSR format.*/
        void create_matrix(void);
        /* transfers and executes computation*/
        void exec_transfer(void);
        /* fill matrix*/
        void fill_matrix(Sky* sky, PolBeam* beam, float ra_bc, float dec_bc, float psi_bc);
        /* execute a "sky beam multiplication" */
        void exec_single_convolution_step(int s);
        /* detector data back to host*/
        void iqu_to_tod(float* a, float *b);
        /* synchronize device */
        void sync(void);
        
    private:
        /** Parameters. **/
        /* NSIDE parameter of sky. */
        int nsideSky;
        /* NSIDE parameter of the beam. */
        int nsideBeam;
        /* number of samples*/
        int nSamples;
        /* true if beam has been attached. */
        bool hasBeam;
        /* true if sky has been attached. */
        bool hasSky;       
        /* true if matrix was created.*/
        bool hasMatrix;
        /* number of pixels in sky*/
        int nspix;
        /* number of pixels in beam*/
        int nbpix;
        /* number of non-zero elements in matrix*/
        int nnz;
        /* flag to check if this is the first run*/
        bool firstRun = true;
        /* status, for reporting */
        cusparseStatus_t status;
        /* cuSparse handler*/
        cusparseHandle_t cuspH = NULL;
        /* cuBLAS handler*/
        cublasHandle_t cublasH = NULL;
        /* transfer streams*/
        cudaStream_t transferStreams[N_TRANSFER_STREAMS];
        cudaStream_t procStreams[N_PROC_STREAMS];

        // temporal buffer for cusparse/
        void* dev_tempBuffer1 = NULL;
        void* dev_tempBuffer2 = NULL;
        void* dev_tempBuffer3 = NULL;
        void* dev_tempBuffer4 = NULL;
        void* dev_tempBuffer5 = NULL;
        void* dev_tempBuffer6 = NULL;
        /* buffers to store detector data*/
        float* dev_ia;
        float* dev_qa;
        float* dev_ua;
        float* dev_ib;
        float* dev_qb;
        float* dev_ub;
        /* Matrix that rotates the beam into sky basis*/
        cusparseSpMatDescr_t R;
        /* Sky (Stokes I, Q, U and V) as dense vectors.*/
        float* dev_stokesI;
        float* dev_stokesQ;
        float* dev_stokesU;
        float* dev_stokesV;
        cusparseDnVecDescr_t stokesI;
        cusparseDnVecDescr_t stokesQ;
        cusparseDnVecDescr_t stokesU;
        cusparseDnVecDescr_t stokesV;
        /* for Graph Capturing */
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;
        bool useGraph = false;
        /* device variables for cuSparse*/
        float* dev_alpha;
        float* dev_beta;
        float* dev_addme;
        float* dev_subme;
        /* Beam as dense vectors too.*/
        float* dev_beamIa;
        float* dev_beamQa;
        float* dev_beamUa;
        float* dev_beamVa;
        cusparseDnVecDescr_t beamIa;
        cusparseDnVecDescr_t beamQa;
        cusparseDnVecDescr_t beamUa;
        cusparseDnVecDescr_t beamVa;
        float* dev_beamIb;
        float* dev_beamQb;
        float* dev_beamUb;
        float* dev_beamVb;
        cusparseDnVecDescr_t beamIb;
        cusparseDnVecDescr_t beamQb;
        cusparseDnVecDescr_t beamUb;
        cusparseDnVecDescr_t beamVb;
        /* and their "aligned" counterparts too . */
        float* dev_AbeamIa;
        float* dev_AbeamQa;
        float* dev_AbeamUa;
        float* dev_AbeamVa;
        cusparseDnVecDescr_t AbeamIa;
        cusparseDnVecDescr_t AbeamQa;
        cusparseDnVecDescr_t AbeamUa;
        cusparseDnVecDescr_t AbeamVa;
        float* dev_AbeamIb;
        float* dev_AbeamQb;
        float* dev_AbeamUb;
        float* dev_AbeamVb;
        cusparseDnVecDescr_t AbeamIb;
        cusparseDnVecDescr_t AbeamQb;
        cusparseDnVecDescr_t AbeamUb;
        cusparseDnVecDescr_t AbeamVb;
        /* buffers to store D */
        float* dev_DQ1a;
        float* dev_DQ2a;
        float* dev_DU1a;
        float* dev_DU2a;
        float* dev_DQ1b;
        float* dev_DQ2b;
        float* dev_DU1b;
        float* dev_DU2b;
        /* coordinates of sky pixels (right ascention and declination)*/
        double* raPixSky;
        double* decPixSky;
        /* Sine and cosine of twice chi (host). */
        float* s2chi;
        float* c2chi;
        /* Sine and cosine of twice chi (device). */
        float* dev_s2chi;
        float* dev_c2chi;
        /* Transformation matrix (host) in CSR format. */
        float* csrValR;
        int* csrRowPtrR;
        int* csrColIndR;
        /* Transformation matrix (device) . */
        float* dev_csrValR;
        int* dev_csrRowPtrR;
        int* dev_csrColIndR;

        /* Updates rotation matrix (transfers from host to device)*/
        void update_matrix(int pix, float* weights, int* neigh);
        /* Update buffers for cosine and sine of 2 chi.*/
        void update_chi(int pix, float chi);
};

#endif