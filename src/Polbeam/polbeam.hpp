/*
 * polbeam.hpp
 * 
 * PolBeam encapsulates methods and buffers to store a polarized 
 * representation of a telescope beam as described in Rosset et al 2010
 * with some modifications that optimize the computation of the 
 * convolution between the beam and the sky. Internally, PolBeam stores
 * the invidual components as partial Healpix maps. All beams consider 
 * the North pole as the beam center. Maps are stored up to a given 
 * distance from beam center (controlled by the rhoMax parameter). 
 * 
 *
 */

// begin include guard
#ifndef PISCO_POLBEAMH 
#define PISCO_POLBEAMH

#include <complex>

/* number of components in the polarized beams. */
#define NPOLBEAMS (4)

class PolBeam
{
    public:
        PolBeam
        (
            int nside, long nPixels, 
            double epsilon, char enabledDets
        );

       ~PolBeam();

        int get_nside() const;
        int get_npixels() const;
        int get_enabled_detectors() const;
        double get_rho_max() const;
        #ifdef USE_CUDA
        void set_gpu_device(int devideId);
        #endif
        
        void beam_from_fields
        (
            char polFlag,
            double* magEco, double* phsEco,
            double* magEcx, double* phsEcx
        );

        void build(int nsideSky);

        // this function will go away as it should live on
        // the Python side of things.
        void make_unpol_gaussian_elliptical_beams
        (
            double fwhmx,
            double fwhmy,
            double phi0
        );
        
        void load_beam_data_from_txt(std::string path);

        const float* get_beam_a() const;
        const float* get_beam_b() const;

        #ifdef USE_CUDA
        void transfer_to_gpu();
        #endif
        
    private:
        
        bool buffersOK;
        size_t beamBufferSize;
        
        float* Ia;
        float* Qa;
        float* Ua;
        float* Va;

        float* Ib;
        float* Qb;
        float* Ub;
        float* Vb;

        float* aBeams;
        float* bBeams;
        #ifdef USE_CUDA
        float* cuda_aBeams;
        float* cuda_bBeams;
        #endif
        
        double rhoMax;
        double epsilon;

        void alloc_buffers();
        void free_buffers();
};

#endif // end include guard
