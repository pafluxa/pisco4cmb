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
#include <healpix_base.h>

class PolBeam
{
    public:
        
        /* constructor */
        PolBeam(int nside);
        /* constructor for the future */
        /*
        PolBeam
        (
            int nside, long nPixels, 
            double epsilon, char enabledDets
        );
        */
        /* destructor */
       ~PolBeam();
        
        /* returns nside parameter. */
        int get_nside() const;
        /* returns number of pixels in the beam. */
        int get_npixels() const;
        /* return 'a' or 'b' if only one detector is on, 'p' for both on. */
        char get_psb_mode() const;
        /* get beam extension on the sky. */
        double get_rho_max() const;
        /* returns true if buffers are allocated. */
        bool buffers_allocated() const;
        /* compensate for pixel size of the sky, apply pol. efficiency and normalize.*/
        void normalize(int nsideSky);
        /* creates (un-normalized) elliptical gaussian beam for specified detector. */
        void make_unpol_gaussian_elliptical_beam
        (
            char detector,
            double fwhmx,
            double fwhmy,
            double phi0
        );
        /* loads beam data from text file for specified detector. */
        void load_beam_data_from_txt(char det, std::string path);
        /* return I, Q, U and V for specified detector (a or b) */
        const float* get_I_beam(char det) const;
        const float* get_Q_beam(char det) const;
        const float* get_U_beam(char det) const;
        const float* get_V_beam(char det) const;
        /* allocate buffers to store beam data. */
        void allocate_buffers();
        /* free beam buffers. */
        void free_buffers();
        
        Healpix_Base hpxBase;
        
    private:
        /* indicates if buffers are allocated. */
        bool buffersOK;
        /* size (in bytes) of a single beam. */
        size_t beamBufferSize;
        /* beams for detector 'a'. */
        float* Ia;
        float* Qa;
        float* Ua;
        float* Va;
        /* beams for detector 'b'. */
        float* Ib;
        float* Qb;
        float* Ub;
        float* Vb;
        /* beam extension on the sky. */
        double rhoMax;
        /* polarization efficiency. */
        double epsilon;
        /* PSB mode (a, b, or p). */
        char psbmode;
        
        int nside;
        int nPixels;
};

#endif // end include guard
