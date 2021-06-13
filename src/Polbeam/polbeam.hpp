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
#ifndef _POLBEAMH 
#define _POLBEAMH

#include <complex>
// include healpix related routines
#include <healpix_base.h>

class PolBeam
{
    public:
        PolBeam
        (
            int _nside, long _nPixels, 
            double _epsilon, char _enabledDets
        );

       ~PolBeam();

        int get_nside() const
        {
            return nside;
        };
        
        unsigned long size() const
        {
            return nPixels;
        };
        
        double get_rho_max() const
        {
            return rhoMax;
        };

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

        float* aBeams[4];
        float* bBeams[4];
        int  nside;
        long nPixels;

        Healpix_Base hpxBase;

        char enabledDets;

        float* Ia;
        float* Qa;
        float* Ua;
        float* Va;

        float* Ib;
        float* Qb;
        float* Ub;
        float* Vb;

    private:

        double rhoMax;
        double epsilon;

        void alloc_buffers();
        void free_buffers();
};

#endif // end include guard
