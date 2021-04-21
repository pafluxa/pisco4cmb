/*
 * polbeam.hpp
 *
 * Header file implementing a Polarized Beam as described
 * in Rosset et al. 2010.
 *
 */

#ifndef _POLBEAMH // begin include guard
#define _POLBEAMH

#include <complex>
#include <array>

// include healpix related routines
#include <healpix_base.h>

class PolBeam
{
    public:
        PolBeam(int nside, long nPixels);

       ~PolBeam();

        int get_nside(void) const
        {
            return nside;
        };
        unsigned long size(void) const
        {
            return nPixels;
        };
        double get_rho_max(void) const
        {
            return rhoMax;
        };

        void beam_from_fields
        (
            char polFlag,
            // Jones vectors
            double* magEco, double* phsEco,
            double* magEcx, double* phsEcx
        );

        void build_beams(void);

        // this function will go away as it should live on
        // the Python side of things.
        void make_unpol_gaussian_elliptical_beams
        (
            double fwhmx,
            double fwhmy,
            double phi0
        );

        float* aBeams[6];
        float* bBeams[6];
        int  nside;
        long nPixels;

        Healpix_Base* hpxBase;

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

        void alloc_buffers( void );
        void free_buffers(void);
};

#endif // end include guard
