/*
 * polbeam.hpp 
 * 
 * Header file to describe a Polarized Beam as described in Rosset et al. 2010.
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
        PolBeam(unsigned int nside, unsigned long nPixels);
       
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
		
        void half_beam_from_fields
        ( 
            char polFlag,
            // Jones vectors
            float* magEco_x, float* phaseEco_x,
            float* magEco_y, float* phaseEco_y,
            float* magEcx_x, float* phaseEcx_x
            float* magEcx_y, float* phaseEcx_y
        );
		
        void build_beams(void);
        
        // this function will go away as it should live on
        // the Python side of things.
        void make_unpol_gaussian_elliptical_beam
        ( 
            double fwhmx, 
			double fwhmy, 
			double phi0 
        );
        
        float* Da_I;
        float* Da_Qcos;
        float* Da_Qsin;
        float* Da_Ucos;
        float* Da_Usin;
        float* Da_V;

        float* Db_I;
        float* Db_Qcos;
        float* Db_Qsin;
        float* Db_Ucos;
        float* Db_Usin;
        float* Db_V;

        float omI;
        float omQ;
        float omU;
        float omV;
        
    private:
        Healpix_Base hpxBase;    

		int  nside;
        long nPixels;
		double rhoMax;
        double epsilon;
        
		float* Ia;
		float* Qa;
		float* Ua;
		float* Va;
    
		float* Ib;
		float* Qb;
		float* Ub;
		float* Vb;
    
        void alloc_buffers( void );
        void free_buffers(void);
};

#endif // end include guard
