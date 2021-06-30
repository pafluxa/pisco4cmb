/*
 * skymap.hpp 
 * 
 */

// begin include guard
#ifndef _SKYMAPH
#define _SKYMAPH

// include healpix related routines
#include <healpix_base.h>

class SkyMap
{
    const int INSUFFCOV = 0;
    
    public:
        SkyMap(int _nside); 
       ~SkyMap();
        
        void accumulate_data
        (
            int nsamples, float ra[], float dec[], float pa[],
            int ndetectors, float detPolAngles[],
            float data[]
        );
        void solve_map(void);
    
        int get_nside (void) const { return nside; };
        int size (void) const { return nPixels; };
        
        const float* get_stokes_I(void){ return stokesI; };
        const float* get_stokes_Q(void){ return stokesQ; };
        const float* get_stokes_U(void){ return stokesU; };
        const int* get_hitmap(void){ return hitsMap; };
        
    private:
		
		int  nside;
        long nPixels;
        
        double* AtA;
        double* AtD;

		Healpix_Base hpx;

		int* hitsMap;
        float* stokesI;
		float* stokesQ;
		float* stokesU;
        
        void allocate_buffers(void);
        void deallocate_buffers(void);
                
};

// end include guard
#endif 
