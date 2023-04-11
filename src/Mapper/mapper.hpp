/*
 * mapper.hpp 
 * 
 */

// begin include guard
#ifndef __MAPPERH__
#define __MAPPERH__

// include healpix related routines
#include <healpix_base.h>

class Mapper
{
    const int INSUFFCOV = -1;
    const int ILLCOND = -100;
    public:
        Mapper(int _nside); 
       ~Mapper();
        
        void accumulate_data
        (
            int nsamples, const float *ra, const float *dec, const float *pa, 
            float detPolAngle,
            float *detdata
        );

        void solve_map(void);
    
        int get_nside (void) const { return nside; };

        int size (void) const { return nPixels; };
        
        const int* get_hitmap(void){ return hitsMap; };
        const float* get_stokes_I(void){ return stokesI; };
        const float* get_stokes_Q(void){ return stokesQ; };
        const float* get_stokes_U(void){ return stokesU; };
        
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
