/*
 * sky.hpp 
 * 
 */

#ifndef PISCO_SKYH 
#define PISCO_SKYH

#include <healpix_base.h>

class Sky
{
    public:
        Sky(int _nside); 
       ~Sky(void);
        
        int get_nside (void) const;
        int get_npixels (void) const;

        void allocate_buffers(void);
        void free_buffers(void);
        void load_sky_data_from_txt(std::string path);
        void make_point_source_sky(float ra0, float dec0, float I0, float Q0, float U0, float V0);

        const float* get_I(void) const;
        const float* get_Q(void) const;
        const float* get_U(void) const;
        const float* get_V(void) const;

        Healpix_Base hpxBase;
        
    private:
		
        float* I;
        float* Q;
        float* U;
        float* V;
        
        int nside;
        int nPixels;
        size_t skyBufferSize;
        bool buffersOK;
};

#endif 
