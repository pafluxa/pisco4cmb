/*
 * scan.hpp 
 * 
 */

#ifndef __SCANNINGH__ // begin include guard
#define __SCANNINGH__

class Scanning
{
    public:

        Scanning();	
       ~Scanning();

        long size() const { return nsamp; };

        void allocate_buffers();
        void free_buffers();

		const float* get_ra_ptr(void);
		const float* get_dec_ptr(void);
		const float* get_pa_ptr(void);
        
        void make_raster_scan(
            int nra, int ndec, int npa, 
            float ra0, float dra, 
            float dec0, float ddec, 
            float pa0, float dpa);

        void make_simple_full_sky_scan(int nside, int npa, float pa0, float dpa);

    private:

		long nsamp;
        float* ra;
        float* dec;
        float* pa;        
};

#endif // end include guard
