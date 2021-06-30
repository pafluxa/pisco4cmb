/*
 * scan.hpp 
 * 
 */

#ifndef _SCANH // begin include guard
#define _SCANH

class Scan
{
    public:

        Scan(long nsamples);
        Scan( 
			long nsamples,
			float* ra,
			float* dec,
			float* pa);
			
       ~Scan(void){};
        
        long get_nsamples (void) const {return nsamp;};
		
		float* get_ra_ptr(void);
		float* get_dec_ptr(void);
		float* get_pa_ptr(void);
	
    private:
		
        float* ra;
        float* dec;
        float* pa;        
        long nsamp;
};

#endif // end include guard
