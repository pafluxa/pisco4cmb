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
			const float* ra,
			const float* dec,
			const float* pa);
			
       ~Scan(void){};
        
        long get_nsamples (void) const {return nsamp;};
		
		const float* get_ra_ptr(void) const;
		const float* get_dec_ptr(void) const;
		const float* get_pa_ptr(void) const;
	
    private:
		
        const float* ra;
        const float* dec;
        const float* pa;        
        long nsamp;
};

#endif // end include guard
