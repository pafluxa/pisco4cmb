/*
 * sky.hpp 
 * 
 */

#ifndef PISCO_SKYH 
#define PISCO_SKYH

#define NPOLSKY

class Sky
{
    public:
        Sky(int _nside); 
       ~Sky(void);
        
        int get_nside (void) const;
        int get_npixels (void) const;
        
        void load_sky_data_from_txt(std::string path);
        
        const float* get_buffer() const;
        
        #ifdef USE_CUDA
        void set_gpu_device(int devideId);
        #endif
        
    private:
		
        float* skyData;
        #ifdef USE_CUDA
        float* cuda_skyData;
        void transfer_to_gpu();
        #endif
		int nside;
        int nPixels;
        size_t skyBufferSize;
        
        void alloc_buffers(void);
        void free_buffers(void);
};

#endif 
