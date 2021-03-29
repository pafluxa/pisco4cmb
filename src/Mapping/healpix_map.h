#ifndef __HPIX_MAPH__
#define __HPIX_MAPH__

struct healpixMap {
  
    // HEALPix map parameters
    int param_nside;
    int param_npix;
    int param_seen_npix;
    int param_unseen_npix;

    int param_min_seen_pix;
    int param_max_seen_pix;

    // Flag to indicate that map has:
    short flag_has_pixels; // pixel values
    short flag_has_mask;   // mask
    short flag_has_ptg;    // pointing data

    // Buffer to store pixel values only
    float*   buff_pixels;

    // Buffer to store a mask
    short* buff_mask; 

    // Buffer to store pixel-to-pixel jumping for sparse
    // like operations on the map
    int*  buff_idx_to_pix;

    // Buffer to store pixel pointing
    float*  buff_ptg_pix;
};

typedef struct healpixMap healpixMap;

healpixMap*
healpixMap_new( void );

void
healpixMap_allocate( int nside, healpixMap* );

void
healpixMap_free( healpixMap* );

void
healpixMap_set_map( healpixMap*, float* pixels );

void
healpixMap_set_mask( healpixMap*, int*, int );

void
healpixMap_set_unsorted_mask( healpixMap* H, int* seen_pixels, int mask_size );

int
healpixMap_pix2idx( healpixMap *H, int pix );

float
healpixMap_get_pixel_value( healpixMap *H, int pix );

void
healpixMap_generate_pointing ( healpixMap* );


#endif
