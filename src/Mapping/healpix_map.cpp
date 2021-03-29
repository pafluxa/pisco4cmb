#include <chealpix.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

#include "Mapping/healpix_map.h"

healpixMap*
healpixMap_new( void )
{

    healpixMap* H;
    H = (healpixMap*)malloc( sizeof(healpixMap) * 1 );

    return H;
}

void
healpixMap_allocate( int nside, healpixMap* H )
{

    H->param_nside = nside;
    H->param_npix  = nside2npix( nside ); 

    H->flag_has_pixels = 0;
    H->flag_has_mask   = 0;
    H->flag_has_ptg    = 0;
}

void
healpixMap_free( healpixMap* H )
{

    if( H->flag_has_mask )
        free( H->buff_idx_to_pix );

    free( H );
}

void 
healpixMap_set_map( healpixMap* H, float *pixels )
{
    assert( "pixel buffer is a null pointer." && pixels != NULL );   
    
    H->flag_has_pixels = 1;
    H->buff_pixels = pixels;
}

void
healpixMap_set_mask( healpixMap* H, int* seen_pixels, int mask_size )
{
    assert( "mask is a null pointer." && seen_pixels != NULL );
    
    H->param_seen_npix   = mask_size;
    H->param_unseen_npix = H->param_npix - H->param_seen_npix;

    // Assign to the index-to-pixel array
    H->buff_idx_to_pix    = seen_pixels;
    H->param_min_seen_pix = seen_pixels[0];
    H->param_max_seen_pix = seen_pixels[ mask_size - 1 ];
    
    H->flag_has_mask = 1;
}

void
healpixMap_set_unsorted_mask( healpixMap* H, int* seen_pixels, int mask_size )
{
    assert( "mask is a null pointer." && seen_pixels != NULL );
    
    H->param_seen_npix   = mask_size;
    H->param_unseen_npix = H->param_npix - H->param_seen_npix;
    H->flag_has_mask = 1;

    // Lazy insertion sort
    int *a = (int *)malloc( sizeof(int) * mask_size );
    memcpy( a, seen_pixels, sizeof(int) * mask_size ); 
    int temp,i,j;
    for( i=1;i<mask_size;i++){
        temp=a[i];
        j=i-1;
        while((temp<a[j])&&(j>=0)){
            a[j+1]=a[j];
            j=j-1;
        }
        a[j+1]=temp;
    }
    // Assign to the index-to-pixel array
    H->buff_idx_to_pix = a;
    H->param_min_seen_pix = a[0];
    H->param_max_seen_pix = a[ mask_size - 1 ];
}


int
healpixMap_pix2idx( healpixMap *H, int pix )
{

    int idx = -1;

    assert( "queried pixel outside range." &&
             H->param_npix > pix );

    if( H->flag_has_mask )
    {
        // binary search :-D
        int mid, low=0, high = H->param_seen_npix-1;
        
        while( low <= high )
        {
            mid = (high + low)/2;
            
            //printf( "%ld buff_idx_to_pix[%ld]\n", mid, H->buff_idx_to_pix[mid] );
            
            if( H->buff_idx_to_pix[mid] == pix )
            {
                idx = mid;
                break;
            }
            else if( pix < H->buff_idx_to_pix[mid] )
                high = mid - 1;
            else
                low = mid + 1;
        }
    }

    else
    {
        idx = pix;
    }

    return idx;
}


float
healpixMap_get_pixel_value( healpixMap *H, int pix )
{

    assert( "queried map does not have pixels." &&
             H->flag_has_pixels );

    assert( "queried pixel outside range." &&
             H->param_npix > pix );
    
    int idx;
    idx = healpixMap_pix2idx( H, pix );

    return H->buff_pixels[ idx ];

}


void
healpixMap_generate_pointing( healpixMap *H )
{   
    if( H->flag_has_mask ) 
    {
        H->buff_ptg_pix = (float*)malloc( 4*sizeof(float) * H->param_seen_npix );

        int i;
        // Iterate over non masked pixels
        for( i=0; i < H->param_seen_npix; i++ )
        {    
            double r[3],v[4];
            
            // Get pixel number
            int pixel = H->buff_idx_to_pix[i];

            // C-Healpix call: transform healpix pixel number 
            // to a vector and store that in r
            pix2vec_ring( H->param_nside, pixel, r );
            
            // Transfer x,y,z coordinates to buffer
            v[0] = r[0]; v[1] = r[1]; v[2] = r[2];
            // Last value is the actual pixel value. This is only done
            // for faster CPU to GPU memory transfers.
            v[3] = H->buff_pixels[ pixel ];
            
            // memcpy is cleaner
            memcpy( H->buff_ptg_pix+i, v, sizeof(float)*4 );
        }
    }
}
