#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <bpoint.h>

int main( void )
{
    int secs     = 180;
    int sps      =   1;   
    int ndets    =   1;
    int nsamples = secs * sps;
    int i,det;

    double  ctime[ nsamples ]; 
    double  az_ac[ nsamples ]; 
    double alt_ac[ nsamples ];
    double rot_ac[ nsamples ];

    double            dx[ndets];
    double            dy[ndets];
    double det_pol_angle[ndets];

    double *fp_dist;
    double *fp_bear;
    double *fp_rot ;
    
    fp_dist = (double *)malloc( sizeof(double) * ndets * nsamples );
    fp_bear = (double *)malloc( sizeof(double) * ndets * nsamples );
    fp_rot  = (double *)malloc( sizeof(double) * ndets * nsamples );

    for( i=0; i < nsamples; i++ )
    {
        ctime[i] = 1528977180 + 1.0/sps * i;
        az_ac[i] = (67.1371 + 0.0) * M_PI/180.0;
       alt_ac[i] =  0.0            * M_PI/180.0;
       rot_ac[i] = 0.0;
    }

    // Dummy focal plane
    for( i=0; i < ndets; i++ )
    {
        dx[i] = 0.0 *  M_PI/180.0;
        dy[i] = 0.0;
        det_pol_angle[i] = 0.0;
    }

    bpoint_recenter_focal_plane_to_moon(
        nsamples, ctime, az_ac, alt_ac, rot_ac,
        ndets   , dx, dy, det_pol_angle,
        "/home/pafluxa/software/bpoint/data/jpleph2013_40.405",
        fp_dist, fp_bear, fp_rot );
    
    struct tm *gm_time;
    time_t time_ctime; 
    for( i=0; i < nsamples; i++ )
    {
        time_ctime = ctime[ i ];
        // Convert ctime to broken-down calendar time
        gm_time  = gmtime( &time_ctime );
        // Convert broken-down time to 2-part JD
        // round off seconds
        double frac_secs = ctime[i] - (long)( time_ctime );
        double seconds   = gm_time->tm_sec + frac_secs;     
        
        printf( "%d-%d-%d %02d:%02d:%02lf ", 
            gm_time->tm_year + 1900,
            gm_time->tm_mon  + 1   ,
            gm_time->tm_mday       ,
            gm_time->tm_hour       ,
            gm_time->tm_min        ,
            seconds );

        for( det=0; det < ndets; det++ )
        {

            printf( "%lf %lf \n", fp_dist[ det*nsamples + i ]*180/M_PI, fp_bear[det*nsamples+i]*180/M_PI );
        }
    }

    free( fp_dist );
    free( fp_bear );
    free( fp_rot  );

    return 0;
}

