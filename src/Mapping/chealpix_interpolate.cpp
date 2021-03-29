/*
 * This code is a port/fork a portion of healpix_base.cc
 * I only ported the functions that are used by get_interpol() 
 * in order to be able to interpolate between pixel values.
 */

#include "chealpix_interpolate.h"

#include <math.h>

#ifndef M_PI
#define M_PI (3.14159265)
#endif


int
chealpix_order2nside( int order )
{
    return 1 << order;
}

int
chealpix_nside2order( int nside )
{
    
    int index = nside;

    int targetlevel = 0;
    while (index >>= 1) ++targetlevel;
    
    return targetlevel;
}

int
chealpix_ring_above( int nside_, double z )
{
    double az=fabs(z);
    
    if (az<=2./3.) // equatorial region
        return  (int)(nside_*(2-1.5*z));
    
    int iring = (int)(nside_*sqrt(3*(1-az)));

    return (z>0) ? iring : 4*nside_-iring-1;
}


void
chealpix_get_ring_info2( int nside, int ring, int *startpix, int *ringpix, double *theta, int *shifted )
{

    // Ported from healpix_base.cc
    int order_  = chealpix_nside2order( nside );
    int nside_  = nside;
    int npface_ = nside_ << order_;
    int ncap_   = (npface_-nside_)<<1;
    int npix_   = 12*npface_;
    int fact2_  = 4./npix_;
    int fact1_  = (nside_<<1)*fact2_;
    
    int northring = (ring>2*nside_) ? 4*nside_-ring : ring;
    
    if (northring < nside_)
    {
        double tmp = northring*northring*fact2_;
        double costheta = 1 - tmp;
        double sintheta = sqrt(tmp*(2-tmp));
        
        *theta = atan2(sintheta,costheta);
        *ringpix = 4*northring;
        *shifted = 1;
        *startpix = 2*northring*(northring-1);
    }
    else
    {
        *theta = acos((2*nside_-northring)*fact1_);
        *ringpix = 4*nside_;
        *shifted = ((northring-nside_) & 1) == 0;
        *startpix = ncap_ + (northring-nside_)*(*ringpix);
    }
    if (northring != ring) // southern hemisphere
    {
        *theta = M_PI-(*theta);
        *startpix = npix_ - (*startpix) - (*ringpix);
    }
  
}

void
chealpix_get_interpolation( int nside, double phi, double theta, int *pix, double *wgt )
{  
    // Ported from healpix_base.cc
    int order_  = chealpix_nside2order( nside );
    int nside_  = nside;
    int npface_ = nside_ << order_;
    //int ncap_   = (npface_-nside_)<<1;
    int npix_   = 12*npface_;
    //int fact2_  = 4./npix_;
    //int fact1_  = (nside_<<1)*fact2_;
    
    double z = cos (theta);
    int ir1 = chealpix_ring_above(nside, z);
    int ir2 = ir1+1;
    double theta1, theta2, w1, tmp, dphi;
    int sp,nr;
    int shift;
    int i1,i2;
    
    if (ir1>0)
    {
        chealpix_get_ring_info2 (nside, ir1, &sp, &nr, &theta1, &shift);
        dphi = 2*M_PI/nr;
        tmp = (phi/dphi - .5*shift);
        i1 = (tmp<0) ? (int)(tmp)-1 : (int)(tmp);
        w1 = (phi-(i1+.5*shift)*dphi)/dphi;
        i2 = i1+1;
        
        if (i1<0) 
            i1 +=nr;

        if (i2>=nr) 
            i2 -=nr;
        
        pix[0] = sp+i1; 
        pix[1] = sp+i2;
        
        wgt[0] = 1-w1; 
        wgt[1] = w1;
    }
    
    if (ir2<(4*nside_))
    {
        chealpix_get_ring_info2 (nside, ir2, &sp, &nr, &theta2, &shift);
        dphi = 2*M_PI/nr;
        tmp = (phi/dphi - .5*shift);
        i1 = (tmp<0) ? (int)(tmp)-1 : (int)(tmp);
        w1 = (phi-(i1+.5*shift)*dphi)/dphi;
        i2 = i1+1;
        
        if (i1<0) 
            i1 +=nr;
        
        if (i2>=nr) 
            i2 -=nr;
        
        pix[2] = sp+i1; 
        pix[3] = sp+i2;
        
        wgt[2] = 1-w1; 
        wgt[3] = w1;
    }

    if (ir1==0)
    {
        double wtheta = theta/theta2;
        double fac = (1-wtheta)*0.25;
        
        wgt[2] *= wtheta; 
        wgt[3] *= wtheta;        
        wgt[0]  = fac; 
        wgt[1]  = fac; 
        wgt[2] += fac; 
        wgt[3] += fac;
        
        pix[0] = (pix[2]+2)&3;
        pix[1] = (pix[3]+2)&3;
    
    }
    
    else if (ir2==4*nside_)
    {
        double wtheta = (theta-theta1)/(M_PI-theta1);
        double fac = wtheta*0.25;
        
        wgt[0] *= (1-wtheta); 
        wgt[1] *= (1-wtheta);
        wgt[0] += fac; 
        wgt[1] += fac; 
        wgt[2]  = fac; 
        wgt[3]  = fac;

        pix[2] = ((pix[0]+2)&3)+npix_-4;
        pix[3] = ((pix[1]+2)&3)+npix_-4;
    }
    else
    {
    
        double wtheta = (theta-theta1)/(theta2-theta1);
        
        wgt[0] *= (1-wtheta); 
        wgt[1] *= (1-wtheta);
        wgt[2] *= wtheta; 
        wgt[3] *= wtheta;
    }
}

