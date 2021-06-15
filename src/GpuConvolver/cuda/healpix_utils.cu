#include "healpix_math.h"
#include "healpix_utils.h"

// Ported from Healpix_base.cc
__device__ int
ring_above (int nside_, float z) 
{                                                                                                           
  float az = fabs(z);                                                                                           
  if (az <= twothird) // equatorial region                                                                      
    return int( nside_*(2-1.5*z) );                                                                               

  int iring = int(nside_ * sqrtf(3 * (1 - az)));                                                                         
  return (z > 0) ? iring : 4 * nside_ - iring - 1;                                                                    
}  

// Ported from Healpix_base.cc
__device__ void 
pix2ang_ring_z_phi (int nside_, int pix, float *z, float *phi)                                  
{                                                                                                           

  long ncap_ = nside_ * (nside_ - 1) * 2;                                                                             
  long npix_ = 12 * nside_ * nside_;                                                                                
  float fact2_ = 4. / npix_;                                                                                   

  if (pix < ncap_) /* North Polar cap */                                                                        
    {                                                                                                         
    int iring = (1 + isqrt(1 + 2 * pix)) >> 1; /* counted from North pole */                                          
    int iphi  = (pix + 1) - 2 * iring * (iring - 1);                                                                  
                                                                                                              
    *z = 1.0f - (iring * iring) * fact2_;                                                                          
    *phi = (iphi - 0.5f) * half_pi / iring;                                                                         
    }                                                                                                         
  else if (pix<(npix_-ncap_)) /* Equatorial region */                                                         
    {                                                                                                         
    float fact1_  = (nside_ << 1) * fact2_;                                                                      
    int ip  = pix - ncap_;                                                                                    
    int iring = ip / (4 * nside_) + nside_; /* counted from North pole */                                         
    int iphi  = ip % (4 * nside_) + 1;                                                                            
    /* 1 if iring+nside is odd, 1/2 otherwise */                                                              
    float fodd = ((iring + nside_) & 1) ? 1 : 0.5f;                                                               
                                                                                                              
    int nl2 = 2 * nside_;                                                                                       
    *z = (nl2 - iring) * fact1_;                                                                                  
    *phi = (iphi - fodd) * pi / nl2;                                                                              
    }                                                                                                         
  else /* South Polar cap */                                                                                  
    {                                                                                                         
    int ip = npix_ - pix;                                                                                     
    int iring = (1 + isqrt(2 * ip - 1)) >> 1; /* counted from South pole */                                           
    int iphi  = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));                                                       
                                                                                                              
    *z = -1.0 + (iring * iring) * fact2_;                                                                         
    *phi = (iphi - 0.5) * half_pi / iring;                                                                         
    }                                                                                                          
  }                                                                                                           

// Ported from Healpix_base.cc
__device__ int 
ang2pix_ring_z_phi ( int nside_, float z, float phi, float sth )
{
  
  float za = fabs(z);
  float tt = fmod(phi * inv_halfpi, 4.0f); /* in [0,4) */

  if (za <= twothird) /* Equatorial region */
    {
    float temp1 = nside_ * (0.5 + tt);
    float temp2 = nside_ * z * 0.75;
    int jp = int(temp1 - temp2); /* index of  ascending edge line */
    int jm = int(temp1 + temp2); /* index of descending edge line */

    /* ring number counted from z=2/3 */
    int ir = nside_ + 1 + jp - jm; /* in {1,2n+1} */
    int kshift = 1 - (ir & 1); /* kshift=1 if ir even, 0 otherwise */

    int ip = (jp + jm - nside_ + kshift + 1) / 2; /* in {0,4n-1} */
    ip = imodulo(ip, 4 * nside_);

    return nside_ * (nside_ - 1) * 2 + (ir - 1) * 4 * nside_ + ip;
    }
  else  /* North & South polar caps */
    {
    float tp = tt - int(tt);
    float tmp = -1;
    if( za < 0.990f )
      tmp = nside_ * sqrtf(3.0f * (1.0f - za));
    else
      tmp = nside_ * sth / sqrtf((1.0f + za) / 3.0f);

    int jp = int(tp * tmp); /* increasing edge line index */
    int jm = int((1.0f - tp) * tmp); /* decreasing edge line index */

    int ir = jp + jm + 1; /* ring number counted from the closest pole */
    int ip = int(tt * ir); /* in {1,4*ir-1} */
    //ip = cudachealpix_imodulo(ip,4*ir);
    int r1 = z > 0 ? 
        2 * ir * (ir - 1) + ip : 
        12 * nside_ * nside_ - 2 * ir * (ir + 1) + ip;;
    return r1;
    /*
    if (z>0)
      return 2*ir*(ir-1) + ip;
    else
      return 12*nside_*nside_ - 2*ir*(ir+1) + ip;
    */
    }
}

#ifdef CUDACONV_USE_INTERPOLATION
// Ported from Healpix_base.cc
__device__ void
get_ring_info2 (int nside_, int ring, int *startpix, int *ringpix, double *theta, bool *shifted)
{
  long ncap_=nside_*(nside_-1)*2;
  long npix_=12*nside_*nside_;
  double fact2_ = 4./npix_;
  double fact1_  = (nside_<<1)*fact2_;
  int northring = (ring>2*nside_) ? 4*nside_-ring : ring;
  if (northring < nside_)
    {
    double tmp = northring*northring*fact2_;
    double costheta = 1 - tmp;
    double sintheta = sqrt(tmp*(2-tmp));
    *theta = atan2(sintheta,costheta);
    *ringpix = 4*northring;
    *shifted = true;
    *startpix = 2*northring*(northring-1);
    }
  else
    {
    *theta = acos((2*nside_-northring)*fact1_);
    *ringpix = 4*nside_;
    *shifted = ((northring-nside_) & 1) == 0;
    *startpix = ncap_ + (northring-nside_)*( *ringpix );
    }
  if (northring != ring) // southern hemisphere
    {
    *theta = pi-*theta;
    *startpix = npix_ - *startpix - *ringpix;
    }
}
#endif

// Ported from Healpix_base.cc
__device__ int cudaHealpix::ang2pix(int nside_map, float tht, float phi)
{
    int map_pix;
    float cth = cosf(tht);
    float sth = sinf(tht);
    map_pix = (int)(ang2pix_ring_z_phi( nside_map, cth, phi, sth));
    return map_pix;
}

// Ported from Healpix_base.cc
__device__ void cudaHealpix::pix2ang(long nside, long ipix, float *theta, float *phi)                                          
{                                                                                                           
  float z;                                                                                                   
  pix2ang_ring_z_phi (nside,ipix,&z,phi);                                                                     
  *theta = acosf(z);                                                                                             
}

#ifdef CUDACONV_USE_INTERPOLATION
// Ported from Healpix_base.cc
__device__ void cudaHealpix::get_interpol( int nside_, double theta, double phi, int pix[4], double wgt[4] )
{
  int npix_ = 12*nside_*nside_;
  double z = cos (theta);                                                                                 
  int ir1 = ring_above(nside_, z);                                                                                      
  int ir2 = ir1+1;                                                                                              
  double theta1, theta2, w1, tmp, dphi;                                                                       
  int sp,nr;                                                                                                    
  bool shift;                                                                                                 
  int i1,i2;                                                                                                    
  if (ir1>0)                                                                                                  
    {                                                                                                         
    get_ring_info2 ( nside_, ir1, &sp, &nr, &theta1, &shift);                                                              
    dphi = twopi/nr;                                                                                          
    tmp = (phi/dphi - .5*shift);                                                                          
    i1 = (tmp<0) ? int(tmp)-1 : int(tmp);                                                                         
    w1 = (phi-(i1+.5*shift)*dphi)/dphi;                                                                   
    i2 = i1+1;                                                                                                
    if (i1<0) i1 +=nr;                                                                                        
    if (i2>=nr) i2 -=nr;                                                                                      
    pix[0] = sp+i1; pix[1] = sp+i2;                                                                           
    wgt[0] = 1-w1; wgt[1] = w1;                                                                               
    }                                                                                                         
  if (ir2<(4*nside_))                                                                                         
    {                                                                                                         
    get_ring_info2 (nside_, ir2, &sp, &nr, &theta2, &shift);                                                              
    dphi = twopi/nr;                                                                                          
    tmp = (phi/dphi - .5*shift);                                                                          
    i1 = (tmp<0) ? int(tmp)-1 : int(tmp);                                                                         
    w1 = (phi-(i1+.5*shift)*dphi)/dphi;                                                                   
    i2 = i1+1;                                                                                                
    if (i1<0) i1 +=nr;                                                                                        
    if (i2>=nr) i2 -=nr;                                                                                      
    pix[2] = sp+i1; pix[3] = sp+i2;                                                                           
    wgt[2] = 1-w1; wgt[3] = w1;                                                                               
    }                                                                                                         
                                                                                                              
  if (ir1==0)                                                                                                 
    {                                                                                                         
    double wtheta = theta/theta2;                                                                         
    wgt[2] *= wtheta; wgt[3] *= wtheta;                                                                       
    double fac = (1-wtheta)*0.25;                                                                             
    wgt[0] = fac; wgt[1] = fac; wgt[2] += fac; wgt[3] +=fac;                                                  
    pix[0] = (pix[2]+2)&3;                                                                                    
    pix[1] = (pix[3]+2)&3;                                                                                    
    }                                                                                                         
  else if (ir2==4*nside_)                                                                                     
    {                                                                                                         
    double wtheta = (theta-theta1)/(pi-theta1);                                                           
    wgt[0] *= (1-wtheta); wgt[1] *= (1-wtheta);                                                               
    double fac = wtheta*0.25;                                                                                 
    wgt[0] += fac; wgt[1] += fac; wgt[2] = fac; wgt[3] =fac;                                                  
    pix[2] = ((pix[0]+2)&3)+npix_-4;                                                                          
    pix[3] = ((pix[1]+2)&3)+npix_-4;                                                                          
    }                                                                                                         
  else                                                                                                        
    {                                                                                                         
    double wtheta = (theta-theta1)/(theta2-theta1);                                                       
    wgt[0] *= (1-wtheta); wgt[1] *= (1-wtheta);                                                               
    wgt[2] *= wtheta; wgt[3] *= wtheta;                                                                       
    }                                                                                                         
}
#endif
