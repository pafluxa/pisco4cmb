#ifndef __CUDDAHEALPIXMATHH__
#define __CUDDAHEALPIXMATHH__

#include <cuda.h>

#define twothird   (2.0/3.0)
#define pi         (3.141592653589793238462643383279502884197)
#define twopi      (6.283185307179586476925286766559005768394)
#define half_pi    (1.570796326794896619231321691639751442099)
#define inv_halfpi (0.636619772367581343075535053490057400000)

inline  __device__ int
imodulo (int v1, int v2)
{
    int v = v1 % v2;
    return (v >= 0) ? v : v + v2;
}

inline  __device__ int
isqrt(int v)
{
    return __double2int_rd( sqrt( (double)(v) + 0.5) );
}

#endif
