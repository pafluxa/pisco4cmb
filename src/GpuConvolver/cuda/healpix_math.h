#ifndef __CUDDAHEALPIXMATHH__
#define __CUDDAHEALPIXMATHH__

#include <cuda.h>

#define twothird   (2.0f/3.0f)
#define pi         (3.141592653589793238462643383279502884197f)
#define twopi      (6.283185307179586476925286766559005768394f)
#define half_pi    (1.570796326794896619231321691639751442099f)
#define inv_halfpi (0.636619772367581343075535053490057400000f)

inline  __device__ int
imodulo (int v1, int v2)
{
    int v = v1 % v2;
    return (v >= 0) ? v : v + v2;
}

inline  __device__ int
isqrt(int v)
{
    return int(sqrtf((float)(v) + 0.5f));
}

#endif
