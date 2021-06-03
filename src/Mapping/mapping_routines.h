#ifndef __MAPPINGH__
#define __MAPPINGH__

void
libmapping_project_data_to_matrices
(
    // input
    int nsamples, 
    float phi[], float theta[], float psi[],
    int ndets, double det_pol_angles[],
    float data[],
    // output
    int map_nside,
    double AtA[], double AtD[]
);

void
libmapping_get_IQU_from_matrices
(
    // input
    int map_nside,
    double AtA[], double AtD[],
    // output
    float I[], float Q[], float U[], float W[]
);

#endif
