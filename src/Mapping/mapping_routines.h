#ifndef __MAPPINGH__
#define __MAPPINGH__

void                                                                                                          
libmapping_project_data_to_matrices                                                                           
(                                                                                                             
    // input                                                                                                  
    int nsamples , int ndets, 
    double phi[], double theta[], double psi[],  
    double det_pol_angles[],
    float data[] , int bad_data_samples[], 
    int dets_to_map[], 
    int map_nside,                                                                                            
    int map_size , int pixels_in_the_map[],
    // output                                                                                              
    double AtA[], double AtD[]
);

void   
libmapping_get_IQU_from_matrices                                                                              
(                                                                                                             
    // input                                                                                                  
    int map_nside,                                                                                            
    int map_size ,                                                                                            
    double AtA[], double AtD[], int pixels_in_the_map[],                                                      
    // output                                                                                                 
    float I[], float Q[], float U[], float W[]                                                            
);

#endif
