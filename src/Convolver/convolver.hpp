/*
 * convolver.hpp
 *
 */
#ifndef _CPUCONVOLVERH
#define _CPUCONVOLVERH

#include "Sky/sky.hpp"
#include "Scan/scan.hpp"
#include "Polbeam/polbeam.hpp"

#include <vector>

class Convolver
{
    public:

        Convolver(Scan* scan, Sky* sky, PolBeam* beam);
       ~Convolver();
        void exec_convolution(char polFlag, float* data_a, float* data_b);

    private:

        unsigned long nsamples;

        Scan* scan;
        Sky* sky;
        PolBeam* beam;

        void beam_times_sky(
            double phi0,
            double theta0,
            double psi0,
            double* data_a, double *data_b);

};

#endif
