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

        Convolver(unsigned long nsamples);
       ~Convolver();
        //bool set_threads( unsigned int n );
        //unsigned int get_threads( void );
        void exec_convolution(
            float* data_a, float* data_b,
            char polFlag,
            Scan& scan, Sky& sky, PolBeam& beam);

    private:

        unsigned long nsamples;
        unsigned int nthreads;

        float* masterBuffer;
        std::vector<long> bufferStart;
        std::vector<long> bufferEnd;

        void beam_times_sky(
            Sky& sky,
            PolBeam& beam,
            float phi0,
            float theta0,
            float psi0,
            float* data_a, float *data_b);

};

#endif
