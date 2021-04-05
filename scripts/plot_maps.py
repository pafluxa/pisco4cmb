import sys

import healpy
import numpy
import matplotlib.pyplot as plt

I1, Q1, U1, V1 = numpy.loadtxt('maps_input.txt', unpack=True)
I2, Q2, U2 = numpy.loadtxt('maps_output.txt', unpack=True)

# normalize
Is, Qs, Us = healpy.smoothing(
    (I1, Q1, U1), fwhm=numpy.deg2rad(2.0), pol=True)

I1m = healpy.gnomview(Is, rot=(180, 0.0, 0.0),
    reso=5.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
Q1m = healpy.gnomview(Qs, rot=(180, 0.0, 0.0),
    reso=5.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
U1m = healpy.gnomview(Us, rot=(180, 0.0, 0.0),
    reso=5.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)

I2m = healpy.gnomview(I2, rot=(180, 0.0, 0.0),
    reso=5.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
Q2m = healpy.gnomview(Q2, rot=(180, 0.0, 0.0),
    reso=5.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
U2m = healpy.gnomview(U2, rot=(180, 0.0, 0.0),
    reso=5.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)

fix, axes = plt.subplots(2, 3, figsize=(12, 5))
axes[0][0].imshow(I1m)
axes[0][1].imshow(Q1m)
axes[0][2].imshow(U1m)

axes[1][0].imshow(I2m)
axes[1][1].imshow(Q2m)
axes[1][2].imshow(U2m)

plt.show()
