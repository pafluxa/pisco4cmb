import sys

import healpy
import numpy
import matplotlib.pyplot as plt

map1 = sys.argv[1]
map2 = sys.argv[2]
I1, Q1, U1, V1 = numpy.loadtxt(map1, unpack=True)
I2, Q2, U2 = numpy.loadtxt(map2, unpack=True)

# normalize
Is, Qs, Us = healpy.smoothing(
    (I1, Q1, U1), fwhm=numpy.deg2rad(1.1), pol=True)
ns = numpy.sum(Is)
np = numpy.sum(I2)
r = ns/np
I2 = I2*r
Q2 = Q2*r
U2 = U2*r

I1m = healpy.gnomview(Is, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
Q1m = healpy.gnomview(Qs, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
U1m = healpy.gnomview(Us, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)

I2m = healpy.gnomview(I2, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
Q2m = healpy.gnomview(Q2, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
U2m = healpy.gnomview(U2, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)

fig, axes = plt.subplots(2, 3, figsize=(12, 5))
axes[0][0].imshow(I1m)
axes[0][0].set_title("In. Stokes I (smooth)")
axes[0][1].imshow(Q1m)
axes[0][1].set_title("In. Stokes Q (smooth)")
axes[0][2].imshow(U1m)
axes[0][2].set_title("In. Stokes U (smooth)")

axes[1][0].imshow(I2m)
axes[1][0].set_title("Out Stokes I (PISCO)")
axes[1][1].imshow(Q2m)
axes[1][1].set_title("Out Stokes Q (PISCO)")
axes[1][2].imshow(U2m)
axes[1][2].set_title("Out Stokes U (PISCO)")

fig.tight_layout()

plt.show()
