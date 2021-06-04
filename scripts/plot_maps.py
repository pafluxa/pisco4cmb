import os
import sys
import healpy
import numpy
import matplotlib.pyplot as plt

nside = int(sys.argv[1])
npix = 12 * nside * nside
pols = sys.argv[2]
datapath = sys.argv[3]
plotpath = os.path.basename(datapath) + ".png"
plotpath = os.path.join(os.path.dirname(datapath), plotpath)
Idata, Qdata, Udata, Vdata = numpy.loadtxt(datapath, unpack=True)
I2 = numpy.zeros_like(Idata)
Q2 = numpy.zeros_like(Qdata)
U2 = numpy.zeros_like(Udata)
I2[numpy.argmax(Idata)] = 1.0
if pols == 'Q':
    Q2[numpy.argmax(Idata)] = 1.0
if pols == 'U':
    U2[numpy.argmax(Idata)] = 1.0
    
# normalize
Is, Qs, Us = healpy.smoothing((I2, Q2, U2), fwhm=numpy.deg2rad(2.0), pol=True)

I1m = healpy.gnomview(Is, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
Q1m = healpy.gnomview(Qs, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
U1m = healpy.gnomview(Us, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)

I2m = healpy.gnomview(Idata, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
Q2m = healpy.gnomview(Qdata, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)
U2m = healpy.gnomview(Udata, rot=(180, 0.0, 0.0),
    reso=2.0, xsize=200, ysize=200,
    no_plot=True, return_projected_map=True)

exnt = (-2*100/60, 2*100/60.0, -2*100/60, 2*100/60)
fig, axes = plt.subplots(2, 3, figsize=(12, 5))
imI1 = axes[0][0].imshow(I1m, extent=exnt)
axes[0][0].set_title("In. Stokes I (smooth)")
imQ1 = axes[0][1].imshow(Q1m, extent=exnt)
axes[0][1].set_title("In. Stokes Q (smooth)")
imU1 = axes[0][2].imshow(U1m, extent=exnt)
axes[0][2].set_title("In. Stokes U (smooth)")

imI2 = axes[1][0].imshow(I2m, extent=exnt)
axes[1][0].set_title("Out Stokes I (PISCO)")
imQ2 = axes[1][1].imshow(Q2m, extent=exnt)
axes[1][1].set_title("Out Stokes Q (PISCO)")
imU2 = axes[1][2].imshow(U2m, extent=exnt)
axes[1][2].set_title("Out Stokes U (PISCO)")

plt.colorbar(imI1, ax=axes[0][0])
plt.colorbar(imQ1, ax=axes[0][1])
plt.colorbar(imU1, ax=axes[0][2])

plt.colorbar(imI2, ax=axes[1][0])
plt.colorbar(imQ2, ax=axes[1][1])
plt.colorbar(imU2, ax=axes[1][2])

fig.tight_layout()

plt.show()
#plt.savefig(plotpath)
