import os
import io
import sys
import healpy
import numpy
import matplotlib.pyplot as plt

pols = sys.argv[1]
datapath = sys.argv[2]
fwhmbeam = float(sys.argv[3])
plotpath = os.path.basename(datapath) + ".png"
plotpath = os.path.join(os.path.dirname(datapath), plotpath)

f = open(datapath, 'r')
fhandler = io.StringIO(f.read())
mapData = numpy.loadtxt(fhandler)
Idata = mapData[:, 0]  
Qdata = mapData[:, 1] 
Udata = mapData[:, 2] 
Hdata = mapData[:, 3] 
fhandler.close()
f.close()

npix = len(Idata)
nside = healpy.npix2nside(npix)
I2 = numpy.zeros_like(Idata)
Q2 = numpy.zeros_like(Qdata)
U2 = numpy.zeros_like(Udata)
# get center of map assuming max of I
centerpix = numpy.argmax(Idata)
tcent, pcent = numpy.degrees(healpy.pix2ang(nside, centerpix))
ra0 = pcent
dec0 = 90.0 - tcent
# "paint" point source
I2[centerpix] = 1.0
if pols == 'Q':
    Q2[centerpix] = 1.0
if pols == 'U':
    U2[centerpix] = 1.0

# smooth analytically
Is, Qs, Us = healpy.smoothing((I2, Q2, U2), fwhm=numpy.deg2rad(fwhmbeam), pol=True)

# mask unseen pixels
Is[Hdata < 1] = numpy.nan
Qs[Hdata < 1] = numpy.nan
Us[Hdata < 1] = numpy.nan
Idata[Hdata < 1] = numpy.nan
Qdata[Hdata < 1] = numpy.nan
Udata[Hdata < 1] = numpy.nan

# get minimums and maximums for later
imax = numpy.nanmax(Is)
qmax = numpy.nanmax(Qs)
umax = numpy.nanmax(Us)
imin = -imax#numpy.nanmin(Is)
qmin = -qmax#numpy.nanmin(Qs)
umin = -umax#numpy.nanmin(Us)

plotsize = 400
# in arcmin
reso = 2.0
plotext = reso * (plotsize/2) * (1.0 / 60.0)
I1m = healpy.gnomview(Is, rot=(ra0, dec0, 0.0),
    reso=reso, xsize=plotsize, ysize=plotsize, min=imin, max=imax,
    no_plot=True, return_projected_map=True)
Q1m = healpy.gnomview(Qs, rot=(ra0, dec0, 0.0),
    reso=reso, xsize=plotsize, ysize=plotsize, min=qmin, max=qmax,
    no_plot=True, return_projected_map=True)
U1m = healpy.gnomview(Us, rot=(ra0, dec0, 0.0),
    reso=reso, xsize=plotsize, ysize=plotsize, min=umin, max=umax,
    no_plot=True, return_projected_map=True)

I2m = healpy.gnomview(Idata, rot=(ra0, dec0, 0.0),
    reso=reso, xsize=plotsize, ysize=plotsize, min=imin, max=imax,
    no_plot=True, return_projected_map=True)
Q2m = healpy.gnomview(Qdata, rot=(ra0, dec0, 0.0),
    reso=reso, xsize=plotsize, ysize=plotsize, min=qmin, max=qmax,
    no_plot=True, return_projected_map=True)
U2m = healpy.gnomview(Udata, rot=(ra0, dec0, 0.0),
    reso=reso, xsize=plotsize, ysize=plotsize, min=umin, max=umax,
    no_plot=True, return_projected_map=True)

I2r = healpy.gnomview(Idata - Is, rot=(ra0, dec0, 0.0),
    reso=reso, xsize=plotsize, ysize=plotsize, min=imin / 10, max=imax / 10,
    no_plot=True, return_projected_map=True)
Q2r = healpy.gnomview(Qdata - Qs, rot=(ra0, dec0, 0.0),
    reso=reso, xsize=plotsize, ysize=plotsize, min=qmin / 10, max=qmax / 10,
    no_plot=True, return_projected_map=True)
U2r = healpy.gnomview(Udata - Us, rot=(ra0, dec0, 0.0),
    reso=reso, xsize=plotsize, ysize=plotsize, min=umin / 5, max=umax / 5,
    no_plot=True, return_projected_map=True)

exnt = (-plotext, plotext, -plotext, plotext)
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
axes[0][0].set_title("In. Stokes I (smooth)")
axes[0][1].set_title("In. Stokes Q (smooth)")
axes[0][2].set_title("In. Stokes U (smooth)")
axes[1][0].set_title("Out Stokes I (PISCO)")
axes[1][1].set_title("Out Stokes Q (PISCO)")
axes[1][2].set_title("Out Stokes U (PISCO)")
axes[2][0].set_title("Residuals (I)")
axes[2][1].set_title("Residuals (Q)")
axes[2][2].set_title("Residuals (U)")

imI1 = axes[0][0].imshow(I1m, extent=exnt, vmin=imin, vmax=imax)
imQ1 = axes[0][1].imshow(Q1m, extent=exnt, vmin=qmin, vmax=qmax)
imU1 = axes[0][2].imshow(U1m, extent=exnt, vmin=umin, vmax=umax)

imI2 = axes[1][0].imshow(I2m, extent=exnt, vmin=imin, vmax=imax)
imQ2 = axes[1][1].imshow(Q2m, extent=exnt, vmin=qmin, vmax=qmax)
imU2 = axes[1][2].imshow(U2m, extent=exnt, vmin=umin, vmax=umax)

imrI = axes[2][0].imshow(I2r, extent=exnt, vmin=imin / 10, vmax=imax / 10)
imrQ = axes[2][1].imshow(Q2r, extent=exnt, vmin=qmin / 10, vmax=qmax / 10)
imrU = axes[2][2].imshow(U2r, extent=exnt, vmin=umin / 5, vmax=umax / 5)

plt.colorbar(imI1, ax=axes[0][0])
plt.colorbar(imQ1, ax=axes[0][1])
plt.colorbar(imU1, ax=axes[0][2])

plt.colorbar(imI2, ax=axes[1][0])
plt.colorbar(imQ2, ax=axes[1][1])
plt.colorbar(imU2, ax=axes[1][2])

plt.colorbar(imrI, ax=axes[2][0])
plt.colorbar(imrQ, ax=axes[2][1])
plt.colorbar(imrU, ax=axes[2][2])

fig.tight_layout()

plt.show()
plt.savefig(plotpath)
