import os
import io
import sys
import healpy
import numpy
import matplotlib.pyplot as plt

inputmap = sys.argv[1]
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

f = open(inputmap, 'r')
fhandler = io.StringIO(f.read())
mapData = numpy.loadtxt(fhandler)
I2 = mapData[:, 0]  
Q2 = mapData[:, 1] 
U2 = mapData[:, 2] 
fhandler.close()
f.close()

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

fig, axes = plt.subplots(3, 3, figsize=(12, 9))

plotext = 180.0
I1m = healpy.mollview(Is,  
    min=imin, max=imax,
    return_projected_map=True, sub=(3, 3, 1))
Q1m = healpy.mollview(Qs, 
    min=qmin, max=qmax,
    return_projected_map=True, sub=(3, 3, 2))
U1m = healpy.mollview(Us, 
    min=umin, max=umax,
    return_projected_map=True, sub=(3, 3, 3))

I2m = healpy.mollview(Idata, 
    min=imin, max=imax,
    return_projected_map=True, sub=(3, 3, 4))
Q2m = healpy.mollview(Qdata, 
    min=qmin, max=qmax,
    return_projected_map=True, sub=(3, 3, 5))
U2m = healpy.mollview(Udata, 
    min=umin, max=umax,
    return_projected_map=True, sub=(3, 3, 6))

I2r = healpy.mollview(Idata - Is, 
    min=imin / 10, max=imax / 10,
    return_projected_map=True, sub=(3, 3, 7))
Q2r = healpy.mollview(Qdata - Qs, 
    min=qmin / 10, max=qmax / 10,
    return_projected_map=True, sub=(3, 3, 8))
U2r = healpy.mollview(Udata - Us, 
    min=umin / 5, max=umax / 5,
    return_projected_map=True, sub=(3, 3, 9))
plt.close()

fig, axes = plt.subplots(3, 3, figsize=(12, 9))

exnt = (-180, 180, -90, 90)
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

imrI = axes[2][0].imshow(I2r, extent=exnt, vmin=imin / 100, vmax=imax / 100)
imrQ = axes[2][1].imshow(Q2r, extent=exnt, vmin=qmin / 100, vmax=qmax / 100)
imrU = axes[2][2].imshow(U2r, extent=exnt, vmin=umin / 100, vmax=umax / 100)

plt.colorbar(imI1, ax=axes[0][0], extend="both",location="bottom")
plt.colorbar(imQ1, ax=axes[0][1], extend="both",location="bottom")
plt.colorbar(imU1, ax=axes[0][2], extend="both",location="bottom")

plt.colorbar(imI2, ax=axes[1][0], extend="both",location="bottom")
plt.colorbar(imQ2, ax=axes[1][1], extend="both",location="bottom")
plt.colorbar(imU2, ax=axes[1][2], extend="both",location="bottom")

plt.colorbar(imrI, ax=axes[2][0], extend="both",location="bottom")
plt.colorbar(imrQ, ax=axes[2][1], extend="both",location="bottom")
plt.colorbar(imrU, ax=axes[2][2], extend="both",location="bottom")

fig.tight_layout()

plt.show()
#plt.savefig(plotpath)
