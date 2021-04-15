"""
This program loads the data from the GRASP file (a columnar file with a 
header), reads in some parameters like the grid size and resolution and 
then the data. It uses StringIO because it is faster.  The program then 
figures out the healpix pixel coordinates of a spherical cap of 5 
degrees (that's arbitrary) , sets up interpolators for the magnitude 
and phases (using sin/cos so the interpolation doesn't blow up if the 
phase moves from 360 to 1)  finally, rebuilds everything and saves 
it to disk.
"""
import sys
import os
import io
import numpy
import scipy
import healpy
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


datapath = sys.argv[1]
nside = int(sys.argv[2])
bpth = os.path.dirname(datapath)
fnm = os.path.basename(datapath)

f = open(datapath, 'r')
goodStuff = "++++"
for rawline in f:
    line = rawline.strip()
    print(line)
    if line.strip('\n') == goodStuff:
        break
data = f.readline()
otherdata = f.readline()
moredata = f.readline()
ranges = list(map(float, f.readline().strip().split()))
gridsizes = list(map(int, f.readline().strip().split()))
fhandler = io.StringIO(f.read())
fieldData = numpy.loadtxt(fhandler).view(complex)
fhandler.close()
f.close()

nx, ny, _ = gridsizes
umin, vmin, umax, vmax = ranges
u = numpy.linspace(umin, umax, nx, endpoint=True)
v = numpy.linspace(vmin, vmax, ny, endpoint=True)
# setup coordinates for interpolation
maxPix = healpy.ang2pix(nside, numpy.deg2rad(5.0), 0.0)
pixels = numpy.arange(0, maxPix)
theta, phi = healpy.pix2ang(nside, pixels)
uh = numpy.sin(theta)*numpy.cos(phi)
vh = numpy.sin(theta)*numpy.sin(phi)
# make plots of polarized beams as described in Rosset et al. 2010
tildeI = numpy.abs(fieldData[:, 0])**2 + numpy.abs(fieldData[:, 1])**2
tildeQ = numpy.abs(fieldData[:, 0])**2 - numpy.abs(fieldData[:, 1])**2
tildeU = 2*numpy.real(fieldData[:, 0] * numpy.conj(fieldData[:, 1]))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

imI = axes[0].imshow(tildeI.reshape((ny, nx)),
    extent=(umin, umax, vmin, vmax))
axes[0].set_xlim((-0.2, 0.2))
axes[0].set_ylim((-0.2, 0.2))
axes[0].set_xlabel('u')
axes[0].set_xlabel('v')
axes[0].set_title("tilde I")
plt.colorbar(imI, ax=axes[0])

imQ = axes[1].imshow(tildeQ.reshape((ny, nx)),
    extent=(umin, umax, vmin, vmax))
axes[1].set_xlim((-0.2, 0.2))
axes[1].set_ylim((-0.2, 0.2))
axes[1].set_xlabel('u')
axes[1].set_xlabel('v')
axes[1].set_title("tilde Q")
plt.colorbar(imQ, ax=axes[1])

imU = axes[2].imshow(tildeU.reshape((ny, nx)),
    extent=(umin, umax, vmin, vmax))
axes[2].set_xlim((-0.2, 0.2))
axes[2].set_ylim((-0.2, 0.2))
axes[2].set_xlabel('u')
axes[2].set_xlabel('v')
axes[2].set_title("tilde U")
plt.colorbar(imU, ax=axes[2])
fig.tight_layout()
plt.show()

plotname = 'polbeamplot_' + fnm
plotpth = os.path.join(bpth, plotname)
#plt.savefig(plotpth)

# extract real and phase parts of the data
fieldMag = numpy.abs(fieldData)

fieldPhase = numpy.angle(fieldData)
plt.imshow(fieldPhase[:, 1].reshape((ny, nx)))
plt.show()
# put phase in [0, 2pi] range
fieldPhase[fieldPhase < 0] += 2*numpy.pi
# compute cos/sin phases for better interpolation
cosFieldPhase = numpy.cos(fieldPhase)
sinFieldPhase = numpy.sin(fieldPhase)
# build interpolators
magir_co = RectBivariateSpline(u, v, fieldMag[:,0].reshape((ny, nx)))
cphsir_co = RectBivariateSpline(u, v, cosFieldPhase[:, 0].reshape((ny, nx)))
sphsir_co = RectBivariateSpline(u, v, sinFieldPhase[:, 0].reshape((ny, nx)))
magir_cx = RectBivariateSpline(u, v, fieldMag[:,1].reshape((ny, nx)))
cphsir_cx = RectBivariateSpline(u, v, cosFieldPhase[:, 1].reshape((ny, nx)))
sphsir_cx = RectBivariateSpline(u, v, sinFieldPhase[:, 1].reshape((ny, nx)))
# calculate x and y components of jones vectors
magco = magir_co(uh, vh, grid=False)
cphsco = cphsir_co(uh, vh, grid=False)
sphsco = sphsir_co(uh, vh, grid=False)
phsco = numpy.arctan2(sphsco, cphsco)

magcx = magir_cx(uh, vh, grid=False)
cphscx = cphsir_cx(uh, vh, grid=False)
sphscx = sphsir_cx(uh, vh, grid=False)
phscx = numpy.arctan2(sphscx, cphscx)

pltdata = numpy.zeros(12*nside**2)
pltdata[0:len(phscx)] = phscx
healpy.orthview(pltdata, rot=(0, 90, 0))
plt.show()

# setup output file
newfnm = 'healpix_' + fnm
finalpth = os.path.join(bpth, newfnm)
beamData = numpy.vstack([magco, phsco, magcx, phscx]).swapaxes(0,1)
numpy.savetxt(finalpth, beamData)
