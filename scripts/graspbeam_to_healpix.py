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
# extract real and phase parts of the data
fieldMag = numpy.abs(fieldData)
fieldPhase = numpy.angle(fieldData)
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
# setup output file
bpth = os.path.dirname(datapath)
fnm = os.path.basename(datapath)
newfnm = 'healpix_' + fnm
finalpth = os.path.join(bpth, newfnm)
beamData = numpy.vstack([magco, phsco, magcx, phscx]).swapaxes(0,1)
numpy.savetxt(finalpth, beamData)
