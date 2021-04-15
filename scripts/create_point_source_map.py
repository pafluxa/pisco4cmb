import healpy
import numpy
import sys

nside = int(sys.argv[1])
sI = float(sys.argv[2])
sQ = float(sys.argv[3])
sU = float(sys.argv[4])
npix = healpy.nside2npix(nside)


skymap = numpy.zeros(npix, dtype='float32')
skymap[healpy.ang2pix(nside, numpy.pi/2.0, numpy.pi)] = 1.0

data = numpy.vstack([sI*skymap, sQ*skymap, sU*skymap, 0*skymap]).T
numpy.savetxt('maps_input.txt', data)
print('done')

