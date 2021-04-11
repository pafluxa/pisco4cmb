import healpy
import numpy
import sys

nside = int(sys.argv[1])
npix = healpy.nside2npix(nside)

skymap = numpy.zeros(npix, dtype='float32')
skymap[healpy.ang2pix(nside, numpy.pi/2.0, numpy.pi)] = 1.0

data = numpy.vstack([skymap, skymap, skymap*0.0, skymap*0.0]).T
numpy.savetxt('point_source_IQ.txt', data)
print('done')

