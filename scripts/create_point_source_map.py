import healpy
import numpy

nside = 128
npix = healpy.nside2npix(nside)

skymap = numpy.zeros(npix, dtype='float32')
skymap[healpy.ang2pix(nside, numpy.pi/2.0, numpy.pi)] = 1.0

data = numpy.vstack([skymap, skymap, skymap*0.0, skymap*0.0]).T
print(data.shape)
numpy.savetxt('maps_input.txt', data)


