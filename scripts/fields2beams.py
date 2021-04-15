"""
This program loads electric far-field data from a GRASP simulation and
outputs a file with 4 Healpix maps with their corresponding t-beams
(see doc/t-beams.pdf). 

Input corresponds to a txt file with the output of GRASP as a uv-grid,
and a command line parameter specifying the NSIDE parameter of the 
HEALPIX grid.

The program will uses the field data (co and cross-polar compontents
of the electric field density (per unit frequency per unit solid angle)
to compute the "tilde beams", a polarized representation of the 
electromagnetic response of a Polarization Sensitive Bolometer placed 
behind (in the time-forward sense) an optical system. 

T-beams are then interpolated from their original grid onto HEALPIX
pixels, defined by the input NSIDE parameter. The data is written to 
disk as a single columnar text file**, with one column per T-beam. The
output file is named after the input file, but with "hpx_tilde_beams_" 
prepended to the file name. Relative paths are allowed.
"""
import os
import io
import sys

import numpy
import scipy
import healpy
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

#import matplotlib.pyplot as plt

if __name__ == "__main__":

    datapath = sys.argv[1]
    nside = int(sys.argv[2])
    bpth = os.path.dirname(datapath)
    fnm = os.path.basename(datapath)
    # read in GRASP field data
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
    # setup interpolation grid
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
    # make polarized beams as described in Rosset et al. 2010
    tildeI = numpy.abs(fieldData[:, 0])**2 + numpy.abs(fieldData[:, 1])**2
    tildeQ = numpy.abs(fieldData[:, 0])**2 - numpy.abs(fieldData[:, 1])**2
    tildeU = 2*numpy.real(fieldData[:, 0] * numpy.conj(fieldData[:, 1]))
    tildeV = -2*numpy.imag(fieldData[:, 0] * numpy.conj(fieldData[:, 1]))
    # make plots some noise, I mean plots
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
    plotname = 'polbeamplot_' + fnm
    plotpth = os.path.join(bpth, plotname)
    plt.show()
    #plt.savefig(plotpth)

    # build interpolators
    intrpI = RectBivariateSpline(u, v, tildeI.reshape((ny, nx)))
    intrpQ = RectBivariateSpline(u, v, tildeQ.reshape((ny, nx)))
    intrpU = RectBivariateSpline(u, v, tildeU.reshape((ny, nx)))
    intrpV = RectBivariateSpline(u, v, tildeV.reshape((ny, nx)))
    
    # calculate x and y components of jones vectors
    tildeI = intrpI(uh, vh, grid=False)
    tildeQ = intrpQ(uh, vh, grid=False)
    tildeU = intrpU(uh, vh, grid=False)
    tildeV = intrpV(uh, vh, grid=False)
    
    pltdata = numpy.zeros(12*nside**2)
    pltdata[0:len(tildeU)] = tildeU
    healpy.orthview(pltdata, rot=(0, 90, 0))
    plt.show()
    
    # setup output file
    newfnm = 'hpx_tilde_beams_' + fnm
    finalpth = os.path.join(bpth, newfnm)
    beamData = numpy.vstack([tildeI, tildeQ, tildeU, tildeV]).swapaxes(0,1)
    numpy.savetxt(finalpth, beamData)
    
