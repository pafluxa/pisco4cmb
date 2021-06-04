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
    cxpolar = fieldData[:, 0]
    copolar = fieldData[:, 1]
    fhandler.close()
    f.close()
    # setup interpolation grid
    nx, ny, _ = gridsizes
    umin, vmin, umax, vmax = ranges
    u = numpy.linspace(umin, umax, nx, endpoint=True)
    v = numpy.linspace(vmin, vmax, ny, endpoint=True)
    # make polarized beams as described in Rosset et al. 2010
    tildeI = numpy.abs(copolar)**2 + numpy.abs(cxpolar)**2
    tildeQ = numpy.abs(copolar)**2 - numpy.abs(cxpolar)**2
    tildeU = 2*numpy.real(copolar * numpy.conj(cxpolar))
    tildeV = -2*numpy.imag(copolar * numpy.conj(cxpolar))
    
    #''' uncomment to plot beams as written to disk by GRASP
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    imI = axes[0].imshow(tildeI.reshape((ny, nx)),
        extent=(umin, umax, vmin, vmax), origin='lower')
    axes[0].set_xlim((-0.05, 0.05))
    axes[0].set_ylim((-0.05, 0.05))
    axes[0].set_xlabel('az (rad)')
    axes[0].set_ylabel('el (rad)')
    axes[0].set_title("tilde I (GRASP)")
    plt.colorbar(imI, ax=axes[0])
    imQ = axes[1].imshow(tildeQ.reshape((ny, nx)),
        extent=(umin, umax, vmin, vmax), origin='lower')
    axes[1].set_xlim((-0.05, 0.05))
    axes[1].set_ylim((-0.05, 0.05))
    axes[1].set_xlabel('az (rad)')
    axes[1].set_ylabel('el (rad)')
    axes[1].set_title("tilde Q (GRASP)")
    plt.colorbar(imQ, ax=axes[1])
    imU = axes[2].imshow(tildeU.reshape((ny, nx)),
        extent=(umin, umax, vmin, vmax), origin='lower')
    axes[2].set_xlim((-0.05, 0.05))
    axes[2].set_ylim((-0.05, 0.05))
    axes[2].set_xlabel('az (rad)')
    axes[2].set_ylabel('el (rad)')
    axes[2].set_title("tilde U (GRASP)")
    plt.colorbar(imU, ax=axes[2])
    fig.tight_layout()
    plotname = 'polbeamplot_' + fnm
    plotpth = os.path.join(bpth, plotname)
    #plt.show()
    #plt.savefig(plotpth)
    #'''
    
    # setup coordinates for interpolation
    maxPix = healpy.ang2pix(nside, numpy.deg2rad(5.0), 0.0)
    pixels = numpy.arange(0, maxPix)
    rho, sigma = healpy.pix2ang(nside, pixels)
    # build interpolation grid coordinates
    # flip the sign of vh to account for GRASP using a different
    # coordinate system than PISCO. In particular, looking out at the 
    # sky U points East and V points North. On the other hand, our 
    # Healpix beam has it's theta_hat unit vector pointing South at 
    # psi = 0 and East at psi = 90
    uh = numpy.sin(rho)*numpy.sin(sigma)
    vh = -numpy.sin(rho)*numpy.cos(sigma)
    # build interpolators
    intrpI = RectBivariateSpline(v, u, tildeI.reshape((ny, nx)))
    intrpQ = RectBivariateSpline(v, u, tildeQ.reshape((ny, nx)))
    intrpU = RectBivariateSpline(v, u, tildeU.reshape((ny, nx)))
    intrpV = RectBivariateSpline(v, u, tildeV.reshape((ny, nx)))
    # calculate polarized beams
    tildeI = intrpI(vh, uh, grid=False)
    tildeQ = intrpQ(vh, uh, grid=False)
    tildeU = intrpU(vh, uh, grid=False)
    tildeV = intrpV(vh, uh, grid=False)
    # setup plotting of polarized beams
    pltdata = numpy.zeros(12*nside**2)
    extndeg = numpy.asarray(
        (200.0/2.0, -200.0/2.0, -200/2.0, 200/2.0))/60.0 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    pltdata[0:len(tildeI)] = tildeI
    tildeIp = healpy.gnomview(pltdata, 
        rot=(0, 90, 0), 
        reso=2, xsize=400, ysize=400,
        return_projected_map=True, no_plot=True)
        
    imI = axes[0].imshow(tildeIp, extent=extndeg, origin='lower')
    axes[0].set_xlim((-1.0, 1.0))
    axes[0].set_ylim((-1.0, 1.0))
    axes[0].set_xlabel('Az (deg)')
    axes[0].set_ylabel('El (deg)')
    axes[0].set_title("tilde I")
    plt.colorbar(imI, ax=axes[0])

    pltdata[0:len(tildeQ)] = tildeQ
    tildeQp = healpy.gnomview(pltdata, 
        rot=(0, 90, 0),
        reso=2, xsize=400, ysize=400,
        return_projected_map=True, no_plot=True)
    imQ = axes[1].imshow(tildeQp, extent=extndeg, origin='lower')
    axes[1].set_xlim((-1.0, 1.0))
    axes[1].set_ylim((-1.0, 1.0))
    axes[1].set_xlabel('Az (deg)')
    axes[1].set_ylabel('El (deg)')
    axes[1].set_title("tilde Q")
    plt.colorbar(imQ, ax=axes[1])

    pltdata[0:len(tildeU)] = tildeU
    tildeUp = healpy.gnomview(pltdata, 
        rot=(0, 90, 0), 
        reso=2, xsize=400, ysize=400,
        return_projected_map=True, no_plot=True)
    imU = axes[2].imshow(tildeUp, extent=extndeg, origin='lower')
    axes[2].set_xlim((-1.0, 1.0))
    axes[2].set_ylim((-1.0, 1.0))
    axes[2].set_xlabel('Az (deg)')
    axes[2].set_ylabel('El (deg)')
    axes[2].set_title("tilde U")
    plt.colorbar(imU, ax=axes[2])
    fig.tight_layout()
    plotname = 'polbeamplot_' + fnm
    plotpth = os.path.join(bpth, plotname)
    #plt.show()
    
    # write polarized beams to disk
    newfnm = 'hpx_tilde_beams_' + fnm
    finalpth = os.path.join(bpth, newfnm)
    beamData = numpy.vstack([tildeI, tildeQ, tildeU, tildeV]).swapaxes(0,1)
    numpy.savetxt(finalpth, beamData)
    
