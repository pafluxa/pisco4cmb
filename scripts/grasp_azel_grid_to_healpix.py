from argparse import ArgumentParser

import os
import io

import math
import numpy
import healpy
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

helpstr = '''
Description
-----------------------------------------------------------------------
This program loads the data from the GRASP file (a columnar file with a 
header), reads in some parameters like the grid size and resolution and 
then the data. It uses StringIO because for speed.  

The program then figures out the healpix pixel coordinates of the full 
map, sets up interpolators for the magnitude and phases (using sin/cos 
so the interpolation doesn't blow up if the phase moves from 360 to 1) 
and finally, rebuilds everything and saves it to disk as an ASCII file
with 4 columns, suitable to be read by Polbeam.load_beam_data_from_txt()

Command-line arguments
----------------------------------------------------------------------

-i (--input-grasp-beam)
    File name of file with the GRASP beam. The file must be in the 
    specified working directory (see 'w' flag). The beam must also be 
    saved as a Spherical Grid of type "Az-over-El".

-o (--output-file)
    File name of the output beam, in HEALPix format. The file will be
    saved in the specified working directory (see 'w' flag). 

-w (--working-directory)
    System path to directory where files will be read/written from/to.

-n (--nside-healpix-beam)
    NSIDE parameter of the output beam. Integer, must be a power of two.

'''

parser = ArgumentParser(prog='graspbeam_to_healpix', description=helpstr)

parser.add_argument('-i', '--input-grasp-beam', dest='input', required=True, type=str)
parser.add_argument('-o', '--output-file', dest='output', required=True, type=str)
parser.add_argument('-w', '--working-directory', dest='workdir', required=True, type=str)
parser.add_argument('-n', '--nside-healpix-beam', dest='nside', required=True, type=int)

def parse_grasp_azoverel_grid_file(datapath, display_header=True, **kwargs):
    """ Parses GRASP Azimuth over Elevation spherical grid file.

    Throws exceptions if the file does not follow the expected format. 
    It does NOT check if the grid is in Az-over-El format.
    
    Returns dictionaries 'gridData' and `gridSpec`
    """

    f = open(datapath, 'r')
    # file starts with comments, which are delimited by `goodStuff`
    goodStuff = "++++"
    for rawline in f:
        line = rawline.strip()
        if display_header:
            print(line)
        if line.strip('\n') == goodStuff:
            break
    
    # ktype must be 1 for files used by TICRA Tools
    ktype = int(f.readline())
    if ktype != 1:
        raise ValueError("[ERROR] KTYPE must be 1.")
    
    # read in nset, icomp, ncom and igrid
    params = f.readline().strip().split()
    if len(params) != 4:
        raise ValueError("[ERROR] This line must have exactly 4 integers.")
    nset, icomp, ncomp, igrid = map(int, params)
    # only 2 components are supported
    if ncomp != 2:
        raise ValueError("[ERROR] Only 2 field components are supported.")

    beamCenters = numpy.zeros((nset, 2), dtype='int')
    for i in range(nset):
        x0y0 = f.readline().strip().split()
        if len(x0y0) != 2:
            print(x0y0)
            raise ValueError("[ERROR] This line should have exactly 2 integers.")
        beamCenters[i, 0] = int(x0y0[0])
        beamCenters[i, 1] = int(x0y0[1])
    
    # read in grid ranges
    gridRanges = numpy.zeros((nset, 4), dtype='float')
    for i in range(nset):
        data = f.readline().strip().split()
        if len(data) != 4:
            raise ValueError("[ERROR] This line should have exactly 4 float.")
        xs, ys, xe, ye = map(float, data)
        gridRanges[i:, ] = xs, ys, xe, ye
    
    # read in grid sizes
    gridSizes = numpy.zeros((nset, 2), dtype='int')
    for i in range(nset):
        data = f.readline().strip().split()
        if len(data) != 3:
            raise ValueError("[ERROR] This line should have exactly 3 integers.")
        nx, ny, klimit = map(int, data)
        if klimit != 0:
            raise ValueError("[ERROR] This grid does not contain all data for all rows, and is not currently supported.")
        gridSizes[i:, ] = nx, ny
    
    # read field data
    fhandler = io.StringIO(f.read())
    beamData = list()
    for i in range(nset):
        nx, ny = gridSizes[i, :]
        maxrows = int(nx * ny)
        fieldData = numpy.loadtxt(fhandler, max_rows = maxrows, **kwargs).view(complex)
        beamData.append(fieldData)
    fhandler.close()
    f.close()
    
    gridSpec = dict()
    gridSpec['beam_centers'] = beamCenters
    gridSpec['grid_ranges'] = gridRanges
    gridSpec['grid_sizes'] = gridSizes
    
    return gridSpec, beamData

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
        
    datapath = args.input
    output = args.output
    nside = args.nside
    workdir = args.workdir
    # files will be looked up in <workdir>/<file>
    datapath = os.path.join(workdir, datapath)
    output = os.path.join(workdir, output)
    # nside must be a power of two
    if not math.log2(nside).is_integer():
        raise ValueError("NSIDE parameter must be a power of 2.")
    
    # read beam data
    grs, fdata = parse_grasp_azoverel_grid_file(datapath)
    nx, ny  = grs['grid_sizes'][0]
    ranges = grs['grid_ranges'][0]
    fieldData = fdata[0]

    # setup coordinates of beam data points (az-alt grid)
    azmin, elmin, azmax, elmax = numpy.radians(ranges)
    az = numpy.linspace(azmin, azmax, nx, endpoint=True)
    el = numpy.linspace(elmin, elmax, ny, endpoint=True)

    # convienence re-naming of variables
    magEco = numpy.abs(fieldData[:, 0])
    phsEco = numpy.angle(fieldData[:, 0])
    phsEco[phsEco < 0] += 2 * numpy.pi
    magEcx = numpy.abs(fieldData[:, 1])
    phsEcx = numpy.angle(fieldData[:, 1])
    phsEcx[phsEcx < 0] += 2 * numpy.pi
    
    # compute cos/sin phases for better interpolation
    cosPhsCo = numpy.cos(phsEco)
    sinPhsCo = numpy.sin(phsEco)
    cosPhsCx = numpy.cos(phsEcx)
    sinPhsCx = numpy.sin(phsEcx)
    
    # build interpolators for co-polarization
    interp_magco = RectBivariateSpline(az, el, magEco.reshape((ny, nx)))
    interp_cphsco = RectBivariateSpline(az, el, cosPhsCo.reshape((ny, nx)))
    interp_sphsco = RectBivariateSpline(az, el, sinPhsCo.reshape((ny, nx)))
    # build interpolators for cross-polarization
    interp_magcx = RectBivariateSpline(az, el, magEcx.reshape((ny, nx)))
    interp_cphscx = RectBivariateSpline(az, el, cosPhsCx.reshape((ny, nx)))
    interp_sphscx = RectBivariateSpline(az, el, sinPhsCx.reshape((ny, nx)))
    
    # setup coordinates to interpolate
    rhomax = numpy.arccos(numpy.cos(azmax) * numpy.cos(elmax))
    maxpix = healpy.ang2pix(nside, rhomax, 2 * numpy.pi)
    pixels = numpy.arange(0, maxpix)
    x, y, z = healpy.pix2vec(nside, pixels)
    elhpx = numpy.arcsin(y)
    # that pi/2 took me a while to figure out...
    azhpx = numpy.pi / 2 - numpy.arctan2(z, -x)
    
    # interpolate over beam extension (co-polarization)
    magco  = interp_magco(azhpx, elhpx, grid=False)
    cphsco = interp_cphsco(azhpx, elhpx, grid=False)
    sphsco = interp_sphsco(azhpx, elhpx, grid=False)
    phsco = numpy.arctan2(sphsco, cphsco)
    # interpolate over beam extension (cross-polarization)
    magcx = interp_magcx(azhpx, elhpx, grid=False)
    cphscx = interp_cphscx(azhpx, elhpx, grid=False)
    sphscx = interp_sphscx(azhpx, elhpx, grid=False)
    phscx = numpy.arctan2(sphscx, cphscx)

    # append zeros to data (beam must be written to disk with all the zeroes)
    zeroes = numpy.zeros(12 * nside * nside - maxpix)
    magco = numpy.concatenate([magco, zeroes])
    phsco = numpy.concatenate([phsco, zeroes])
    magcx = numpy.concatenate([magcx, zeroes])
    phscx = numpy.concatenate([phscx, zeroes])
    
    # make a plot
    healpy.mollview(magco, rot=(0, 90, 0))
    healpy.mollview(magcx, rot=(0, 90, 0))
    plt.show()

    # write to disk. 
    # this step is very slow and should be updated to something faster!
    fo = open(output, 'w')
    for i in range(12 * nside * nside):
        fo.write("{:.6f} {:.6f} {:.6f} {:.6f}\n".format(
            magco[i], phsco[i], magcx[i], phscx[i]))
    fo.close()
