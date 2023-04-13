import os
import sys
import healpy
import numpy
import pandas

Ii, Qi, Ui, Vi = numpy.loadtxt(sys.argv[1], unpack=True)
Ip, Qp, Up, Vp = numpy.loadtxt(sys.argv[2], unpack=True)
fwhm = numpy.deg2rad(float(sys.argv[3]))
output_path = sys.argv[4]

skyNside = healpy.npix2nside(len(Ii))

# build window function
nsidemap = healpy.npix2nside(len(Ii))
wfuncs = healpy.gauss_beam(fwhm, lmax=2048, pol=True)
pfuncs = healpy.pixwin(skyNside, pol=True)
pfunct = pfuncs[0]
pfuncp = pfuncs[1]

# smooth input maps for comparison
Is, Qs, Us = healpy.smoothing((Ii, Qi, Ui), fwhm=fwhm, pol=True)

# calculate power spectra
TTi, EEi, BBi, TEi, EBi, TBi = healpy.anafast((Is, Qs, Us), pol=True)
TTp, EEp, BBp, TEp, EBp, TBp = healpy.anafast((Ip, Qp, Up), pol=True)

lmax = min(len(TTi), len(TTp))
ell = numpy.arange(lmax) + 2
wfuncs = wfuncs[0:lmax, ]

data = pandas.DataFrame()
data['ell'] = ell
data['TT_out'] = TTp
data['EE_out'] = EEp
data['BB_out'] = BBp
data['TT_in'] = TTi
data['EE_in'] = EEi
data['BB_in'] = BBi
data['wl_TT'] = wfuncs[:, 0]
data['wl_EE'] = wfuncs[:, 1]
data['wl_BB'] = wfuncs[:, 2]
data.to_csv(os.path.join(output_path, 'ps.csv'))
