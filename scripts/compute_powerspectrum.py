import sys
import healpy
import numpy
import pandas
import matplotlib.pyplot as plt

Ii, Qi, Ui, Vi = numpy.loadtxt(sys.argv[1], unpack=True)
Ip, Qp, Up, Vp = numpy.loadtxt(sys.argv[2], unpack=True)

skyNside = healpy.npix2nside(len(Ii))
beampath = sys.argv[3]
beamNside = int(sys.argv[4])
fwhm = float(sys.argv[5])
# load beam from disk
beamdata = numpy.loadtxt(beampath)
beam = beamdata[:, 0]
beamMap = numpy.zeros(healpy.nside2npix(beamNside), dtype=numpy.float)
beamMap[0:len(beam)] = beam
imnpix = 256
# degrees
imextn = 5.0
# pixels/arcmin
resval = (imextn * 60) / imnpix
beamImage = healpy.gnomview(beamMap, 
    reso=resval, xsize=imnpix, ysize=imnpix,
    rot=(0, 90, 0), 
    flip='geo', 
    return_projected_map=True, no_plot=True)
x = numpy.linspace(-imextn/2.0, imextn/2.0, imnpix, endpoint=True) 
realbeam = beamImage[len(beamImage)//2]
realbeam /= numpy.max(realbeam)

sig = fwhm / 2.35482
sig2 = sig * sig
mockbeam = numpy.exp(-0.5 * ((x**2) / sig2))
mockbeam /= numpy.max(mockbeam)
plt.cla()
plt.plot(x, realbeam, label='beam')
plt.plot(x, mockbeam, label='gaussian beam')
plt.legend()
plt.show()
# build window function
nsidemap = healpy.npix2nside(len(Ii))
wfuncs = healpy.gauss_beam(numpy.deg2rad(fwhm), lmax=2048, pol=True)
pfuncs = healpy.pixwin(skyNside, pol=True)
pfunct = pfuncs[0]
pfuncp = pfuncs[1]
# smooth input maps for comparison
Is, Qs, Us = healpy.smoothing((Ii, Qi, Ui), fwhm=numpy.deg2rad(fwhm), pol=True)
healpy.cartview(Is)
healpy.cartview(Ip)
plt.show()
# calculate power spectra
TTi, EEi, BBi, TEi, EBi, TBi = healpy.anafast((Ii, Qi, Ui), pol=True)
TTp, EEp, BBp, TEp, EBp, TBp = healpy.anafast((Ip, Qp, Up), pol=True)

lmax = min(len(TTi), len(TTp))
ell = numpy.arange(lmax) + 2
wfuncs = wfuncs[0:lmax, ]

plt.plot(ell*(ell+1)*TTi[0:lmax])
plt.plot(ell*(ell+1)*TTp[0:lmax] / (wfuncs[:, 0] * wfuncs[:, 0]))
plt.xlim(0.01, 250)
plt.ylim(0.01, 45000)
plt.show()

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
data.to_csv('ps.csv')
