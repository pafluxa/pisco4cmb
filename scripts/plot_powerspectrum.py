import sys
import healpy
import numpy
import matplotlib.pyplot as plt

I1, Q1, U1, V1 = numpy.loadtxt(sys.argv[1], unpack=True)
I2, Q2, U2, V2 = numpy.loadtxt(sys.argv[2], unpack=True)
beamNside = int(sys.argv[3])
tI, tQ, tU, tV = beamData = numpy.loadtxt(sys.argv[4], unpack=True)

# compute solid angle of beam
gain = numpy.max(tI)
tI = tI/gain
sa = numpy.sum(tI) * 4 * numpy.pi/(12*beamNside**2)
print("beam solid angle is {:2.2f} ustrad".format(sa*1e6))

cells = numpy.loadtxt('./data/cls/lcdm_cls_r=0.0100.dat')
TTRef = cells[1]
EERef = cells[2]
BBRef = cells[3]
BBlmax = len(BBRef)
# window function
nsidemap = healpy.npix2nside(len(I2))
wfuncs = healpy.gauss_beam(numpy.deg2rad(1.50), lmax=2048, pol=True)
pfunT, pfunP = healpy.pixwin(nsidemap, pol=True, lmax=2048)
print(pfunT)

# normalize by solid angle
I2 *= (nsidemap / 2048) * (sa)**0.5 / gain
Q2 *= (nsidemap / 2048) * (sa)**0.5 / gain
U2 *= (nsidemap / 2048) * (sa)**0.5 / gain
                     

TT1, EE1, BB1, TE1, EB1, TB1 = healpy.anafast((I1, Q1, U1), pol=True)
TT2, EE2, BB2, TE2, EB2, TB2 = healpy.anafast((I2, Q2, U2), pol=True)
ell = numpy.arange(len(TT1)) + 2
# correct PISCO power spectra using the window function of a circular
# gaussian beam of FWHM = 1.1 deg
wfuncs = wfuncs[0:len(TT1), ]
pfunT = pfunT[0:len(TT1), ]
pfunP = pfunP[0:len(TT1), ]

TT2 *= (1./(wfuncs[:, 0])**2 * 1/pfunT)
EE2 *= (1./(wfuncs[:, 1])**2 * 1/pfunP)
BB2 *= (1./(wfuncs[:, 2])**2 * 1/pfunP)

fig, axes = plt.subplots(1, 3, \
    sharex=True, figsize=(9, 5.7/2.0))
# plot stuff
axes[0].plot(ell*(ell+1)*TT1, \
    color='blue', linestyle='dashed', label='TT', alpha=0.7)
axes[0].plot(ell*(ell+1)*TT2, \
    color='red', linestyle='dashdot', label='TT (pisco)', alpha=0.7)
axes[0].plot(TTRef[0:760], \
    color='black', linestyle='dashed', label='TT (ref)', alpha=0.7)

# EE
axes[1].plot(ell*(ell+1)*EE1, \
    color='blue', linestyle='dashed', label='EE', alpha=0.7)
axes[1].plot(ell*(ell+1)*EE2, \
    color='red', linestyle='dashdot', label='EE (pisco)', alpha=0.7)
axes[1].plot(EERef[0:760], \
    color='black', linestyle='dashed', label='EE (ref)', alpha=0.7)
# BB
axes[2].plot(ell*(ell+1)*BB1, \
    color='blue', linestyle='dashed', label='BB', alpha=0.7)
axes[2].plot(ell*(ell+1)*BB2, \
    color='red', linestyle='dashdot', label='BB (pisco)', alpha=0.7)
axes[2].plot(BBRef[0:760], \
    color='black', linestyle='dashed', label='BB (ref., r=0.001)', alpha=0.7)

idx2sp = {0:'TT', 1:'EE', 2:'BB'}
for idx, ax in enumerate(axes):
    ax.set_xlim((2, 550))
    if idx == 0:
        ax.set_ylim((1e-12, 5e-9))
    if idx == 1:
        ax.set_ylim((1e-16, 1e-10))
    if idx == 2:
        ax.set_ylim((1e-20, 1e-13))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('ell')
    ax.set_ylabel('power')

plt.tight_layout()
plt.show()
#plt.savefig("ps.png")
