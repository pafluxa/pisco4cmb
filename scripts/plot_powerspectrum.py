import healpy
import numpy
import matplotlib.pyplot as plt

# get window function of appropiate beam
Bl_T, Bl_E, Bl_B, Bl_TE = \
    healpy.gauss_beam(fwhm=numpy.deg2rad(1.0), lmax=1024, pol=True).T
pwT, pwP = healpy.pixwin(256, pol=True, lmax=2014)

I1, Q1, U1, V1 = numpy.loadtxt('maps_input.txt', unpack=True)
I2, Q2, U2 = numpy.loadtxt('maps_output.txt', unpack=True)

# normalize
Is, Qs, Us = healpy.smoothing((I1, Q1, U1),\
    fwhm=numpy.deg2rad(1.0),
    pol=True)
pwr1 = numpy.sum(Is) + numpy.sqrt(numpy.sum(Qs)**2 + numpy.sum(Us)**2)
pwr2 = numpy.sum(I2) + numpy.sqrt(numpy.sum(Q2)**2 + numpy.sum(U2)**2)
r = pwr1/pwr2
I2 *= r
Q2 *= r
U2 *= r

#healpy.mollview(Us, title='U smooth')
#healpy.mollview(U2, title='U pisco')
#plt.show()

TT1, EE1, BB1, TE1, EB1, TB1 = healpy.anafast((Is, Qs, Us), pol=True)
TT2, EE2, BB2, TE2, EB2, TB2 = healpy.anafast((I2, Q2, U2), pol=True)
ell_max = len(TT1)
# un-smooth using analytic window function
TT1 /= Bl_T[:ell_max]**2
EE1 /= Bl_E[:ell_max]**2
BB1 /= Bl_B[:ell_max]**2
TT2 /= (Bl_T[:ell_max]/pwT[:ell_max])**2
EE2 /= (Bl_E[:ell_max]/pwP[:ell_max])**2
BB2 /= (Bl_B[:ell_max]/pwP[:ell_max])**2

#TT2 /= pixwin_T**2
#EE2 /= pixwin_P**2
#BB2 /= pixwin_P**2
ell = numpy.arange(len(TT1))

fig, axes = plt.subplots(1, 3, \
    sharex=True, sharey=True, figsize=(9, 5.7/2.0))
# plot stuff
axes[0].plot(ell*(ell+1)/2.0*TT1, \
    color='blue', linestyle='dashed', label='TT', alpha=0.7)
axes[0].plot(ell*(ell+1)/2.0*TT2, \
    color='red', linestyle='dashdot', label='TT (pisco)', alpha=0.7)
# EE
axes[1].plot(ell*(ell+1)/2.0*EE1, \
    color='blue', linestyle='dashed', label='EE', alpha=0.7)
axes[1].plot(ell*(ell+1)/2.0*EE2, \
    color='red', linestyle='dashdot', label='EE (pisco)', alpha=0.7)
# BB
axes[2].plot(ell*(ell+1)/2.0*BB1, \
    color='blue', linestyle='dashed', label='BB', alpha=0.7)
axes[2].plot(ell*(ell+1)/2.0*BB2, \
    color='red', linestyle='dashdot', label='BB (pisco)', alpha=0.7)

for idx, ax in enumerate(axes):
    ax.set_xlim((2, 750))
    ax.set_ylim((5e-21, 1e-9))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('ell')
    ax.set_ylabel('power')

fig.tight_layout()
plt.show()
