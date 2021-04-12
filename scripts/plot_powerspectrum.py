import healpy
import numpy
import matplotlib.pyplot as plt

I1, Q1, U1, V1 = numpy.loadtxt('maps_input.txt', unpack=True)

I2, Q2, U2 = numpy.loadtxt('maps_output.txt', unpack=True)
# manual normalization by beam solid angle
I2 = I2/(778.241*1e-6)**0.5
Q2 = Q2/(778.241*1e-6)**0.5
U2 = U2/(778.241*1e-6)**0.5

cells = numpy.loadtxt('./data/cls/lcdm_cls_r=0.1000.dat')
TTRef = cells[1]
EERef = cells[2]
BBRef = cells[3]
BBlmax = len(BBRef)

Is, Qs, Us = healpy.smoothing((I1, Q1, U1), fwhm=numpy.deg2rad(1.0), pol=True)

print(numpy.sum(I2)/numpy.sum(Is))
print(numpy.sum(Q2)/numpy.sum(Qs))
print(numpy.sum(U2)/numpy.sum(Us))

TT1, EE1, BB1, TE1, EB1, TB1 = healpy.anafast((Is, Qs, Us), pol=True)
TT2, EE2, BB2, TE2, EB2, TB2 = healpy.anafast((I2, Q2, U2), pol=True)
ell = numpy.arange(len(TT1))

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
