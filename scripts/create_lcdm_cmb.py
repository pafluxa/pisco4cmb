import sys
import platform
import os

import matplotlib
from matplotlib import pyplot as plt

import numpy
import numpy as np

import pandas
import healpy

import camb
from camb import model
from camb import initialpower
print('Using CAMB %s installed at %s' % (camb.__version__,os.path.dirname(camb.__file__)))

out_nside = int(sys.argv[1])
workdir = sys.argv[2]

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.01)
pars.set_for_lmax(4000, lens_potential_accuracy=0)
pars.WantTensors = True
pars.DoLensing = False
#calculate results for these parameters
results = camb.get_results(pars)
#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
#plot the total lensed CMB power spectra versus unlensed, and fractional difference
totCL = powers['total']
#Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
#The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
ls = np.arange(totCL.shape[0]) + 2
fig, ax = plt.subplots(1, 3, figsize = (12, 4.2))
ax[0].plot(ls,totCL[:,0], color='k')
ax[0].set_title('TT')
ax[0].set_ylabel(r'$\ell(\ell+1)C_\ell^{TT}/ (2\pi \mu{\rm K}^2)$')
ax[0].set_xlabel(r'$\ell$')
ax[0].set_ylim(0, 6000)
ax[1].plot(ls,totCL[:,1], color='k')
ax[1].set_title(r'$EE$')
ax[1].set_ylabel(r'$\ell(\ell+1)C_\ell^{EE}/ (2\pi \mu{\rm K}^2)$')
ax[1].set_xlabel(r'$\ell$')
ax[1].set_ylim(0, 50)
ax[2].plot(ls,totCL[:,2], color='k')
ax[2].set_title(r'$BB$')
ax[2].set_ylabel(r'$\ell(\ell+1)C_\ell^{BB}/ (2\pi \mu{\rm K}^2)$')
ax[2].set_xlabel(r'$\ell$')
ax[2].set_ylim(0, 1e-3)
for ax in ax.ravel():
     ax.set_xlim([2, 550])
fig.tight_layout()
plt.savefig(os.path.join(workdir, "input_dl.png"))

cl_TT_in = (ls * (ls + 1) / (2 * numpy.pi))**(-1) * totCL[:, 0]
cl_EE_in = (ls * (ls + 1) / (2 * numpy.pi))**(-1) * totCL[:, 1]
cl_BB_in = (ls * (ls + 1) / (2 * numpy.pi))**(-1) * totCL[:, 2]
cl_TE_in = (ls * (ls + 1) / (2 * numpy.pi))**(-1) * totCL[:, 3]

# Create maps using synfast
I,Q,U = healpy.synfast((cl_TT_in, cl_EE_in, cl_BB_in, cl_TE_in), out_nside, pol=True , new=True )
# Poor V, always zero
V = numpy.zeros_like(I)
# Make some noise (plots)
fig_maps = plt.figure(0)
healpy.mollview(I, sub=(1,3,1), fig=fig_maps)
healpy.mollview(Q, sub=(1,3,2), fig=fig_maps)
healpy.mollview(U, sub=(1,3,3), fig=fig_maps)
plt.savefig(os.path.join(workdir, "input_maps.png"))

# Check output CL's are consistent with input
cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB = healpy.anafast((I, Q, U), pol=True, alm=False )
ls = numpy.arange(cl_TT.size)
cl2dl = (ls * (ls + 1) / (2 * numpy.pi))

fig, ax = plt.subplots(1, 3, figsize = (12, 4.2))
ax[0].plot(ls, cl2dl * cl_TT_in[0:len(cl_TT)], color='k', linestyle='--')
ax[0].plot(ls, cl2dl * cl_TT, color='red', alpha=0.7)
ax[0].set_title('TT')
ax[0].set_ylabel(r'$\ell(\ell+1)C_\ell^{TT}/ (2\pi \mu{\rm K}^2)$')
ax[0].set_xlabel(r'$\ell$')
ax[0].set_ylim(0, 6000)
ax[1].plot(ls, cl2dl*cl_EE_in[0:len(cl_TT)], color='k', linestyle='--')
ax[1].plot(ls, cl2dl*cl_EE, color='red', alpha=0.7)
ax[1].set_title(r'$EE$')
ax[1].set_ylabel(r'$\ell(\ell+1)C_\ell^{EE}/ (2\pi \mu{\rm K}^2)$')
ax[1].set_xlabel(r'$\ell$')
ax[1].set_ylim(0, 50)
ax[2].plot(ls, cl2dl*cl_BB_in[0:len(cl_TT)], color='k', linestyle='--')
ax[2].plot(ls, cl2dl*cl_BB, color='red', alpha=0.7)
ax[2].set_title(r'$BB$')
ax[2].set_ylabel(r'$\ell(\ell+1)C_\ell^{BB}/ (2\pi \mu{\rm K}^2)$')
ax[2].set_xlabel(r'$\ell$')
ax[2].set_ylim(0, 1e-3)
for ax in ax.ravel():
     ax.set_xlim([2, 550])
fig.tight_layout()
plt.savefig(os.path.join(workdir, "input_ps.png"))

data = numpy.vstack([I, Q, U, V])
numpy.savetxt('cmb.txt', data.T)

psdata = pandas.DataFrame()
psdata['cl_TT'] = cl_TT_in
psdata['cl_EE'] = cl_EE_in
psdata['cl_BB'] = cl_BB_in
psdata['cl_TE'] = cl_TE_in
psdata.to_csv(os.path.join(workdir, 'cmb_cl.csv'))
