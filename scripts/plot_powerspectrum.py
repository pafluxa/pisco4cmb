#from matplotlib import rc_file
#rc_file('./matplotlibrc')  # <-- the file containing your settings
import os
import sys
import pandas
import numpy
import healpy

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here

def set_size( width, fraction=1.2, subplot=[1,1] ):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    _w = fig_width_pt * inches_per_pt
    fig_width_in = _w * fraction
    # Figure height in inches
    #fig_height_in = fig_width_in * golden_ratio
    fig_height_in = _w * golden_ratio * ( subplot[0]*1.0/subplot[1]) 

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

width = 600

nice_fonts = {
        # Use LaTex to write all text
        "text.usetex": False,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 11,
        "font.size": 11,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
}

mpl.rcParams.update(nice_fonts)
# configure fond to Seri# make x/y-ticks large

# make 3 axes in a row
lmax = 250

data = pandas.read_csv(sys.argv[1])
cldata = pandas.read_csv(sys.argv[2])
workdir = sys.argv[3]

clT = cldata['cl_TT'] 
clE = cldata['cl_EE'] 
clB = cldata['cl_BB'] 

pi = numpy.pi
clT = clT[0:lmax+1]
clE = clE[0:lmax+1]
clB = clB[0:lmax+1]

# x 1e12 to put in uK
ell = data['ell']
TT  = data['TT_out'][0:lmax+1]
EE  = data['EE_out'][0:lmax+1]
BB  = data['BB_out'][0:lmax+1]

TTo = data['TT_in'][0:lmax+1]
EEo = data['EE_in'][0:lmax+1]
BBo = data['BB_in'][0:lmax+1]

wl_TT = data['wl_TT'][0:lmax+1]
wl_EE = data['wl_EE'][0:lmax+1]
wl_BB = data['wl_BB'][0:lmax+1]

ell2 = (ell * (ell + 1)) / (2 * pi)

#width = 345 * 2
fig, axes = plt.subplots(1, 3, figsize=set_size( width,subplot=[1,2] ), sharex=True)
plt.subplots_adjust( wspace=.28, bottom=0.15 )

axTT = axes[0]
axEE = axes[1]
axBB = axes[2]

axTT.set_xlabel( r'$\ell$' )
axEE.set_xlabel( r'$\ell$' )
axBB.set_xlabel( r'$\ell$' )

axTT.set_title( r'$ D_{\ell}^{TT}$' )
axTT.set_xlabel( r'$\ell$' )
axTT.set_ylabel( r'$\mu \rm{K}^2$' )
axTT.set_ylim( (100, 7000) )
axTT.set_xlim( (2, lmax) )
axTT.ticklabel_format(axis='y', style='sci')
axTT.yaxis.major.formatter.set_powerlimits((0,1))

axEE.set_title( r'$ D_{\ell}^{EE}$' )
axEE.set_xlim( (2, lmax) )
axEE.set_ylim( (0, 2.0) )
#axEE.set_yscale('log', linthreshy=1e-1)
axEE.ticklabel_format(axis='y', style='sci')


axBB.set_title( r'$ D_{\ell}^{BB}$' )
axBB.set_xlim( (2, lmax) )
axBB.set_ylim( (0, 2*1e-3) )
#axBB.set_yscale('symlog', linthreshy=1e-3)
axBB.ticklabel_format(axis='y', style='sci')
axBB.yaxis.major.formatter.set_powerlimits((0,1))

axTT.plot( ell2 * clT, alpha=0.8, label='ref', linestyle='dashed', color='black')
axTT.plot( ell2 * (TTo / wl_TT**2), alpha=0.7, label= 'in', linestyle='dotted', color='green')
axTT.plot( ell2 * (TT / (wl_TT**2)), alpha=0.5, label='out', linestyle= 'solid', color= 'red' )
axTT.legend()

axEE.plot( ell2*clE, alpha=0.8, label='ref', linestyle='dashed', color='black')
axEE.plot( ell2*(EEo / wl_EE**2) ,     alpha=0.7, label= 'in', linestyle='dotted', color='green')
axEE.plot( ell2*(EE / (wl_EE**2)), alpha=0.5, label='out', linestyle= 'solid', color= 'red' )
axEE.legend()

axBB.plot( ell2 * clB, alpha=0.8, label='ref', linestyle='dashed', color='black')
axBB.plot( ell2 * (BBo / wl_BB**2),     alpha=0.7, label= 'in', linestyle='dotted', color='green')
axBB.plot( ell2 * (BB / (wl_BB**2)), alpha=0.5, label='out', linestyle= 'solid', color= 'red' )
axBB.legend()

# configure them like this: 
# first axis (TT) has ticks on the left
ax = axes[0]
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.major.formatter._useMathText = True
ax.yaxis.major.formatter._useMathText = True
ax.yaxis.set_minor_locator(  AutoMinorLocator(5) )
ax.xaxis.set_minor_locator(  AutoMinorLocator(5) )

# second axis (EE) has ticks on the right
ax = axes[1]
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.major.formatter._useMathText = True
ax.yaxis.major.formatter._useMathText = True
ax.xaxis.set_minor_locator(  AutoMinorLocator(2) )
ax.yaxis.set_minor_locator(  AutoMinorLocator(5) )

ax = axes[2]
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.major.formatter._useMathText = True
ax.yaxis.major.formatter._useMathText = True
ax.yaxis.set_minor_locator(  AutoMinorLocator(5) )
ax.xaxis.set_minor_locator(  AutoMinorLocator(5) )

fig.tight_layout()

plt.savefig(os.path.join(workdir, "ps.png"))
plt.show()

    
