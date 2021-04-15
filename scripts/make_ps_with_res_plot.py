#from matplotlib import rc_file
#rc_file('./matplotlibrc')  # <-- the file containing your settings
import sys
import pandas

import matplotlib as mpl
from matplotlib import pyplot as plt

from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar

def set_size( width, fraction=1, subplot=[1,1] ):
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
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    #fig_height_in = fig_width_in * golden_ratio
    fig_height_in = fig_width_in * golden_ratio * ( subplot[0]*1.0/subplot[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

width = 700

nice_fonts = {
        # Use LaTex to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
}

mpl.rcParams.update(nice_fonts)
# configure fond to Seri# make x/y-ticks large
#plt.rc('xtick', labelsize='x-large')
#plt.rc('ytick', labelsize='x-large')

# make 3 axes in a row
lmax = 250

data = pandas.read_csv( sys.argv[1] )

ell = data['ell']
ell2 = (ell)*(ell+1)/2.0

# x 1e12 to put in uK
TT  = data['pTT'] * 1e12
EE  = data['pEE'] * 1e12
BB  = data['pBB'] * 1e12

TTo = data['oTT']  * 1e12
EEo = data['oEE']  * 1e12
BBo = data['oBB']  * 1e12

wl_TT = data['wTT']
wl_EE = data['wEE']
wl_BB = data['wBB']

#width = 345 * 2
fig, axes = plt.subplots( 1, 3, figsize=set_size( width,subplot=[1,2] ), sharex=True )

# remove horizontal gap between subplots
plt.subplots_adjust( wspace=.0 )


axTT = axes[0]
axEE = axes[1]
axBB = axes[2]

axTT.set_ylabel( r'$\mu \rm{K}^2$' )
#axBB.set_ylabel( r'$\mu \rm{K}^2$' )

axTT.set_xlabel( r'$\ell$' )
axEE.set_xlabel( r'$\ell$' )
axBB.set_xlabel( r'$\ell$' )

axTT.set_title( r'$C_{\ell}^{TT}$' )
axTT.set_ylim( (-0.05,0.005) )
axTT.set_xlim( (2,lmax) )
#axTT.set_yscale('symlog', linthreshy=0.1)
axTT.set_xscale('log')

axEE.set_title( r'$C_{\ell}^{EE}$' )
axEE.set_xlim( (2,lmax) )
#axEE.set_ylim( (-0.00001,0.00001) )
#axEE.set_yscale('symlog', linthreshy=1e-4)
axEE.set_xscale('log')

axBB.set_title( r'$C_{\ell}^{BB}$' )
axBB.set_xlim( (2,lmax) )
axBB.set_ylim( (-1e-4,1e-4) )
#axBB.set_yscale('symlog', linthreshy=1e-4)
axBB.set_xscale('log')

axTT.plot( (TTo - TT/wl_TT), alpha=1.0, linestyle='--', color='black')
#axTT.plot( ell2*TT/wl_TT , label='PISCO' , alpha=0.4, linestyle= '-', color= 'red' )
axTT.legend()

axEE.plot( (EEo - EE/wl_EE)  , alpha=1.0, linestyle='--', color='black')
#axEE.plot( ell2*EE/wl_EE , label='PISCO' , alpha=0.4, linestyle= '-', color= 'red' )
axEE.legend()

axBB.plot( (BBo - BB/wl_BB), alpha=1.0, linestyle='--', color='black')
#axBB.plot( ell2*BB/wl_BB , label='PISCO' , alpha=0.4, linestyle= '-', color= 'red' )
axBB.legend()

# configure them like this: 
# first axis (TT) has ticks on the left
ax = axes[0]
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.major.formatter._useMathText = True
#ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.major.formatter._useMathText = True
#ax.yaxis.set_minor_locator(  AutoMinorLocator(5) )
#ax.xaxis.set_minor_locator(  AutoMinorLocator(5) )
ax.yaxis.tick_left()
# remove last tick label for the second subplot
xticks = ax.xaxis.get_major_ticks()
xticks[-1].label1.set_visible(False)

# second axis (EE) has ticks on the right
ax = axes[1]
ax.yaxis.tick_right()
labels = [item.get_text() for item in ax.get_yticklabels()]
empty_string_labels = ['']*len(labels)
ax.set_yticklabels(empty_string_labels)

# remove last tick label for the second subplot
xticks = ax.xaxis.get_major_ticks()
xticks[-1].label1.set_visible(False)
#yticks = ax.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)
#yticks[ 0].label1.set_visible(False)

# third axis (BB) has ticks on the right
ax = axes[2]
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.major.formatter._useMathText = True
#ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.major.formatter._useMathText = True
#ax.yaxis.set_minor_locator(  AutoMinorLocator(5) )
#ax.xaxis.set_minor_locator(  AutoMinorLocator(5) )
ax.yaxis.tick_right()

plt.savefig( "ps.pdf" )
plt.show()
    
