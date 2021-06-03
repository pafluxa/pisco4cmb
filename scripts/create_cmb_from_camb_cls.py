import sys
import healpy
import pylab
import numpy

r_value   = (float)( sys.argv[1] )
out_nside = (int)( sys.argv[2] )

# Read cl's from input file
ell, dl_TT_in, dl_EE_in, dl_BB_in, dl_TE_in = numpy.loadtxt( sys.argv[3] )
pylab.plot(ell, dl_TT_in * 1e12)
pylab.show()
cl2dl = (ell * (ell+1))/(2 * numpy.pi)

# Create maps using synfast
I,Q,U = healpy.synfast((
    ((2 * numpy.pi) / (ell * (ell + 1))) * cl_TT_in,
    ((2 * numpy.pi) / (ell * (ell + 1))) * cl_EE_in,
    ((2 * numpy.pi) / (ell * (ell + 1))) * cl_BB_in,
    ((2 * numpy.pi) / (ell * (ell + 1))) * cl_TE_in), out_nside, pol=True , new=True )
# Poor V, always zero
V = numpy.zeros_like(I)

# Make some noise
fig_maps = pylab.figure(0)
healpy.mollview(I, sub=(1,3,1), fig=fig_maps)
healpy.mollview(Q, sub=(1,3,2), fig=fig_maps)
healpy.mollview(U, sub=(1,3,3), fig=fig_maps)
pylab.show()

# Check output CL's are consistent with input
cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB = healpy.anafast( (I, Q, U), pol=True, alm=False )
ls = numpy.arange( cl_TT.size )

cl_TT *= 1e12
cl_EE *= 1e12
cl_BB *= 1e12
cl_TT_in *= 1e12
cl_EE_in *= 1e12
cl_BB_in *= 1e12
cl2dl = cl2dl[0:len(cl_TT)]

fig_cls = pylab.figure()
pylab.subplot( 131 )
pylab.plot(cl_TT_in)
pylab.plot(cl2dl * cl_TT)
pylab.xlim( 0,300 )

pylab.subplot( 132 )
pylab.plot(cl_EE_in)
pylab.plot(cl2dl * cl_EE)
pylab.xlim( 0,300 )

pylab.subplot( 133 )
pylab.plot(cl_BB_in)
pylab.plot(cl2dl * cl_BB)
pylab.xlim(0,300)
pylab.show()

data = numpy.vstack([I, Q, U, V])
numpy.savetxt('cmb.txt', data.T)
