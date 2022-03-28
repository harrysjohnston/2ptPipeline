import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
# can replace this with e.g. CAMB transfer function
# for greater accuracy of chi(z) computation, but
# will be a bit slower - for user to decide
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
					  n_s=0.96, sigma8=0.8, m_nu=0.06,
					  transfer_function='bbks')
_z = np.linspace(0, 10, 10000)
sf = 1. / (1. + _z)
chi = ccl.comoving_radial_distance(cosmo, sf)
comov = interp1d(_z, chi, bounds_error=0, fill_value=0.)

def fivept_stencil(func, x, h):
	# returns f'(x), via 5pt stencil, for grid-spacing h
	return (-func(x+2*h)+8*func(x+h)-8*func(x-h)+func(x-2*h))/(12*h)

def compute_Wz(z, nofz_1, nofz_2):
	# Wz = [p^2 / X^2*X'] / int[p^2 / X^2*X' dz] -- see Mandelbaum et al., 2011

	# compute p(z) = unconditional pdf
	pz_1 = nofz_1/sum(nofz_1)
	pz_2 = nofz_2/sum(nofz_2)
	assert pz_1.shape==z.shape, "p(z) vs. z mismatch"
	assert pz_2.shape==z.shape, "p(z) vs. z mismatch"

	# compute X(z) = comoving coordiante
	Xz = comov(z)
	Xz2 = Xz**2

	# compute X'(z) = first deriv.
	h = z[1] - z[0]
	Xprime = fivept_stencil(comov, z, h)

	# combine & integrate (Riemann sum) over z
	Wz_nom = np.nan_to_num((pz_1 * pz_2) / (Xz2 * Xprime))
	Wz_dom = np.sum(Wz_nom)*h
	Wz = Wz_nom/Wz_dom

	return Wz

def compute_w(Wz, w_rz, nbin, dz):
	# Riemann sum over W(z) for w(r,z) -> w(r)
	for i in range(nbin):
		w_rz[i] = w_rz[i] * Wz[i]
	w_r = np.sum(w_rz, axis=0) * dz
	return w_r

