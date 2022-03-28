from cosmosis.datablock import names, option_section
from scipy.interpolate import interp2d, interp1d
import numpy as np
cosmo = names.cosmological_parameters


def setup(options):
	# section containing IA power spectra
    power_section = options.get_string(option_section,'power_section',default='matter_intrinsic_power')
	# section containing a redshift baseline
    nz_section = options.get(option_section,'nz_section')

    return power_section, nz_section

def execute(block, config):
	power_section, nz_section = config

	# load from datablock
	k_h = block[power_section,'k_h']
	p_k = block[power_section,'p_k']
	z_pk = block[power_section, 'z']
	z = block[nz_section, 'z']

	# some options below in case one encounters ringing/aliasing effects
	# in Hankel transformation; upsampling, extrapolation, and zero-padding
	# might help to stabilise the integrations

	# limit scales?
	#cut = (k_h > 1e-4) & (k_h < 1e3)
	#k_h, p_k = k_h[cut], p_k[:, cut]

	# define new k range?
	#k_h_ = np.logspace(np.log10(k_h.min()), np.log10(k_h.max()), 600)
	k_h_ = k_h.copy()

	# interpolate to sample redshift coordinates
	p_k_new = interp2d(k_h, z_pk, p_k, kind='cubic', bounds_error=False, fill_value=0.)(k_h_, z)

	# pad with zeros?
	#nzeros = 10
	#logk = np.log10(k_h_)
	#dlogk = np.diff(logk)[0]
	#low_k = (logk[0] - dlogk * np.arange(1, nzeros+1))[::-1]
	#high_k = logk[-1] + dlogk * np.arange(1, nzeros+1)
	#k_h_ = np.hstack((10**low_k, k_h_, 10**high_k))
	#p_k_new = np.array([np.hstack(([0]*nzeros, p_k_new[i], [0]*nzeros)) for i in range(len(p_k_new))])

	block[power_section,'ell'] = k_h_
	block[power_section,'nbin'] = len(z)
	for i in range(len(p_k_new)):
		block[power_section, 'bin_%s_%s'%(i+1,i+1)] = p_k_new[i]
		if any(np.isnan(p_k_new[i])):
			print('NaNs in power spectrum?')
			import pdb; pdb.set_trace()

	return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass

