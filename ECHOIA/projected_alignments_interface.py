from cosmosis.datablock import names, option_section
from projected_alignments import *
cosmo = names.cosmological_parameters

def setup(options):
	# section containing w(r, z)
	input_section = options.get_string(option_section, 'input_section_name', default='projected_galaxy_intrinsic')
	# identify wgp (wg+), wpp (w++), or clustering wgg -- will determine what to read from the nicaea output and how to treat
	corr_type = options.get_string(option_section, 'corr_type', default='wgp')
	# n(z) section names; 1=density/shape, 2=shape
	nz_sec1, nz_sec2 = options.get_string(option_section, 'nz_sections', default='test test').split()
	if not corr_type in ['wgp','wpp','wgg']:
		raise TypeError("Must choose from [wgp, wpp, wgg] for corr_type argument")

	return input_section, corr_type, nz_sec1, nz_sec2

def execute(block, config):
	input_section, corr_type, nz_sec1, nz_sec2 = config

	# can generalise this part for multiple samples
	z = block[nz_sec1, 'z']
	nbin = len(z)
	dz = z[1] - z[0]
	nz1 = block[nz_sec1, 'bin_1']
	nz2 = block[nz_sec2, 'bin_1']
	# read Hankel-transformed correlation functions
	if corr_type in ['wgp', 'wgg']:
		w_rz = np.array([block[input_section, f'bin_{i}_{i}'] for i in range(1, len(z)+1)])
	if corr_type == 'wpp':
		w_rz = np.array([block[input_section, f'xiplus_{i}_{i}'] + block[input_section, f'ximinus_{i}_{i}'] for i in range(1, len(z)+1)])
	# the range of 'r_p' will be crazy, but it inherits the (inverse) units of wavenumber k
	# from previous modules, and should be ready for use after cutting away the extremes
	r_p = block[input_section, 'theta']

	# compute window function (Mandelbaum et al. 2011) & integrate over it
	Wz = compute_Wz(z, nz1, nz2)
	w_r = compute_w(Wz, w_rz, nbin, dz)

	# save projected correlations to datablock
	block[input_section, f'{corr_type}_r_1_1'] = w_r
	block[input_section, 'r_p'] = r_p

	return 0

def cleanup(config):
	pass

