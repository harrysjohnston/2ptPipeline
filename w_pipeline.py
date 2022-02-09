print("\n2-point correlation pipeline -- Harry Johnston 2019")
print("Work in progress!")
print("\nNotes:")
print("meanr and meanlogr in outputs are preliminary and untested (30/04/21)")
print("Also, shot/shape-noise for projected correlations require further testing,")
print("so do not trust these implicitly. Shot noise for w(theta) should be fine.")
print("h.s.johnston@uu.nl\n")
import os
import gc
import sys
import pickle
import treecorr
import argparse
import skyknife
import numpy as np
import configparser
from os import mkdir, makedirs
from tqdm import tqdm
from os.path import isdir, join, expandvars, basename, dirname
from astropy.io import fits, ascii
from astropy.table import Table
cp = configparser.ConfigParser()
nn = lambda x: x[~np.isnan(x) & ~np.isinf(x)]

def idmatch(c1, c2, idcol1, idcol2):
	return np.isin(c1[idcol1], c2[idcol2])

def ds_func(cat, rcat, target=10.):
	# downsample size of <rcat> to <target> * size of <cat>
	randnum = np.random.rand(len(rcat))
	cut = randnum < (len(cat)*target) / len(rcat)
	return rcat[cut]

def uhalfcut(arr,q=0):
	# remove the largest half of parameter <arr>
	# or largest quarter, if q=1
	perc50 = np.percentile(nn(arr), 50.)
	perc25 = np.percentile(nn(arr), 25.)
	if q:
		return arr < perc25
	else:
		return arr < perc50

def lhalfcut(arr,q=0):
	# remove the smallest half of parameter <arr>
	# or smallest quarter, if q=1
	perc50 = np.percentile(nn(arr), 50.)
	perc25 = np.percentile(nn(arr), 75.)
	if q:
		return arr > perc25
	else:
		return arr > perc50

def lqcut(arr,q=10.):
	# remove all parameter <arr> less than quantile <q>
	perc = np.percentile(nn(arr), q)
	return arr > perc

def hqcut(arr,q=10.):
	# remove all parameter <arr> greater than quantile <q>
	perc = np.percentile(nn(arr), q)
	return arr < perc

midpoints = lambda x: (x[1:] + x[:-1]) / 2.

class Correlate:
	def __init__(self, args):
		print('\n== Preparing correlations\n')
		cp.read(args.config_file)
		if args.p is not None: # override configuration file arguments on command line: -p <section>.<arg>=value (values containing . or = will not be processed)
			print('== Overriding config arguments from command line:')
			for ap in args.p:
				s = ap.split('.')[0]
				p = ap.split('.')[1].split('=')[0]
				v = ap.split('=')[1]
				if args.verbosity >= 1:
					print('==== %s.%s:->%s'%(s, p, v))
				cp.set(s, p, value=v)
		cpd = cp._sections

		# get catalogue path lists
		paths_data1 = expandvars(cp.get('catalogs', 'data1')).replace(' ', '').replace('\n', '').split('//')
		paths_data2 = expandvars(cp.get('catalogs', 'data2')).replace(' ', '').replace('\n', '').split('//')
		paths_rand1 = expandvars(cp.get('catalogs', 'rand1')).replace(' ', '').replace('\n', '').split('//')
		paths_rand2 = expandvars(cp.get('catalogs', 'rand2')).replace(' ', '').replace('\n', '').split('//')

		# get data/random selection/weighting expressions
		data_cuts1 = cp.get('catalogs', 'data_cuts1').replace(' ', '').replace('\n', '').split('//')
		data_cuts2 = cp.get('catalogs', 'data_cuts2').replace(' ', '').replace('\n', '').split('//')
		rand_cuts1 = cp.get('catalogs', 'rand_cuts1').replace(' ', '').replace('\n', '').split('//')
		rand_cuts2 = cp.get('catalogs', 'rand_cuts2').replace(' ', '').replace('\n', '').split('//')
		data_weights1 = cp.get('catalogs', 'data_weights1').replace(' ', '').replace('\n', '').split('//')
		data_weights2 = cp.get('catalogs', 'data_weights2').replace(' ', '').replace('\n', '').split('//')
		try:
			rand_weights1 = cp.get('catalogs', 'rand_weights1').replace(' ', '').replace('\n', '').split('//')
			rand_weights2 = cp.get('catalogs', 'rand_weights2').replace(' ', '').replace('\n', '').split('//')
		except:
			rand_weights1 = rand_weights2 = ['']

		# get correlation type list
		corr_types = cp.get('catalogs', 'corr_types').replace(' ', '').replace('\n', '').split('//')

		paths_data1, paths_data2, paths_rand1, paths_rand2 = \
			map(lambda x: [i for i in x if i != ''], [paths_data1, paths_data2, paths_rand1, paths_rand2])
		data_cuts1, data_cuts2, rand_cuts1, rand_cuts2, data_weights1, data_weights2, rand_weights1, rand_weights2 = \
			map(lambda x: [i for i in x if i != ''], [data_cuts1, data_cuts2, rand_cuts1, rand_cuts2, data_weights1, data_weights2, rand_weights1, rand_weights2])
		corr_types = [i for i in corr_types if i != '']

		# replace empty arguments with 'none' for cuts, or 'ones' for weights
		if data_cuts1 in [[''], []]:
			if args.verbosity >= 1: print('== No sample(1) cuts (data_cuts1) specified; none')
			data_cuts1 = ['none']
		if rand_cuts1 in [[''], []]:
			if args.verbosity >= 1: print('== No randoms(1) cuts (rand_cuts1) specified; none')
			rand_cuts1 = ['none']
		if data_weights1 in [[''], []]:
			if args.verbosity >= 1: print('== Galaxy(1) weighting (data_weights1) not set; assuming unit weights for all')
			data_weights1 = ['ones']
		if rand_weights1 in [[''], []]:
			if args.verbosity >= 1: print('== Randoms(1) weighting (rand_weights1) not set; assuming unit weights for all')
			rand_weights1 = ['ones']
		if data_cuts2 in [[''], []]:
			if args.verbosity >= 1: print('== No sample(2) cuts (data_cuts2) specified; none')
			data_cuts2 = ['none']
		if rand_cuts2 in [[''], []]:
			if args.verbosity >= 1: print('== No randoms(2) cuts (rand_cuts2) specified; none')
			rand_cuts2 = ['none']
		if data_weights2 in [[''], []]:
			if args.verbosity >= 1: print('== Galaxy(2) weighting (data_weights2) not set; assuming unit weights for all')
			data_weights2 = ['ones']
		if rand_weights2 in [[''], []]:
			if args.verbosity >= 1: print('== Randoms(2) weighting (rand_weights2) not set; assuming unit weights for all')
			rand_weights2 = ['ones']

		self.supported_corrs = ['wth', 'wgp', 'wgg', 'xigg', 'xigp', 'xigk', 'xikk']
		self.always_cross = ['wgp', 'xigp', 'xigk']
		if corr_types in [[''], []]:
			print('== Correlation types (corr_types) not set; assuming angular clustering for all')
			corr_types = ['wth'] * len(paths_data1)
		else:
			corr_types = [ct for ct in corr_types if ct != '']
			assert all([ct in self.supported_corrs for ct in corr_types]), \
					("must specify corr_types by choosing from:"
					 "\n'wth' (angular clustering)"
					 "\n'wgg' (proj. clustering)"
					 "\n'wgp' (proj. density-shear)"
					 "\n'xigg' (3D clustering, currently only supported for XYZ)"
					 "\n'xigp' (3D density-shear, currently only supported for XYZ)"
					 "\n'xikk' (3D scalar-scalar, currently only supported for XYZ)"
					 "\n'xigk' (3D density-scalar, currently only supported for XYZ)"
					 "-- can add more correlations on request")
		angular_corrs = 'wth' in corr_types
		projected_corrs = any(corr in corr_types for corr in ('wgp','wgg'))
		threedim_corrs = any(corr in corr_types for corr in ('xigg','xigp','xigk','xikk'))
		assert sum((angular_corrs, projected_corrs, threedim_corrs)) == 1, "Please use different config files for angular, projected, and 3D correlations"

		# copy paths arguments if no second set of paths specified
		if paths_data2 in [[''], []]:
			paths_data2 = list(paths_data1)
		if paths_rand2 in [[''], []]:
			paths_rand2 = list(paths_rand1)

		# ensure each argument is specified once,
		# to be carried over for all correlations,
		# or for each correlation individually
		ncats = (len(paths_data1), len(paths_data2), len(paths_rand1), len(paths_rand2))
		ncuts = (len(data_cuts1), len(data_cuts2), len(rand_cuts1), len(rand_cuts2))
		nweights = (len(data_weights1), len(data_weights2), len(rand_weights1), len(rand_weights2))
		ncorrtypes = (len(corr_types),)
		set_ncats = set(ncats)
		set_ncuts = set(ncuts)
		set_nweights = set(nweights)
		set_ncorrtypes = set(ncorrtypes)
		if not len(set_ncats) == 1:
			assert set_ncats == {1, max(set_ncats)}, "if not specifying multiple different galaxy catalogs, give only 1 to be used in all correlations"
		if not len(set_ncuts) == 1:
			assert set_ncuts == {1, max(set_ncuts)}, "if not specifying multiple cuts for galaxy catalogs, give only 1 to be used for all catalogs"
		if not len(set_nweights) == 1:
			assert set_nweights == {1, max(set_nweights)}, "if not specifying multiple weights for galaxy catalogs, give only 1 to be used for all catalogs"
		if not len(set_ncorrtypes) == 1:
			assert set_ncorrtypes == {1, max(set_ncorrtypes)}, "if not specifying multiple correlations for galaxy catalogs, give only 1 to be used for all catalogs"

		# carry single-specifications over for all correlations
		if len(paths_data1) == 1:
			paths_data1 = paths_data1 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(paths_data2) == 1:
			paths_data2 = paths_data2 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(paths_rand1) == 1:
			paths_rand1 = paths_rand1 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(paths_rand2) == 1:
			paths_rand2 = paths_rand2 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(data_cuts1) == 1:
			data_cuts1 = data_cuts1 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(data_cuts2) == 1:
			data_cuts2 = data_cuts2 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(rand_cuts1) == 1:
			rand_cuts1 = rand_cuts1 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(rand_cuts2) == 1:
			rand_cuts2 = rand_cuts2 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(data_weights1) == 1:
			data_weights1 = data_weights1 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(data_weights2) == 1:
			data_weights2 = data_weights2 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(rand_weights1) == 1:
			rand_weights1 = rand_weights1 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(rand_weights2) == 1:
			rand_weights2 = rand_weights2 * max(ncats + ncuts + nweights + ncorrtypes)
		if len(corr_types) == 1:
			corr_types = corr_types * max(ncats + ncuts + nweights + ncorrtypes)

		# get output filenames or assign them
		outs = cp.get('output', 'out_corrs').replace(' ', '').replace('\n', '').split('//')
		if outs in [[''], []]:
			print('== Assigning outfile names == corr2_out_XX.txt')
			outs = ['corr2_out_%s.dat' % str(i).zfill(2) for i in range(len(paths_data1))]
		for i in range(len(outs)):
			if not outs[i].endswith('.dat'):
				outs[i] += '.dat'

		# create paths for outputs
		outdir = expandvars(cp.get('output', 'savedir'))
		outfiles = [join(outdir, out) for out in outs]
		for i in range(len(outfiles)):
			if not isdir(dirname(outfiles[i])):
				makedirs(dirname(outfiles[i]))

		# construct list of correlations to loop over
		loop = list(np.arange(len(paths_data1), dtype=int))
		if (args.index is not None and
			args.rindex is not None): # keep some/remove some
			loop = [i for i in loop if loop.index(i) in args.index and loop.index(i) not in args.rindex]
		elif (args.index is None and
			  args.rindex is not None): # just remove some
			loop = [i for i in loop if loop.index(i) not in args.rindex]
		elif (args.index is not None and
			  args.rindex is None): # just choose some
			loop = [i for i in loop if loop.index(i) in args.index]
		if args.randomise:
			np.random.shuffle(loop)

		if angular_corrs:
			tc_config_path = expandvars(cp.get('angular_correlations', 'default'))
			if tc_config_path != '':
				tc_config = treecorr.read_config(tc_config_path)
			else:
				tc_config = {}
			if args.bin_slop is not None:
				tc_config['bin_slop'] = args.bin_slop
			tc_config['num_threads'] = args.num_threads
			tc_config['verbose'] = args.verbosity
			for option in cpd['angular_correlations'].keys():
				if option == 'default': continue
				tc_config[option] = cpd['angular_correlations'][option]
			self.ra_col = tc_config['ra_col']
			self.dec_col = tc_config['dec_col']
			self.ra_units = tc_config['ra_units']
			self.dec_units = tc_config['dec_units']
			self.tc_config = tc_config
			self.coordinates = 'RADEC'

		if projected_corrs or threedim_corrs:
			tc_proj_config = cpd['projected_correlations']
			if args.bin_slop is not None: tc_proj_config['bin_slop'] = args.bin_slop
			tc_proj_config['num_threads'] = args.num_threads
			tc_proj_config['verbose'] = args.verbosity
			self.tc_proj_config = tc_proj_config
			self.gplus_estimator = tc_proj_config['gplus_estimator']
			self.metric = tc_proj_config['metric']
			self.period = tc_proj_config['period'].split()
			if self.period == ['']:
				assert self.metric != 'Periodic', "must give period of box if specifying periodic boundary conditions!"
				self.period = None
				del tc_proj_config['period']
			elif len(self.period) == 1:
				self.xperiod = self.yperiod = self.zperiod = float(self.period[0])
			elif len(self.period) == 3:
				self.xperiod, self.yperiod, self.zperiod = (float(p) for p in self.period)
			else:
				self.xperiod = self.yperiod = self.zperiod = None
				del tc_proj_config['period']
			
			self.ra_col = tc_proj_config['ra_col']
			self.dec_col = tc_proj_config['dec_col']
			self.ra_units = tc_proj_config['ra_units']
			self.dec_units = tc_proj_config['dec_units']
			self.r_col = tc_proj_config['r_col']
			self.rand_ra_col = tc_proj_config['rand_ra_col']
			self.rand_dec_col = tc_proj_config['rand_dec_col']
			self.rand_r_col = tc_proj_config['rand_r_col']
			if self.rand_ra_col == '': self.rand_ra_col = self.ra_col
			if self.rand_dec_col == '': self.rand_dec_col = self.dec_col
			if self.rand_r_col == '': self.rand_r_col = self.r_col
			self.x_col = tc_proj_config['x_col']
			self.y_col = tc_proj_config['y_col']
			self.z_col = tc_proj_config['z_col']
			self.rand_x_col = tc_proj_config['rand_x_col']
			self.rand_y_col = tc_proj_config['rand_y_col']
			self.rand_z_col = tc_proj_config['rand_z_col']
			if self.rand_x_col == '': self.rand_x_col = self.x_col
			if self.rand_y_col == '': self.rand_y_col = self.y_col
			if self.rand_z_col == '': self.rand_z_col = self.z_col

			if self.x_col != '' and self.ra_col != '':
				self.coordinates = 'RADEC'
			elif self.ra_col != '':
				self.coordinates = 'RADEC'
			elif self.x_col != '':
				self.coordinates = 'XYZ'
			else:
				sys.exit('== Column names not specified in config file?')

			self.g1_col = tc_proj_config['g1_col']
			self.g2_col = tc_proj_config['g2_col']
			self.flip_g1 = int(tc_proj_config['flip_g1'])
			self.flip_g2 = int(tc_proj_config['flip_g2'])
			self.fg1 = (1., -1.)[self.flip_g1]
			self.fg2 = (1., -1.)[self.flip_g2]
			self.k1_col = tc_proj_config['k1_col']
			self.k2_col = tc_proj_config['k2_col']
			
			try:
				self.min_rpar = float(tc_proj_config['min_rpar'])
				self.max_rpar = float(tc_proj_config['max_rpar'])
			except ValueError:
				del self.tc_proj_config['min_rpar']
				del self.tc_proj_config['max_rpar']
			try:
				self.nbins_rpar = int(tc_proj_config['nbins_rpar'])
			except:
				del self.tc_proj_config['nbins_rpar']
			try:
				self.rpar_edges = np.array([float(i) for i in tc_proj_config['rpar_edges'].replace(' ','').strip("'").strip('"').split(',')])
#				if self.rpar_edges[0] != 0:
#					self.rpar_edges = np.insert(self.rpar_edges, 0, 0.)
#				if not self.rpar_edges.min() < 0:
#					self.rpar_edges = np.append(self.rpar_edges[::-1]*-1., self.rpar_edges[1:])
#				assert 0 in self.rpar_edges, "Pi bins asymmetric!"
				self.min_rpar = self.rpar_edges.min()
				self.max_rpar = self.rpar_edges.max()
				self.nbins_rpar = len(self.rpar_edges) - 1
				print("== Specified (line-of-sight) Pi-binning (units of r_col):")
				print(self.rpar_edges)
			except:
				if projected_corrs:
					print('== rpar edges not specified; computing from min/max/nbin_rpar')
				self.rpar_edges = None
			self.nbins = int(tc_proj_config['nbins'])
			self.largePi = int(tc_proj_config['largepi'])
			if self.coordinates == 'XYZ' and self.largePi:
				print('== Warning: largePi has no meaning for XYZ coordinates!')
			self.compensated = int(tc_proj_config['compensated'])
			self.random_oversampling = args.down
			self.save_3d = int(tc_proj_config['save_3d'])

		self.treejack = int(cp.get('jackknife', 'treejack', fallback=0))
		if self.treejack != 0:
			self.run_jackknife = 0
			print(f"== Doing TreeCorr jackknife with {self.treejack} patches")
			if self.coordinates == 'XYZ' and not all('xi' in i for i in corr_types):
				print(f'==== TreeCorr jackknife not supported for requested correlations:\n{corr_types}')
				raise ValueError('set jackknife.treejack=0 and use the jackknife.run= argument instead (must externally define a jackknife_ID column)'
								 '\nAvoid using external jackknife_ID when doing periodic boundaries')
			self.treejack_save = cp.get('jackknife', 'treejack_save', fallback=None)
			if self.treejack_save:
				print(f"== Saving/loading TreeCorr jackknife patches to/from {self.treejack_save}")
		else:
			self.treejack_save = None
			self.run_jackknife = int(cp.get('jackknife', 'run', fallback=0))
			if self.run_jackknife == 1:
				print("== Run_jackknife = 1 -- performing N jackknife correlations excluding regions N")
			if self.run_jackknife == 2:
				print("== Run_jackknife = 2 -- performing jackknife before main correlations")
			if self.run_jackknife == 3:
				print("== Run_jackknife = 3 -- performing jackknife after main correlations")
			if self.run_jackknife == 4:
				print("== Run_jackknife = 4 -- collecting jackknife covariance only")
			try:
				self.jackknife_numbers = [int(i) for i in cp.get('jackknife', 'numbers').split(' ')]
				print("== jackknife indices specified:")
				print(self.jackknife_numbers)
			except:
				self.jackknife_numbers = None
		self.paths_data1 = paths_data1
		self.paths_data2 = paths_data2
		self.paths_rand1 = paths_rand1
		self.paths_rand2 = paths_rand2
		self.data_cuts1 = data_cuts1
		self.data_cuts2 = data_cuts2
		self.rand_cuts1 = rand_cuts1
		self.rand_cuts2 = rand_cuts2
		self.data_weights1 = data_weights1
		self.data_weights2 = data_weights2
		self.rand_weights1 = rand_weights1
		self.rand_weights2 = rand_weights2
		self.corr_types = corr_types
		self.outdir = outdir
		self.outfiles = outfiles
		self.loop = loop

	def get_catalog_args(self):
		if self.treejack != 0:
			var_method = 'jackknife'
			if self.treejack_save and os.path.exists(self.treejack_save):
				kw = {'patch_centers':self.treejack_save}
			else:
				kw = {'npatch':self.treejack}
		else:
			var_method = 'shot'
			kw = {'npatch':1}
		return kw, var_method

	def compute_wtheta(self, d1, r1, outfile, auto=True, d2=None, r2=None, wcol1=None, wcol2=None, rwcol1=None, rwcol2=None):
		kw, var_method = self.get_catalog_args()
		rand1 = treecorr.Catalog(ra=r1[self.ra_col], dec=r1[self.dec_col], ra_units=self.ra_units, dec_units=self.dec_units, is_rand=1, w=rwcol1, **kw)
		data1 = treecorr.Catalog(ra=d1[self.ra_col], dec=d1[self.dec_col], ra_units=self.ra_units, dec_units=self.dec_units, w=wcol1, patch_centers=rand1.patch_centers)
		if not auto:
			data2 = treecorr.Catalog(ra=d2[self.ra_col], dec=d2[self.dec_col], ra_units=self.ra_units, dec_units=self.dec_units, w=wcol2, patch_centers=rand1.patch_centers)
			rand2 = treecorr.Catalog(ra=r2[self.ra_col], dec=r2[self.dec_col], ra_units=self.ra_units, dec_units=self.dec_units, is_rand=1, w=rwcol2, patch_centers=rand1.patch_centers)
		if self.treejack != 0 and self.treejack_save and not os.path.exists(self.treejack_save):
			rand1.write_patch_centers(self.treejack_save)

		nn = treecorr.NNCorrelation(self.tc_config, var_method=var_method)
		nr = treecorr.NNCorrelation(self.tc_config, var_method=var_method)
		rr = treecorr.NNCorrelation(self.tc_config, var_method=var_method)
		if auto:
			nn.process(data1)
			rr.process(rand1)
			nr.process(data1, rand1)
			nn.write(outfile, rr=rr, dr=nr, file_type='ASCII', precision=6)
		else:
			rn = treecorr.NNCorrelation(self.tc_config)
			nn.process(data1, data2)
			rr.process(rand1, rand2)
			nr.process(data1, rand2)
			rn.process(rand1, data2)
			nn.write(outfile, rr=rr, dr=nr, rd=rn, file_type='ASCII', precision=6)
		# remove the additional header with correlation details -- these are in the treecorr config file
		os.system("sed -i '' -e '/##/ { d; }' %s"%outfile)

		if self.treejack != 0 and hasattr(nn, 'cov'):
			np.savetxt(outfile.replace('.dat', '.cov'), nn.cov)
			dat = ascii.read(outfile)
			dat['xi_jackknife_err'] = np.diag(nn.cov)**0.5
			dat.write(outfile, format='ascii.fast_commented_header', overwrite=1)

		del data1, rand1, nn, nr, rr
		if not auto:
			del data2, rand2, rn
		gc.collect()

	def compute_wgplus(self, d1, r1, d2, r2, outfile, wcol1=None, wcol2=None, rwcol1=None, rwcol2=None):
		if self.rpar_edges is not None:
			Pi = self.rpar_edges.copy()
		elif not self.largePi:
			Pi = np.linspace(self.min_rpar, self.max_rpar, self.nbins_rpar + 1)
		else: # define new Pi-range, with |Pi_max| = 1.5 * |max_arg|
			dPi = (self.max_rpar - self.min_rpar) / self.nbins_rpar
			Pi = np.arange(self.min_rpar*1.5, self.max_rpar*1.5 + dPi, step=dPi, dtype=float)
		gt_3D = np.zeros([self.nbins_rpar, self.nbins])
		gx_3D = np.zeros([self.nbins_rpar, self.nbins])
		varg_3D = np.zeros([self.nbins_rpar, self.nbins])
		DD_3D = np.zeros([self.nbins_rpar, self.nbins])
		DS_3D = np.zeros([self.nbins_rpar, self.nbins])
		RS_3D = np.zeros([self.nbins_rpar, self.nbins])
		meanr_3D = np.zeros([self.nbins_rpar, self.nbins])
		meanlogr_3D = np.zeros([self.nbins_rpar, self.nbins])

		data1 = treecorr.Catalog(ra=d1[self.ra_col], dec=d1[self.dec_col], ra_units=self.ra_units, dec_units=self.dec_units, w=wcol1, r=d1[self.r_col])
		data2 = treecorr.Catalog(ra=d2[self.ra_col], dec=d2[self.dec_col], ra_units=self.ra_units, dec_units=self.dec_units, w=wcol2, r=d2[self.r_col],
								 g1=d2[self.g1_col] * self.fg1, g2=d2[self.g2_col] * self.fg2)
		rand1 = treecorr.Catalog(ra=r1[self.rand_ra_col], dec=r1[self.rand_dec_col], ra_units=self.ra_units, dec_units=self.dec_units, r=r1[self.rand_r_col], is_rand=1, w=rwcol1)
		rand2 = treecorr.Catalog(ra=r2[self.rand_ra_col], dec=r2[self.rand_dec_col], ra_units=self.ra_units, dec_units=self.dec_units, r=r2[self.rand_r_col], is_rand=1, w=rwcol2)
		varg = treecorr.calculateVarG(data2)


		# find duplicate objects between d1/2 and r1/2 - should replace this with IDs
		# RA coordinates might be repeated in future...
		set_d1_ra = set(d1[self.ra_col][wcol1!=0])
		set_d2_ra = set(d2[self.ra_col][wcol2!=0])
		data_ind_dict_1 = dict((k,i) for i,k in enumerate(d1[self.ra_col][wcol1!=0]))
		data_ind_dict_2 = dict((k,i) for i,k in enumerate(d2[self.ra_col][wcol2!=0]))
		data_intersection = set_d1_ra.intersection(set_d2_ra)
		data_int_ind_1 = [data_ind_dict_1[x] for x in data_intersection]
		data_int_ind_2 = [data_ind_dict_2[x] for x in data_intersection]
		data_int_w1 = wcol1[wcol1!=0][data_int_ind_1]
		data_int_w2 = wcol2[wcol2!=0][data_int_ind_2]

		set_r1_ra = set(r1[self.rand_ra_col][rwcol1!=0])
		set_r2_ra = set(r2[self.rand_ra_col][rwcol2!=0])
		rand_ind_dict_1 = dict((k,i) for i,k in enumerate(r1[self.rand_ra_col][rwcol1!=0]))
		rand_ind_dict_2 = dict((k,i) for i,k in enumerate(r2[self.rand_ra_col][rwcol2!=0]))
		rand_intersection = set_r1_ra.intersection(set_r2_ra)
		rand_int_ind_1 = [rand_ind_dict_1[x] for x in rand_intersection]
		rand_int_ind_2 = [rand_ind_dict_2[x] for x in rand_intersection]
		rand_int_w1 = rwcol1[rwcol1!=0][rand_int_ind_1]
		rand_int_w2 = rwcol2[rwcol2!=0][rand_int_ind_2]

		# get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation
		ng_tot = 1.*data1.sumw*data2.sumw - np.sum(data_int_w1*data_int_w2)
		rr_tot = 1.*rand1.sumw*rand2.sumw - np.sum(rand_int_w1*rand_int_w2)
		rg_tot = 1.*rand1.sumw*data2.sumw
		rgw = ng_tot / rg_tot
		rrw = ng_tot / rr_tot
		rrgw = rg_tot / rr_tot

		# loop over Pi-bins collecting w(rp, Pi[p])
		for p in range(self.nbins_rpar):
			if args.verbosity >= 2: print('== Pi bin #%s'%(p+1))
			if self.largePi and any(abs(Pi[p:p+2]) < self.max_rpar): # skip any |Pi| < max_arg
				continue

			# limit correlation to this Pi-bin
			conf_pi = self.tc_proj_config.copy()
			conf_pi['min_rpar'] = Pi[p]
			conf_pi['max_rpar'] = Pi[p+1]

			ng = treecorr.NGCorrelation(conf_pi)
			rg = treecorr.NGCorrelation(conf_pi)
			ng.process_cross(data1, data2)
			rg.process_cross(rand1, data2)
			ng.varxi = varg
			ng.varxi = varg

			# calculate appropriate normalisation of terms for chosen gplus_estimator
			if self.gplus_estimator == 'PW1': # RDs (rg) norm
				norm1 = rg.weight * rgw
				norm2 = rg.weight
			elif self.gplus_estimator == 'PW2': # RRs (rr) norm
				rr = treecorr.NNCorrelation(conf_pi)
				rr.process_cross(rand1, rand2)
				norm1 = rr.weight * rrw
				norm2 = rr.weight * rrgw
			elif self.gplus_estimator == 'AS': # DDs (ng), RDs (rg) norms
				norm1 = ng.weight
				norm2 = rg.weight

			# subtract randoms correlations
			if self.compensated:
				gt_3D[p] += (ng.xi / norm1) - (rg.xi / norm2)
				gx_3D[p] += (ng.xi_im / norm1) - (rg.xi_im / norm2)
				varg_3D[p] += (ng.varxi / norm1) + (rg.varxi / norm2)
			else: # or not
				gt_3D[p] += ng.xi / norm1
				gx_3D[p] += ng.xi_im / norm1
				varg_3D[p] += ng.varxi / norm1
			DD_3D[p] += ng.npairs
			DS_3D[p] += ng.weight
			RS_3D[p] += rg.weight
			meanr_3D[p] += ng.meanr
			meanlogr_3D[p] += ng.meanlogr

		# compute/save projected statistics
		if self.rpar_edges is None:
			gt = np.sum(gt_3D * (Pi[1] - Pi[0]), axis=0)
			gx = np.sum(gx_3D * (Pi[1] - Pi[0]), axis=0)
		else:
			gt = np.trapz(gt_3D, x=midpoints(np.squeeze(Pi)), axis=0)
			gx = np.trapz(gx_3D, x=midpoints(np.squeeze(Pi)), axis=0)
		varg = np.sum(varg_3D, axis=0)
		DS = np.sum(DS_3D, axis=0)
		RS = np.sum(RS_3D, axis=0)
		meanr = np.sum(meanr_3D, axis=0) / DS
		meanlogr = np.sum(meanlogr_3D, axis=0) / DS
		r = np.column_stack((ng.rnom, meanr, meanlogr))
		output = np.column_stack((r, gt, gx, varg**0.5, DS, RS))
		np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','wgplus','wgcross','noise','DSpairs','RSpairs')))

		# optionally, save the 2D paircounts (3D name is stupid, I know)
		if self.save_3d:
			if args.verbosity >= 1: print("== Saving 3D correlations..")

			# with normalisations for paircounts
			DSntot = data1.ntot * data2.ntot
			if data1.ntot == data2.ntot:
				DSntot -= data2.ntot
			RSntot = rand1.ntot * data2.ntot

			output_dict = {'r':r,'meanr':meanr_3D,'meanlogr':meanlogr_3D,
						   'Pi':Pi,'w3d':gt_3D,'wx3d':gx_3D,'noise3d':varg_3D**0.5,
						   'DS_3d':DS_3D,'RS_3d':RS_3D,'DD_3d':DD_3D,
						   'DSntot':DSntot,'RSntot':RSntot}
			if outfile.endswith('.dat'):
				outfile3d = outfile.replace('.dat', '.p')
			elif '.jk' in outfile:
				return None
				outfile3d = outfile.replace('.jk', '.pjk')
			else:
				print("==== Unrecognised output type -- not saving 3D correlations")
				return None
			pickle.dump(output_dict, open(outfile3d, 'wb'))

		del data1, rand1, data2, rand2, ng, rg
		gc.collect()

	def compute_wgg(self, d1, r1, outfile, auto=True, d2=None, r2=None, wcol1=None, wcol2=None, rwcol1=None, rwcol2=None):
		if self.rpar_edges is not None:
			Pi = self.rpar_edges.copy()
		elif not self.largePi:
			Pi = np.linspace(self.min_rpar, self.max_rpar, self.nbins_rpar + 1)
		else:
			dPi = (self.max_rpar - self.min_rpar) / self.nbins_rpar
			Pi = np.arange(self.min_rpar*1.5, self.max_rpar*1.5 + dPi, step=dPi, dtype=float)
		wgg_3D = np.zeros([self.nbins_rpar, self.nbins])
		varw_3D = np.zeros([self.nbins_rpar, self.nbins])
		DD_3D = np.zeros([self.nbins_rpar, self.nbins])
		DR_3D = np.zeros([self.nbins_rpar, self.nbins])
		RD_3D = np.zeros([self.nbins_rpar, self.nbins])
		RR_3D = np.zeros([self.nbins_rpar, self.nbins])
		meanr_3D = np.zeros([self.nbins_rpar, self.nbins])
		meanlogr_3D = np.zeros([self.nbins_rpar, self.nbins])

		data1 = treecorr.Catalog(ra=d1[self.ra_col], dec=d1[self.dec_col], r=d1[self.r_col], w=wcol1, ra_units=self.ra_units, dec_units=self.dec_units)
		rand1 = treecorr.Catalog(ra=r1[self.rand_ra_col], dec=r1[self.rand_dec_col], r=r1[self.rand_r_col], is_rand=1, w=rwcol1, ra_units=self.ra_units, dec_units=self.dec_units)
		if not auto:
			data2 = treecorr.Catalog(ra=d2[self.ra_col], dec=d2[self.dec_col], r=d2[self.r_col], w=wcol2, ra_units=self.ra_units, dec_units=self.dec_units)
			rand2 = treecorr.Catalog(ra=r2[self.rand_ra_col], dec=r2[self.rand_dec_col], r=r2[self.rand_r_col], is_rand=1, w=rwcol2, ra_units=self.ra_units, dec_units=self.dec_units)

		# loop over Pi-bins collecting w(rp, Pi[p])
		for p in range(self.nbins_rpar):
			if args.verbosity >= 2:	print('== Pi bin #%s'%(p+1))
			if self.largePi and any(abs(Pi[p:p+2]) < self.max_rpar): # skip any |Pi| < max_arg
				continue

			# limit correlation to this Pi-bin
			conf_pi = self.tc_proj_config.copy()
			conf_pi['min_rpar'] = Pi[p]
			conf_pi['max_rpar'] = Pi[p+1]

			nn = treecorr.NNCorrelation(conf_pi)
			rr = treecorr.NNCorrelation(conf_pi)
			nr = treecorr.NNCorrelation(conf_pi)
			rn = treecorr.NNCorrelation(conf_pi)

			if auto:
				nn.process_cross(data1, data1)
				rr.process_cross(rand1, rand1)
				nr.process_cross(data1, rand1)
				nn.finalize()
				rr.finalize()
				nr.finalize()
				if self.compensated:
					xi, varxi = nn.calculateXi(rr, dr=nr)
				else:
					xi, varxi = nn.calculateXi(rr)
			else:
				nn.process_cross(data1, data2)
				rr.process_cross(rand1, rand2)
				nr.process_cross(data1, rand2)
				rn.process_cross(rand1, data2)
				nn.finalize()
				rr.finalize()
				nr.finalize()
				rn.finalize()
				if self.compensated:
					xi, varxi = nn.calculateXi(rr, dr=nr, rd=rn)
				else:
					xi, varxi = nn.calculateXi(rr)

			wgg_3D[p] += xi
			varw_3D[p] += varxi
			DD_3D[p] += nn.weight
			DR_3D[p] += nr.weight
			if auto:
				RD_3D[p] += nr.weight
			else:
				RD_3D[p] += rn.weight
			RR_3D[p] += rr.weight
			meanr_3D[p] += nn.meanr * nn.weight
			meanlogr_3D[p] += nn.meanlogr * nn.weight

		# compute/save projected statistics
		if self.rpar_edges is None:
			wgg = np.sum(wgg_3D * (Pi[1] - Pi[0]), axis=0)
		else:
			wgg = np.trapz(wgg_3D, x=midpoints(np.squeeze(Pi)), axis=0)
		varw = np.sum(varw_3D, axis=0)
		DDpair = np.sum(DD_3D, axis=0)
		DRpair = np.sum(DR_3D, axis=0)
		RDpair = np.sum(RD_3D, axis=0)
		RRpair = np.sum(RR_3D, axis=0)
		meanr = np.sum(meanr_3D, axis=0) / DDpair
		meanlogr = np.sum(meanlogr_3D, axis=0) / DDpair
		r = np.column_stack((nn.rnom, meanr, meanlogr))
		output = np.column_stack((r, wgg, varw**0.5, DDpair, DRpair, RDpair, RRpair))
		np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','wgg','noise','DDpairs','DRpairs','RDpairs','RRpairs')))

		# optionally, save the 2D paircounts
		if self.save_3d:
			if args.verbosity >= 1: print("== Saving 3D correlations..")

			# with normalisations
			if auto:
				DDntot = data1.ntot * (data1.ntot - 1)
				RRntot = rand1.ntot * (rand1.ntot - 1)
				DRntot = RDntot = data1.ntot * rand1.ntot
			else:
				DDntot = data1.ntot * data2.ntot
				RRntot = rand1.ntot * rand2.ntot
				DRntot = data1.ntot * rand2.ntot
				RDntot = rand1.ntot * data2.ntot

			output_dict = {'r':r,'meanr':meanr_3D,'meanlogr':meanlogr_3D,
						   'Pi':Pi,'w3d':wgg_3D,'noise3d':varw_3D**0.5,
						   'DD3d':DD_3D,'DR3d':DR_3D,'RD3d':RD_3D,'RR3d':RR_3D,
						   'DDntot':DDntot,'DRntot':DRntot,'RDntot':RDntot,'RRntot':RRntot}
			if outfile.endswith('.dat'):
				outfile3d = outfile.replace('.dat', '.p')
			elif '.jk' in outfile:
				outfile3d = outfile.replace('.jk', '.pjk')
			else:
				print("==== Unrecognised output type -- not saving 3D correlations")
				return None
			pickle.dump(output_dict, open(outfile3d, 'wb'))

		del data1, rand1, nn, nr, rr
		if not auto:
			del data2, rand2, rn
		gc.collect()

	def compute_xigplus_xyz(self, d1, r1, d2, r2, outfile, wcol1=None, wcol2=None, rwcol1=None, rwcol2=None):
		gt = np.zeros(self.nbins)
		gx = np.zeros(self.nbins)
		varg = np.zeros(self.nbins)
		DD = np.zeros(self.nbins)
		DS = np.zeros(self.nbins)
		RS = np.zeros(self.nbins)
		meanr = np.zeros(self.nbins)
		meanlogr = np.zeros(self.nbins)

		data1 = treecorr.Catalog(x=d1[self.x_col], y=d1[self.y_col], z=d1[self.z_col], w=wcol1)
		data2 = treecorr.Catalog(x=d2[self.x_col], y=d2[self.y_col], z=d2[self.z_col], w=wcol2, g1=d2[self.g1_col] * self.fg1, g2=d2[self.g2_col] * self.fg2)
		rand1 = treecorr.Catalog(x=r1[self.rand_x_col], y=r1[self.rand_y_col], z=r1[self.rand_z_col], is_rand=1, w=rwcol1)
		rand2 = treecorr.Catalog(x=r2[self.rand_x_col], y=r2[self.rand_y_col], z=r2[self.rand_z_col], is_rand=1, w=rwcol2)
		varg = treecorr.calculateVarG(data2)

		# find duplicate objects between d1/2 and r1/2
		set_d1_x = set(d1[self.x_col][wcol1!=0])
		set_d2_x = set(d2[self.x_col][wcol2!=0])
		data_ind_dict_1 = dict((k,i) for i,k in enumerate(d1[self.x_col][wcol1!=0]))
		data_ind_dict_2 = dict((k,i) for i,k in enumerate(d2[self.x_col][wcol2!=0]))
		data_intersection = set_d1_x.intersection(set_d2_x)
		data_int_ind_1 = [data_ind_dict_1[x] for x in data_intersection]
		data_int_ind_2 = [data_ind_dict_2[x] for x in data_intersection]
		data_int_w1 = wcol1[wcol1!=0][data_int_ind_1]
		data_int_w2 = wcol2[wcol2!=0][data_int_ind_2]

		set_r1_x = set(r1[self.rand_x_col][rwcol1!=0])
		set_r2_x = set(r2[self.rand_x_col][rwcol2!=0])
		rand_ind_dict_1 = dict((k,i) for i,k in enumerate(r1[self.rand_x_col][rwcol1!=0]))
		rand_ind_dict_2 = dict((k,i) for i,k in enumerate(r2[self.rand_x_col][rwcol2!=0]))
		rand_intersection = set_r1_x.intersection(set_r2_x)
		rand_int_ind_1 = [rand_ind_dict_1[x] for x in rand_intersection]
		rand_int_ind_2 = [rand_ind_dict_2[x] for x in rand_intersection]
		rand_int_w1 = rwcol1[rwcol1!=0][rand_int_ind_1]
		rand_int_w2 = rwcol2[rwcol2!=0][rand_int_ind_2]

		# get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation
		ng_tot = 1.*data1.sumw*data2.sumw - np.sum(data_int_w1*data_int_w2)
		rr_tot = 1.*rand1.sumw*rand2.sumw - np.sum(rand_int_w1*rand_int_w2)
		rg_tot = 1.*rand1.sumw*data2.sumw
		rgw = ng_tot / rg_tot
		rrw = ng_tot / rr_tot
		rrgw = rg_tot / rr_tot

		conf = self.tc_proj_config.copy()
		p_args = dict(xperiod=self.xperiod, yperiod=self.yperiod, zperiod=self.zperiod)
		ng = treecorr.NGCorrelation(conf, **p_args)
		rg = treecorr.NGCorrelation(conf, **p_args)
		ng.process_cross(data1, data2)
		rg.process_cross(rand1, data2)
		ng.varxi = varg

		# calculate appropriate normalisation of terms for chosen gplus_estimator
		if self.gplus_estimator == 'PW1': # RDs (rg) norm
			norm1 = rg.weight * rgw
			norm2 = rg.weight
		elif self.gplus_estimator == 'PW2': # RRs (rr) norm
			rr = treecorr.NNCorrelation(conf)
			rr.process_cross(rand1, rand2)
			norm1 = rr.weight * rrw
			norm2 = rr.weight * rrgw
		elif self.gplus_estimator == 'AS': # DDs (ng), RDs (rg) norms
			norm1 = ng.weight
			norm2 = rg.weight

		# subtract randoms correlations
		if self.compensated:
			gt += (ng.xi / norm1) - (rg.xi / norm2)
			gx += (ng.xi_im / norm1) - (rg.xi_im / norm2)
			varg += (ng.varxi / norm1) + (rg.varxi / norm2)
		else: # or not
			gt += ng.xi / norm1
			gx += ng.xi_im / norm1
			varg += ng.varxi / norm1
		DD += ng.npairs
		DS += ng.weight
		RS += rg.weight
		meanr += ng.meanr
		meanlogr += ng.meanlogr

		r = np.column_stack((ng.rnom, meanr, meanlogr))
		output = np.column_stack((r, gt, gx, varg**0.5, DS, RS))
		np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','xigplus','xigcross','noise','DSpairs','RSpairs')))

		del data1, rand1, data2, rand2, ng, rg
		gc.collect()

	def compute_xigg_xyz(self, d1, r1, outfile, auto=True, d2=None, r2=None, wcol1=None, wcol2=None, rwcol1=None, rwcol2=None):
		kw, var_method = self.get_catalog_args()
		rand1 = treecorr.Catalog(x=r1[self.rand_x_col], y=r1[self.rand_y_col], z=r1[self.rand_z_col], is_rand=1, w=rwcol1, **kw)
		data1 = treecorr.Catalog(x=d1[self.x_col], y=d1[self.y_col], z=d1[self.z_col], w=wcol1, patch_centers=rand1.patch_centers)
		if not auto:
			data2 = treecorr.Catalog(x=d2[self.x_col], y=d2[self.y_col], z=d2[self.z_col], w=wcol2, patch_centers=rand1.patch_centers)
			rand2 = treecorr.Catalog(x=r2[self.rand_x_col], y=r2[self.rand_y_col], z=r2[self.rand_z_col], is_rand=1, w=rwcol2, patch_centers=rand1.patch_centers)
		if self.treejack != 0 and self.treejack_save and not os.path.exists(self.treejack_save):
			rand1.write_patch_centers(self.treejack_save)

		conf = self.tc_proj_config.copy()
		p_args = dict(xperiod=self.xperiod, yperiod=self.yperiod, zperiod=self.zperiod, var_method=var_method)
		nn = treecorr.NNCorrelation(conf, **p_args)
		rr = treecorr.NNCorrelation(conf, **p_args)
		nr = treecorr.NNCorrelation(conf, **p_args)
		rn = treecorr.NNCorrelation(conf, **p_args)

		if auto:
			nn.process(data1, data1)
			rr.process(rand1, rand1)
			nr.process(data1, rand1)
			if self.compensated:
				xigg, varxi = nn.calculateXi(rr, dr=nr)
			else:
				xigg, varxi = nn.calculateXi(rr)
		else:
			nn.process(data1, data2)
			rr.process(rand1, rand2)
			nr.process(data1, rand2)
			rn.process(rand1, data2)
			if self.compensated:
				xigg, varxi = nn.calculateXi(rr, dr=nr, rd=rn)
			else:
				xigg, varxi = nn.calculateXi(rr)

		DD = nn.weight
		DR = nr.weight
		if auto: RD = nr.weight
		else: RD = rn.weight
		RR = rr.weight
		meanr = nn.meanr
		meanlogr = nn.meanlogr

		r = np.column_stack((nn.rnom, meanr, meanlogr))
		output = np.column_stack((r, xigg, varxi**0.5, DD, DR, RD, RR))
		np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','xigg','noise','DDpairs','DRpairs','RDpairs','RRpairs')))
		if self.treejack != 0:
			# collect TreeCorr jackknife products
			corrs = [nn]
			plist = [c._jackknife_pairs() for c in corrs]
			plist = list(zip(*plist))
			func = lambda corrs: np.concatenate([c.getStat() for c in corrs])
			v, w = treecorr.binnedcorr2._make_cov_design_matrix(corrs, plist, func, 'jackknife')
			vmean = np.mean(v, axis=0)
			v -= vmean
			C = (1.-1./len(v)) * v.conj().T.dot(v)
			output = np.column_stack((r, xigg, varxi**0.5, DD, DR, RD, RR, np.diag(C)**0.5))
			np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','xigg','noise','DDpairs','DRpairs','RDpairs','RRpairs','xigg_jackknife_err')))
			np.savetxt(outfile.replace('.dat', '.cov'), C)
			np.savetxt(outfile.replace('.dat', '.jk'), v, header='shape = (N jk samples, N sep-bins)')

		del data1, rand1, nn, nr, rr
		if not auto:
			del data2, rand2, rn
		gc.collect()

	def compute_xikk_xyz(self, d1, r1, outfile, auto=True, d2=None, r2=None, wcol1=None, wcol2=None, rwcol1=None, rwcol2=None):
		if self.k1_col != self.k2_col:
			auto = False
		kw, var_method = self.get_catalog_args()
		rand1 = treecorr.Catalog(x=r1[self.rand_x_col], y=r1[self.rand_y_col], z=r1[self.rand_z_col], is_rand=1, w=rwcol1, **kw)
		data1 = treecorr.Catalog(x=d1[self.x_col], y=d1[self.y_col], z=d1[self.z_col], k=d1[self.k1_col], w=wcol1, patch_centers=rand1.patch_centers)
		if not auto:
			rand2 = treecorr.Catalog(x=r2[self.rand_x_col], y=r2[self.rand_y_col], z=r2[self.rand_z_col], is_rand=1, w=rwcol2, patch_centers=rand1.patch_centers)
			data2 = treecorr.Catalog(x=d2[self.x_col], y=d2[self.y_col], z=d2[self.z_col], k=d2[self.k2_col], w=wcol2, patch_centers=rand1.patch_centers)
		else:
			data2 = data1
			rand2 = rand1
		if self.treejack != 0 and self.treejack_save and not os.path.exists(self.treejack_save):
			rand1.write_patch_centers(self.treejack_save)

		conf = self.tc_proj_config.copy()
		p_args = dict(xperiod=self.xperiod, yperiod=self.yperiod, zperiod=self.zperiod, var_method=var_method)
		kk = treecorr.KKCorrelation(conf, **p_args)
		nn = treecorr.NNCorrelation(conf, **p_args)
		rr = treecorr.NNCorrelation(conf, **p_args)
		if auto:
			kk.process(data1)
			nn.process(data1)
			rr.process(rand1)
		else:
			kk.process(data1, data2)
			nn.process(data1, data2)
			rr.process(rand1, rand2)

		xikk = kk.xi * kk.weight / (rr.weight * nn.tot / rr.tot)

		varxi = kk.varxi
		KK = kk.weight
		RR = rr.weight
		meanr = kk.meanr
		meanlogr = kk.meanlogr

		r = np.column_stack((kk.rnom, meanr, meanlogr))
		output = np.column_stack((r, xikk, varxi**0.5, kk.xi, KK, RR))
		np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','xikk','noise','KKraw','KKpairs', 'RRpairs')))
		if self.treejack != 0:
			# collect TreeCorr jackknife products
			corrs = [kk, rr, nn]
			plist = [c._jackknife_pairs() for c in corrs]
			plist = list(zip(*plist))
			func = lambda corrs: corrs[0].xi * corrs[0].weight / (corrs[1].weight * corrs[2].tot / corrs[1].tot)
			v, w = treecorr.binnedcorr2._make_cov_design_matrix(corrs, plist, func, 'jackknife')
			vmean = np.mean(v, axis=0)
			v -= vmean
			C = (1.-1./len(v)) * v.conj().T.dot(v)
			output = np.column_stack((r, xikk, varxi**0.5, kk.xi, KK, RR, np.diag(C)**0.5))
			np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','xikk','noise','KKraw','KKpairs','RRpairs','xikk_jackknife_err')))
			np.savetxt(outfile.replace('.dat', '.cov'), C)
			np.savetxt(outfile.replace('.dat', '.jk'), v, header='shape = (N jk samples, N sep-bins)')

		del data1, rand1, kk, rr
		if not auto:
			del data2, rand2
		gc.collect()

	def compute_xigk_xyz(self, d1, r1, d2, r2, outfile, wcol1=None, wcol2=None, rwcol1=None, rwcol2=None):
		kw, var_method = self.get_catalog_args()
		rand1 = treecorr.Catalog(x=r1[self.rand_x_col], y=r1[self.rand_y_col], z=r1[self.rand_z_col], is_rand=1, w=rwcol1, **kw)
		rand2 = treecorr.Catalog(x=r2[self.rand_x_col], y=r2[self.rand_y_col], z=r2[self.rand_z_col], is_rand=1, w=rwcol2, patch_centers=rand1.patch_centers)
		data1 = treecorr.Catalog(x=d1[self.x_col], y=d1[self.y_col], z=d1[self.z_col], w=wcol1, patch_centers=rand1.patch_centers)
		data2 = treecorr.Catalog(x=d2[self.x_col], y=d2[self.y_col], z=d2[self.z_col], w=wcol2, k=d2[self.k2_col], patch_centers=rand1.patch_centers)
		if self.treejack != 0 and self.treejack_save and not os.path.exists(self.treejack_save):
			rand1.write_patch_centers(self.treejack_save)

		conf = self.tc_proj_config.copy()
		p_args = dict(xperiod=self.xperiod, yperiod=self.yperiod, zperiod=self.zperiod)
		nk = treecorr.NKCorrelation(conf, **p_args)
		rk = treecorr.NKCorrelation(conf, **p_args)
		nn = treecorr.NNCorrelation(conf, **p_args)
		nr = treecorr.NNCorrelation(conf, **p_args)
		rr = treecorr.NNCorrelation(conf, **p_args)
		nk.process(data1, data2)
		rk.process(rand1, data2)
		nn.process(data1, data2)
		nr.process(rand1, data2)
		rr.process(rand1, rand2)

		# subtract randoms correlations
		if self.compensated:
			xigk = (nk.raw_xi * nk.weight / nn.tot - rk.raw_xi * rk.weight / nr.tot) / (rr.weight / rr.tot)
			varxi = nk.raw_varxi + rk.varxi
		else: # or not
			xigk = nk.raw_xi * nk.weight / nn.tot / (rr.weight / rr.tot)
			varxi = nk.raw_varxi

		DK = nk.weight
		RK = rk.weight
		RR = rr.weight
		meanr = nk.meanr
		meanlogr = nk.meanlogr

		r = np.column_stack((nk.rnom, meanr, meanlogr))
		output = np.column_stack((r, xigk, varxi**0.5, DK, RK, RR))
		np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','xigk','noise','DKpairs','RKpairs','RRpairs')))
		if self.treejack != 0:
			# collect TreeCorr jackknife products
			corrs = [nk, rk, rr, nn, nr]
			plist = [c._jackknife_pairs() for c in corrs]
			plist = list(zip(*plist))
			if self.compensated:
				func = lambda corrs: (corrs[0].raw_xi * corrs[0].weight / corrs[3].tot - corrs[1].raw_xi * corrs[1].weight / corrs[4].tot) / (corrs[2].weight / corrs[2].tot)
			else:
				func = lambda corrs: (corrs[0].raw_xi * corrs[0].weight / corrs[3].tot) / (corrs[2].weight / corrs[2].tot)
			v, w = treecorr.binnedcorr2._make_cov_design_matrix(corrs, plist, func, 'jackknife')
			vmean = np.mean(v, axis=0)
			v -= vmean
			C = (1.-1./len(v)) * v.conj().T.dot(v)
			output = np.column_stack((r, xigk, varxi**0.5, DK, RK, RR, np.diag(C)**0.5))
			np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','xigk','noise','DKpairs','RKpairs','RRpairs','xigk_jackknife_err')))
			np.savetxt(outfile.replace('.dat', '.cov'), C)
			np.savetxt(outfile.replace('.dat', '.jk'), v, header='shape = (N jk samples, N sep-bins)')

		del data1, rand1, data2, rand2, nk, rk, rr
		gc.collect()

	def run_loop(self, args, run_jackknife=0, jk_number=0):

		# skip jackknife measurements, if specified
		if run_jackknife:
			if self.jackknife_numbers is not None:
				if jk_number in self.jackknife_numbers:
					pass
				else:
					if jk_number > max(self.jackknife_numbers):
						return None
					else:
						if args.verbosity >= 1: print('====== SKIP jackknife #%i specified; continuing'%jk_number)
						self.run_loop(args, run_jackknife=1, jk_number=jk_number + 1)
						# break out of loop
						return None

		for i in tqdm(self.loop, ascii=True, desc='Running correlations', ncols=100):
			if any((self.paths_data1[i] == '',
					self.paths_rand1[i] == '')): # skip empty rows of config file
				continue

			# identify auto- vs. cross-correlations
			if not (self.paths_data1[i] == self.paths_data2[i] and
					self.paths_rand1[i] == self.paths_rand2[i] and
					self.data_cuts1[i] == self.data_cuts2[i] and
					self.rand_cuts1[i] == self.rand_cuts2[i] and
					self.data_weights1[i] == self.data_weights2[i] and
					self.rand_weights1[i] == self.rand_weights2[i] and
					self.corr_types[i] not in self.always_cross):
				auto = False
			else:
				auto = True

			if args.verbosity >= 1: print('\n== Auto-correlation = ', auto)
			if args.verbosity >= 1:
				print("\n== Correlation: ", self.outfiles[i], '(jackknife #%s)'%jk_number)
			if args.verbosity >= 2:
				print("\n====\n")
				if auto:
					print('Auto-correlation:')
					print(self.paths_data1[i], 'WITH', self.paths_rand1[i])
				else:
					print('Cross-correlation:')
					print(self.paths_data1[i], 'WITH', self.paths_rand1[i])
					print('vs.')
					print(self.paths_data2[i], 'WITH', self.paths_rand2[i])
				print("\n====\n")

			# clean-up 
			try: del d1, r1
			except:	pass
			try: del d2, r2
			except:	pass
			gc.collect()

			try:
				d1 = fits.open(self.paths_data1[i])[1].data
			except IOError:
				print("\n==== %s not found! Skipping.."%self.paths_data1[i])
				continue
			try:
				r1 = fits.open(self.paths_rand1[i])[1].data
			except IOError:
				print("\n==== %s not found! Skipping.."%self.paths_rand1[i])
				continue

			if auto:
				fits_cats = [d1, r1]
				cat_cuts = [self.data_cuts1, self.rand_cuts1]
				piter = iter(('== Calculating data1 cuts', '== Calculating randoms1 cuts'))
			else:
				d2 = fits.open(self.paths_data2[i])[1].data
				r2 = fits.open(self.paths_rand2[i])[1].data
				fits_cats = [d1, d2, r1, r2]
				cat_cuts = [self.data_cuts1, self.data_cuts2, self.rand_cuts1, self.rand_cuts2]
				piter = iter(('== Calculating data1 cuts', '== Calculating data2 cuts',
							 '== Calculating randoms1 cuts', '== Calculating randoms2 cuts'))

			# evaluate data/randoms cuts
			wcols = []
			for j, (cat, cut) in enumerate(zip(fits_cats, cat_cuts)):
				if args.verbosity >= 1: print(next(piter))
				cuts = cut[i].split('&')
				w = np.ones(len(cat), dtype=bool)

				if cuts[0] == 'none':
					if args.verbosity >= 1: print('==== no cuts!')
				else:
					# perform ID cut - special cut
					match_IDs = ['idmatch' in cut for cut in cuts]
					if any(match_IDs):
						idcut = np.array(cuts)[match_IDs][0]
						c1, c2, id1, id2 = idcut.replace(' ','').replace('idmatch','').replace('(','').replace(')','').split(',')
						idcut_bool = eval("idmatch(%s, %s, '%s', '%s')" % (c1, c2, id1, id2))
						if idcut_bool.sum() > 0:
							w &= idcut_bool
							if args.verbosity >= 1: print('==== cut="%s" for %.1f%% losses' % (idcut, (~idcut_bool).sum()*100./len(idcut_bool)))
						else:
							print('====== Error: ID matching failed! Skipping..')
						del cuts[cuts.index(idcut)]

					# perform custom downsample - special cut
					downsample = ['downsample' in cut for cut in cuts]
					if any(downsample):
						dscut = np.array(cuts)[downsample][0]
						custom_frac = dscut.replace('downsample(','').replace(')','')
						dscut_bool = np.random.rand(len(w)) <= float(custom_frac)
						if dscut_bool.sum() > 0:
							w &= dscut_bool
							if args.verbosity >= 1: print('==== cut="%s" for %.1f%% losses' % (dscut, (~dscut_bool).sum()*100./len(dscut_bool)))
						else:
							print('====== Error: custom downsampling failed! Skipping..')
						del cuts[cuts.index(dscut)]

					for col in np.sort(cat.columns.names): # loop over fits columns in catalogue
						for c in cuts: # loop over specified cuts for this correlation
							if col in c: # if this cut refers to this column
								crepl = c.replace(col, 'cat["%s"]'%col) # edit the string
								try:
									colcut = eval(crepl) # and construct the boolean array
									w &= colcut
									lencol = float(len(colcut))
									if args.verbosity >= 1: print('==== cut="%s" for %.1f%% losses' % (c, (~colcut).sum()*100./lencol))
								except:
									if args.verbosity >= 2: print('==== cut="%s" mismatched to column="%s" -- no action' % (c, col))

				zero_jk_cut = False
				if run_jackknife: # if doing jackknife, remove jackknife_ID == jk_number; will loop over all jk_numbers
					jk_set = set(cat['jackknife_ID'])
					if 0 in jk_set:
						jk_set.remove(0)
					self.Njk = len(jk_set)
					if jk_number == 0:
						jk_number = 1

					w &= (cat['jackknife_ID'] != 0) # always exclude ID = 0 -- these galaxies were lost in the jackknife routine
					wj = (cat['jackknife_ID'] != jk_number)
					w &= wj

					if all(cat['jackknife_ID'] != jk_number):
						# jackknife sample may not be relevant given other cuts -- skip to next jk_number
						zero_jk_cut = True

					if args.verbosity >= 1:
						print('==== jackknife #%s / %s excluded for %.1f%% losses'%(jk_number, self.Njk, (~wj).sum()*100./len(wj)))

				# update the samples
				fits_cats[j] = fits_cats[j][w]
				if args.verbosity >= 1: print('==== total loss: %.1f%%' % ((~w).sum()*100./len(w)))
				if auto:
					d1 = fits_cats[0]
					d2 = d1
					r1 = fits_cats[1]
					r2 = r1
				else:
					d1 = fits_cats[0]
					d2 = fits_cats[1]
					r1 = fits_cats[2]
					r2 = fits_cats[3]

			if zero_jk_cut:
				print('==== jackknife resampling does not apply -- skipping')
				continue

			# downsample excess randoms to factor <args.down> more than the data
			if args.down != 0:
				if args.verbosity >= 1: print('== Downsampling randoms..')
			if args.down == 0:
				if args.verbosity >= 2: print('== No downsampling of randoms!')
			elif len(r1) > args.down*(len(d1)):
				r1 = ds_func(d1, r1, target=args.down)
				if args.verbosity >= 2: print('==== Downsampled %s (%i) to %.fx num. of %s galaxies (%i)'
						% (basename(self.paths_rand1[i]), len(r1), float(len(r1))/len(d1), basename(self.paths_data1[i]), len(d1)))

			if not auto:
				if len(r2) > args.down*(len(d2)) and args.down != 0:
					r2 = ds_func(d2, r2, target=args.down)
					if args.verbosity >= 2: print('==== Downsampled %s (%i) to %.fx num. of %s galaxies (%i)'
							% (basename(self.paths_rand2[i]), len(r2), float(len(r2))/len(d2), basename(self.paths_data2[i]), len(d2)))
			else:
				d2 = d1
				r2 = r1

			if args.verbosity >= 1:
				print("==== data1: %s galaxies"%len(d1))
				print("==== rand1: %s galaxies"%len(r1))
				if not auto:
					print("==== data2: %s galaxies"%len(d2))
					print("==== rand2: %s galaxies"%len(r2))

			# optionally, save the treated catalogues for inspection
			if args.save_cats and jk_number == 0:
				td1 = Table(d1)
				tr1 = Table(r1)
				td1.write(self.outfiles[i].replace('.dat', '_d1.fits'), overwrite=1, format='fits')
				tr1.write(self.outfiles[i].replace('.dat', '_r1.fits'), overwrite=1, format='fits')
				del td1, tr1
				if not auto:
					td2 = Table(d2)
					tr2 = Table(r2)
					td2.write(self.outfiles[i].replace('.dat', '_d2.fits'), overwrite=1, format='fits')
					tr2.write(self.outfiles[i].replace('.dat', '_r2.fits'), overwrite=1, format='fits')
					del td2, tr2
				gc.collect()

			# determine galaxy weighting in similar fashion to evaluation of cuts above
			if self.data_weights1[i] in ['ones', 'none', '']:
				wcol1 = np.ones(len(d1))
			else:
				for col in np.sort(d1.columns.names):
					if col in self.data_weights1[i]:
						wcol = self.data_weights1[i].replace(col, 'd1["%s"]'%col)
						try:
							wcol1 = eval(wcol)
							if args.verbosity >= 1: print('==== weight="%s" applied to data1' % self.data_weights1[i])
							break
						except:
							if args.verbosity >= 2: print('==== weights="%s" mismatched to column="%s" -- no action' % (self.data_weights1[i], col))
			if self.data_weights2[i] in ['ones', 'none', '']:
				wcol2 = np.ones(len(d2))
			else:
				for col in np.sort(d2.columns.names):
					if col in self.data_weights2[i]:
						wcol = self.data_weights2[i].replace(col, 'd2["%s"]'%col)
						try:
							wcol2 = eval(wcol)
							if args.verbosity >= 1: print('==== weight="%s" applied to data2' % self.data_weights2[i])
							break
						except:
							if args.verbosity >= 2: print('==== weights="%s" mismatched to column="%s" -- no action' % (self.data_weights2[i], col))
			if self.rand_weights1[i] in ['ones', 'none', '']:
				rwcol1 = np.ones(len(r1))
			else:
				for col in np.sort(r1.columns.names):
					if col in self.rand_weights1[i]:
						wcol = self.rand_weights1[i].replace(col, 'r1["%s"]'%col)
						try:
							rwcol1 = eval(wcol)
							if args.verbosity >= 1: print('==== weight="%s" applied to rand1' % self.rand_weights1[i])
							break
						except:
							if args.verbosity >= 2: print('==== weights="%s" mismatched to column="%s" -- no action' % (self.rand_weights1[i], col))
			if self.rand_weights2[i] in ['ones', 'none', '']:
				rwcol2 = np.ones(len(r2))
			else:
				for col in np.sort(r2.columns.names):
					if col in self.rand_weights2[i]:
						wcol = self.rand_weights2[i].replace(col, 'r2["%s"]'%col)
						try:
							rwcol2 = eval(wcol)
							if args.verbosity >= 1: print('==== weight="%s" applied to rand2' % self.rand_weights2[i])
							break
						except:
							if args.verbosity >= 2: print('==== weights="%s" mismatched to column="%s" -- no action' % (self.rand_weights2[i], col))

			# edit .dat suffix if saving down jackknife measurements
			if run_jackknife:
				outfile = self.outfiles[i].replace('.dat', '.jk%s'%jk_number)
			else:
				outfile = self.outfiles[i]

			# perform correlation -- need to generalise these functions to do projected/3D in RADEC/XYZ
			if self.corr_types[i] == 'wth':
				self.compute_wtheta(d1, r1, outfile, auto=auto, d2=d2, r2=r2, wcol1=wcol1, wcol2=wcol2, rwcol1=rwcol1, rwcol2=rwcol2)
			elif self.coordinates == 'RADEC':
				if self.corr_types[i] == 'wgp':
					self.compute_wgplus(d1, r1, d2, r2, outfile, wcol1=wcol1, wcol2=wcol2, rwcol1=rwcol1, rwcol2=rwcol2)
				if self.corr_types[i] == 'wgg':
					self.compute_wgg(d1, r1, outfile, auto=auto, d2=d2, r2=r2, wcol1=wcol1, wcol2=wcol2, rwcol1=rwcol1, rwcol2=rwcol2)
			elif self.coordinates == 'XYZ':
				if self.corr_types[i] == 'xigp':
					self.compute_xigplus_xyz(d1, r1, d2, r2, outfile, wcol1=wcol1, wcol2=wcol2, rwcol1=rwcol1, rwcol2=rwcol2)
				if self.corr_types[i] == 'xigg':
					self.compute_xigg_xyz(d1, r1, outfile, auto=auto, d2=d2, r2=r2, wcol1=wcol1, wcol2=wcol2, rwcol1=rwcol1, rwcol2=rwcol2)
				if self.corr_types[i] == 'xikk':
					self.compute_xikk_xyz(d1, r1, outfile, auto=auto, d2=d2, r2=r2, wcol1=wcol1, wcol2=wcol2, rwcol1=rwcol1, rwcol2=rwcol2)
				if self.corr_types[i] == 'xigk':
					self.compute_xigk_xyz(d1, r1, d2, r2, outfile, wcol1=wcol1, wcol2=wcol2, rwcol1=rwcol1, rwcol2=rwcol2)
			else:
				print(f'===== {self.outfiles[i]}: correlation {self.corr_types[i]} w/ coordinates {self.coordinates} not suported')

			# clean up
			del d1, d2, r1, r2, fits_cats
			gc.collect()

		# if doing jackknife, +1 to the jk_number and run again
		if run_jackknife:
			if jk_number < self.Njk:
				self.run_loop(args, run_jackknife=1, jk_number=jk_number + 1)
			else:
				return None

	def collect_jackknife(self, columns=['wgplus','wgcross','wgg','xi','xigg','xigplus','xigcross','xigk','xikk']):
		# collect-up jackknife measurements and construct
		# jackknife covariance/add jackknife mean and stderr
		# columns into main measurement output files
		for i in tqdm(self.loop, ascii=True, desc='Collecting covariances', ncols=100):
			if any((self.paths_data1[i] == '',
					self.paths_rand1[i] == '')):
				continue
			if not hasattr(self, 'Njk'):
				jk_set = set(fits.open(self.paths_data1[i])[1].data['jackknife_ID'])
				if 0 in jk_set:
					jk_set.remove(0)
				self.Njk = len(jk_set)
			jk_numbers = range(1, self.Njk + 1)
			try:
				jk_data = [self.outfiles[i].replace('.dat', '.jk%s'%jkn) for jkn in jk_numbers]
			except:
				print(f'==== jackknife correlations missing for {self.outfiles[i]}? Continuing')
				continue

			# collect relevant jackknife measurements
			asc_arr = []
			for jk in jk_data:
				try:
					asc_arr.append(ascii.read(jk))
				except: # some cuts may not have corresponding jackknife samples
					print('\n==== %s jackknife read-in failed -- continuing'%jk)
					continue
			Njk_i = len(asc_arr)
			if Njk_i == 0 or Njk_i < self.Njk//2:
				print('\n==== %s jackknife collection failed -- skipping'%self.outfiles[i])
				continue

			for col in columns:
				if col not in asc_arr[0].colnames:
					continue
				else:
					data_arr = np.array([da[col] for da in asc_arr])
					try:
						# jackknife re-norm without weights for now -- assume samples are not too diverse
						cov = np.cov(data_arr, rowvar=0) * (Njk_i - 1.)**2. / Njk_i
					except:
						print('\n==== %s jackknife collection failed -- skipping'%self.outfiles[i])
						continue
					try:
						stdev = np.sqrt(np.diag(cov))
					except:
						print('\n==== %s jackknife collection failed -- skipping'%self.outfiles[i])
						continue
					mean = data_arr.mean(axis=0)
					data = ascii.read(self.outfiles[i])
					data['%s_jackknife_err'%col] = stdev
					data['%s_jackknife_mean'%col] = mean
					data.write(self.outfiles[i], format='ascii.commented_header', overwrite=1)
					np.savetxt(self.outfiles[i].replace('.dat', '_%s.cov'%col), cov, header='Njk = %s'%Njk_i)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'config_file',
		help='Path to configuration file for correlations')
	parser.add_argument(
		'-bin_slop',
		type=float,
		help='Specify TreeCorr bin_slop parameter (float, default=None -- will use config value)')
	parser.add_argument(
		'--num-threads',
		type=int,
		default=8,
		help='Specify number of threads to employ (default=8)')
	parser.add_argument(
		'--save-cats',
		action='store_true',
		help='Save out FITS catalogues (per correlation, after cuts) for inspection')
	parser.add_argument(
		'-verbosity',
		type=int,
		default=1,
		help='Specify verbosity (int[0,3], default=3) for TreeCorr computations, and reports from this code')
	parser.add_argument(
		'-down',
		type=float,
		default=10.,
		help='Specify randoms oversampling factor = Nrandoms / Ngalaxies (float, default=10. // set to 0. for no downsampling)')
	parser.add_argument(
		'-index',
		type=int,
		nargs='*',
		help='Give indices of correlations in the config file that you desire to run -- others will be skipped. E.g. -index 0 will run only the first correlation')
	parser.add_argument(
		'-rindex',
		type=int,
		nargs='*',
		help='Give indices of correlations in the config file that you desire NOT to run i.e. inverse of -index argument')
	parser.add_argument(
		'-randomise',
		action='store_true',
		help='Randomise the order of correlations for faster coverage when doing multiple realisations of similar statistics')
	parser.add_argument(
		'-p',
		type=str,
		nargs='*',
		help='Override config-file parameters e.g. -p section.param=value (make sure no trailing slashes in file paths, space-separated arguments may not work)')
	args = parser.parse_args()

	Corr = Correlate(args)
	if Corr.run_jackknife == 1:
		Corr.run_loop(args, run_jackknife=1)
		Corr.collect_jackknife()
	elif Corr.run_jackknife == 2:
		Corr.run_loop(args, run_jackknife=1)
		Corr.run_loop(args)
		Corr.collect_jackknife()
	elif Corr.run_jackknife == 3:
		Corr.run_loop(args)
		Corr.run_loop(args, run_jackknife=1)
		Corr.collect_jackknife()
	elif Corr.run_jackknife == 4:
		Corr.collect_jackknife()
	else:
		Corr.run_loop(args)
	print('\n== Done!')



