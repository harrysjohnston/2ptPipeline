print "\n2-point correlation pipeline -- Harry Johnston 2019"
print "Work in progress!"
print "hj@star.ucl.ac.uk\n"

from os.path import join, expandvars, basename
from os import mkdir
from os.path import isdir
from astropy.io import fits, ascii
from astropy.table import Table
import configparser
cp = configparser.ConfigParser()
import numpy as np
import treecorr
from tqdm import tqdm
import argparse
import sys
import skyknife
import pickle

def ds_func(cat, rcat, target=10.):
	# downsample size of <rcat> to <target> * size of <cat>
	randnum = np.random.rand(len(rcat))
	cut = randnum < (len(cat)*target) / len(rcat)
	return rcat[cut]

def uhalfcut(arr,q=0):
	# remove the largest half of parameter <arr>
	# or largest quarter, if q=1
	perc50 = np.percentile(arr, 50.)
	perc25 = np.percentile(arr, 25.)
	if q:
		return arr < perc25
	else:
		return arr < perc50
def lhalfcut(arr,q=0):
	# remove the smallest half of parameter <arr>
	# or smallest quarter, if q=1
	perc50 = np.percentile(arr, 50.)
	perc25 = np.percentile(arr, 75.)
	if q:
		return arr > perc25
	else:
		return arr > perc50
def lqcut(arr,q=10.):
	# remove all parameter <arr> less than quantile <q>
	perc = np.percentile(arr, q)
	return arr > perc
def hqcut(arr,q=10.):
	# remove all parameter <arr> greater than quantile <q>
	perc = np.percentile(arr, q)
	return arr < perc
midpoints = lambda x: (x[1:] + x[:-1]) / 2.

class Correlate:
	def __init__(self, args):
		print '\n== Preparing correlations\n'
		cp.read(args.config_file)
		if args.p is not None: # override configuration file arguments on command line: -p <section>.<arg>=value
			print '== Overriding config arguments from command line:'
			for ap in args.p:
				s = ap.split('.')[0]
				p, v = ap.replace(s, '')[1:].split('=')
				if args.verbosity >= 1 and cp.get(s, p) != v:
					print '==== %s.%s:'%(s, p), cp.get(s, p), '->', v
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
		if data_weights1 == ['']:
			if args.verbosity >= 1: print '== Galaxy weighting (data_weights) not set; assuming unity weights for all'
			data_weights1 = ['ones'] * len(paths_data1)

		# get correlation type list
		corr_types = cp.get('catalogs', 'corr_types').replace(' ', '').replace('\n', '').split('//')
		if corr_types == ['']:
			print '== Correlation types (corr_types) not set; assuming angular clustering for all'
			corr_types = ['wth'] * len(paths_data1)
		else:
			if corr_types[-1] == '': corr_types = corr_types[:-1]
			assert all([ct in ['wth', 'wgp', 'wgg'] for ct in corr_types]), "Must specify corr_types as 'wth' (angular clust.), 'wgg' (proj. clust.) or 'wgp' (proj. direct IA) per correlation -- adding more correlations soon"

		# if not providing different catalogues/cuts/weights for cross-correlation,
		# copy arguments for auto-correlations
		if paths_data2 == ['']:
			paths_data2 = list(paths_data1)
		if paths_rand2 == ['']:
			paths_rand2 = list(paths_rand1)
		if data_cuts2 == ['']:
			data_cuts2 = list(data_cuts1)
		if rand_cuts2 == ['']:
			rand_cuts2 = list(rand_cuts1)
		if data_weights2 == ['']:
			data_weights2 = list(data_weights1)

		# ensure each argument is specified once, to be carried over for all correlations,
		# or for each correlation individually
		ncats = (len(paths_data1), len(paths_data2), len(paths_rand1), len(paths_rand2))
		ncuts = (len(data_cuts1), len(data_cuts2), len(rand_cuts1), len(rand_cuts2))
		nweights = (len(data_weights1), len(data_weights2))
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
		if len(corr_types) == 1:
			corr_types = corr_types * max(ncats + ncuts + nweights + ncorrtypes)

		outdir = expandvars(cp.get('output', 'savedir'))
		if not isdir(outdir):
			mkdir(outdir)
		outs = cp.get('output', 'out_corrs').replace(' ', '').replace('\n', '').split('//')
		if outs == ['']:
			print '== Assigning outfile names == corr2_out_XX.txt'
			outs = ['corr2_out_%s.dat' % str(i).zfill(2) for i in range(len(paths_data1))]
		for i in range(len(outs)):
			if not outs[i].endswith('.dat'):
				outs[i] += '.dat'
		outfiles = [join(outdir, out) for out in outs]

		# construct correlations to loop over
		if args.index is not None:
			loop = np.arange(len(paths_data1))[args.index]
		else:
			loop = range(len(paths_data1))

		# re-write this to include treecorr config options in my own config file?
		tc_config_path = expandvars(cp.get('treecorr_config', 'default'))
		try:
			tc_config = treecorr.read_config(tc_config_path)
			for option in cpd['treecorr_config'].keys():
				if option == 'default': continue
				tc_config[option] = cpd['treecorr_config'][option]
			tc_config['bin_slop'] = args.bin_slop
			tc_config['num_threads'] = args.num_threads
			tc_config['verbose'] = args.verbosity
			self.ra_col = tc_config['ra_col']
			self.dec_col = tc_config['dec_col']
			self.ra_units = tc_config['ra_units']
			self.dec_units = tc_config['dec_units']
			self.tc_config = tc_config
		except ValueError:
			print "== No treecorr config passed -- parameters for projected correlations should be specified in w_pipe config"
			tc_wgp_config = cpd['wgplus_config']
			tc_wgp_config['bin_slop'] = args.bin_slop
			tc_wgp_config['num_threads'] = args.num_threads
			tc_wgp_config['verbose'] = args.verbosity
			for option in cpd['treecorr_config'].keys():
				if option == 'default': continue
				tc_wgp_config[option] = cpd['treecorr_config'][option]
			self.tc_wgp_config = tc_wgp_config
			self.estimator = tc_wgp_config['estimator']
			self.ra_col_proj = tc_wgp_config['ra_col']
			self.dec_col_proj = tc_wgp_config['dec_col']
			self.ra_units = tc_wgp_config['ra_units']
			self.dec_units = tc_wgp_config['dec_units']
			self.r_col = tc_wgp_config['r_col']
			self.rand_ra_col_proj = tc_wgp_config['rand_ra_col']
			self.rand_dec_col_proj = tc_wgp_config['rand_dec_col']
			self.rand_r_col = tc_wgp_config['rand_r_col']
			if self.rand_ra_col_proj == '':
				self.rand_ra_col_proj = self.ra_col_proj
			if self.rand_dec_col_proj == '':
				self.rand_dec_col_proj = self.dec_col_proj
			if self.rand_r_col == '':
				self.rand_r_col = self.r_col
			self.g1_col = tc_wgp_config['g1_col']
			self.g2_col = tc_wgp_config['g2_col']
			self.flip_g1 = int(tc_wgp_config['flip_g1'])
			self.flip_g2 = int(tc_wgp_config['flip_g2'])
			self.fg1 = (1., -1.)[self.flip_g1]
			self.fg2 = (1., -1.)[self.flip_g2]
			self.min_rpar = float(tc_wgp_config['min_rpar'])
			self.max_rpar = float(tc_wgp_config['max_rpar'])
			self.nbins_rpar = int(tc_wgp_config['nbins_rpar'])
			self.nbins = int(tc_wgp_config['nbins'])
			self.largePi = int(tc_wgp_config['largepi'])
			self.compensated = int(tc_wgp_config['compensated'])
			self.random_oversampling = args.down
			self.save_3d = int(tc_wgp_config['save_3d'])

		self.build_jackknife = int(cp.get('jackknife', 'build', fallback=0))
		self.run_jackknife = int(cp.get('jackknife', 'run', fallback=0))
		if self.run_jackknife == 1:
			print "== Run_jackknife = 1 -- performing N jackknife correlations excluding regions N"
		if self.run_jackknife == 2:
			print "== Run_jackknife = 2 -- performing jackknife before main correlations"
		if self.run_jackknife == 3:
			print "== Run_jackknife = 3 -- performing jackknife after main correlations"
		if self.run_jackknife == 4:
			print "== Run_jackknife = 4 -- collecting jackknife covariance only"
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
		self.corr_types = corr_types
		self.outdir = outdir
		self.outfiles = outfiles
		self.loop = loop

	def compute_wtheta(self, d1, r1, outfile, auto=True, d2=None, r2=None, wcol1=None, wcol2=None):
		data1 = treecorr.Catalog(ra=d1[self.ra_col], dec=d1[self.dec_col], ra_units=self.ra_units, dec_units=self.dec_units, w=wcol1)
		rand1 = treecorr.Catalog(ra=r1[self.ra_col], dec=r1[self.dec_col], ra_units=self.ra_units, dec_units=self.dec_units, is_rand=1)
		if not auto:
			data2 = treecorr.Catalog(ra=d2[self.ra_col], dec=d2[self.dec_col], ra_units=self.ra_units, dec_units=self.dec_units, w=wcol2)
			rand2 = treecorr.Catalog(ra=r2[self.ra_col], dec=r2[self.dec_col], ra_units=self.ra_units, dec_units=self.dec_units, is_rand=1)

		nn = treecorr.NNCorrelation(self.tc_config)
		nr = treecorr.NNCorrelation(self.tc_config)
		rr = treecorr.NNCorrelation(self.tc_config)
		if auto:
			nn.process(data1)
			rr.process(rand1)
			nr.process(data1, rand1)
			nn.write(outfile, rr=rr, dr=nr, file_type='ASCII', prec=6)
		else:
			rn = treecorr.NNCorrelation(tc_config)
			nn.process(data1, data2)
			rr.process(rand1, rand2)
			nr.process(data1, rand2)
			rn.process(rand1, data2)
			nn.write(outfile, rr=rr, dr=nr, rd=rn, file_type='ASCII', prec=6)

	def compute_wgplus(self, d1, r1, d2, r2, outfile, wcol1=None, wcol2=None):
		if not self.largePi:
			Pi = np.linspace(self.min_rpar, self.max_rpar, self.nbins_rpar + 1)
		else: # define new Pi-range, with |Pi_max| = 1.5 * |max_arg|
			dPi = (self.max_rpar - self.min_rpar) / self.nbins_rpar
			Pi = np.arange(self.min_rpar*1.5, self.max_rpar*1.5 + dPi, step=dPi, dtype=float)
		gt_3D = np.zeros([self.nbins_rpar, self.nbins])
		gx_3D = np.zeros([self.nbins_rpar, self.nbins])
		varg_3D = np.zeros([self.nbins_rpar, self.nbins])
		DS_3D = np.zeros([self.nbins_rpar, self.nbins])
		RS_3D = np.zeros([self.nbins_rpar, self.nbins])

		data1 = treecorr.Catalog(ra=d1[self.ra_col_proj], dec=d1[self.dec_col_proj], ra_units=self.ra_units, dec_units=self.dec_units, w=wcol1,
								 r=d1[self.r_col], g1=d1[self.g1_col] * self.fg1, g2=d1[self.g2_col] * self.fg2)
		data2 = treecorr.Catalog(ra=d2[self.ra_col_proj], dec=d2[self.dec_col_proj], ra_units=self.ra_units, dec_units=self.dec_units, w=wcol2,
								 r=d2[self.r_col], g1=d2[self.g1_col] * self.fg1, g2=d2[self.g2_col] * self.fg2)
		rand1 = treecorr.Catalog(ra=r1[self.rand_ra_col_proj], dec=r1[self.rand_dec_col_proj], ra_units=self.ra_units, dec_units=self.dec_units,
								 r=r1[self.rand_r_col], is_rand=1)
		rand2 = treecorr.Catalog(ra=r2[self.rand_ra_col_proj], dec=r2[self.rand_dec_col_proj], ra_units=self.ra_units, dec_units=self.dec_units,
								 r=r2[self.rand_r_col], is_rand=1)

		# deprecated normalisation step
		#f1 = data1.ntot * self.random_oversampling / float(rand1.ntot)
		#f2 = data2.ntot * self.random_oversampling / float(rand2.ntot)
		#rand1.w = np.array(np.random.rand(rand1.ntot) < f1, dtype=float)
		#rand2.w = np.array(np.random.rand(rand2.ntot) < f2, dtype=float)
		varg = treecorr.calculateVarG(data2)

		for p in range(self.nbins_rpar):
			if self.largePi and any(abs(Pi[p:p+2]) < self.max_rpar): # skip any |Pi| < max_arg
				continue

			# limit correlation to this Pi-bin
			conf_pi = self.tc_wgp_config.copy()
			conf_pi['min_rpar'] = Pi[p]
			conf_pi['max_rpar'] = Pi[p+1]

			ng = treecorr.NGCorrelation(conf_pi)
			rg = treecorr.NGCorrelation(conf_pi)
			ng.process_cross(data1, data2)
			rg.process_cross(rand1, data2)
			ng.varxi = varg
			ng.varxi = varg
			#ng.finalize(varg)
			#rg.finalize(varg)

			set_d1_ra = set(d1[self.ra_col_proj])
			set_d2_ra = set(d2[self.ra_col_proj])
			intersection = set_d1_ra.intersection(set_d2_ra)
			setattr(ng, 'tot', 1.*data1.ntot*data2.ntot - len(intersection))
			setattr(rg, 'tot', 1.*rand1.ntot*data2.ntot)
			rgw = ng.tot / rg.tot

			# calculate appropriate normalisation of terms for chosen estimator
			if self.estimator == 'PW1': # RDs (rg) norm
				#norm1 = (rg.weight * rgw) / ng.weight
				#norm2 = rgw
				norm1 = rg.weight * rgw
				norm2 = rg.weight
			if self.estimator == 'PW2': # RRs (rr) norm
				rr = treecorr.NNCorrelation(conf_pi)
				rr.process_cross(rand1, rand2)
				rr.finalize()
				rrw = ng.tot / rr.tot
				rrgw = rg.tot / rr.tot
				#norm1 = (rr.weight * rrw) / ng.weight
				#norm2 = (rr.weight * rrw) / (rg.weight * rgw)
				norm1 = rr.weight * rrw
				norm2 = rr.weight * rrgw
			elif self.estimator == 'AS': # DDs (ng), RDs (rg) norms, done by .finalize()
				#norm1 = 1.
				#norm2 = rgw
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
			DS_3D[p] += ng.weight
			RS_3D[p] += rg.weight

		# compute/save projected statistics
		gt = np.sum(gt_3D * (Pi[1] - Pi[0]), axis=0)
		gx = np.sum(gx_3D * (Pi[1] - Pi[0]), axis=0)
		varg = np.sum(varg_3D, axis=0)
		DS = np.sum(DS_3D, axis=0)
		RS = np.sum(RS_3D, axis=0)
		r = np.column_stack((ng.rnom, ng.meanr, ng.meanlogr))
		output = np.column_stack((r, gt, gx, varg**0.5, DS, RS))
		np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','wgplus','wgcross','noise','DSpairs','RSpairs')))

		# optionally, save the 2D paircounts (3D name is stupid, I know)
		if self.save_3d:
			if args.verbosity >= 1: print "== Saving 3D correlations.."

			# with normalisations for paircounts
			DSntot = data1.ntot * data2.ntot
			if data1.ntot == data2.ntot:
				DSntot -= data2.ntot
			RSntot = rand1.ntot * data2.ntot

			output_dict = {'r':r,'Pi':Pi,'w3d':gt_3D,'wx3d':gx_3D,'noise3d':varg_3D**0.5,'DS_3d':DS_3D,'RS_3d':RS_3D,
							'DSntot':DSntot,'RSntot':RSntot}
			if outfile.endswith('.dat'):
				outfile3d = outfile.replace('.dat', '.p')
			elif '.jk' in outfile:
				outfile3d = outfile.replace('.jk', '.pjk')
			else:
				print "==== Unrecognised output type -- not saving 3D correlations"
				return None
			pickle.dump(output_dict, open(outfile3d, 'w'))

	def compute_wgg(self, d1, r1, outfile, auto=True, d2=None, r2=None, wcol1=None, wcol2=None):
		if not self.largePi:
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

		data1 = treecorr.Catalog(ra=d1[self.ra_col_proj], dec=d1[self.dec_col_proj], r=d1[self.r_col], w=wcol1,
								 ra_units=self.ra_units, dec_units=self.dec_units)
		rand1 = treecorr.Catalog(ra=r1[self.rand_ra_col_proj], dec=r1[self.rand_dec_col_proj], r=r1[self.rand_r_col], is_rand=1,
								 ra_units=self.ra_units, dec_units=self.dec_units)
		if not auto:
			data2 = treecorr.Catalog(ra=d2[self.ra_col_proj], dec=d2[self.dec_col_proj], r=d2[self.r_col], w=wcol2,
									 ra_units=self.ra_units, dec_units=self.dec_units)
			rand2 = treecorr.Catalog(ra=r2[self.rand_ra_col_proj], dec=r2[self.rand_dec_col_proj], r=r2[self.rand_r_col], is_rand=1,
									 ra_units=self.ra_units, dec_units=self.dec_units)

		for p in range(self.nbins_rpar):
			if self.largePi and any(abs(Pi[p:p+2]) < self.max_rpar): # skip any |Pi| < max_arg
				continue

			# limit correlation to this Pi-bin
			conf_pi = self.tc_wgp_config.copy()
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

		# compute/save projected statistics
		wgg = np.sum(wgg_3D * (Pi[1] - Pi[0]), axis=0)
		varw = np.sum(varw_3D, axis=0)
		DDpair = np.sum(DD_3D, axis=0)
		DRpair = np.sum(DR_3D, axis=0)
		RDpair = np.sum(RD_3D, axis=0)
		RRpair = np.sum(RR_3D, axis=0)
		r = np.column_stack((nn.rnom, nn.meanr, nn.meanlogr))
		output = np.column_stack((r, wgg, varw**0.5, DDpair, DRpair, RDpair, RRpair))
		np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','wgg','noise','DDpairs','DRpairs','RDpairs','RRpairs')))

		# optionally, save the 2D paircounts
		if self.save_3d:
			if args.verbosity >= 1: print "== Saving 3D correlations.."

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

			output_dict = {'r':r,'Pi':Pi,'w3d':wgg_3D,'noise3d':varw_3D**0.5,
						   'DD3d':DD_3D,'DR3d':DR_3D,'RD3d':RD_3D,'RR3d':RR_3D,
						   'DDntot':DDntot,'DRntot':DRntot,'RDntot':RDntot,'RRntot':RRntot}
			if outfile.endswith('.dat'):
				outfile3d = outfile.replace('.dat', '.p')
			elif '.jk' in outfile:
				outfile3d = outfile.replace('.jk', '.pjk')
			else:
				print "==== Unrecognised output type -- not saving 3D correlations"
				return None
			pickle.dump(output_dict, open(outfile3d, 'w'))


	def run_loop(self, args, run_jackknife=0, jk_number=0):
		for i in tqdm(self.loop, ascii=True, desc='Running correlations', ncols=100):
			if any((self.paths_data1[i] == '',
					self.paths_rand1[i] == '')): # skip empty rows of config file
				continue

			# identify auto- vs. cross-correlations
			if not (self.paths_data1[i] == self.paths_data2[i] and
					self.paths_rand1[i] == self.paths_rand2[i] and
					self.data_cuts1[i] == self.data_cuts2[i] and
					self.rand_cuts1[i] == self.rand_cuts2[i] and
					self.data_weights1[i] == self.data_weights2[i]):
				auto = False
			else:
				auto = True

			if self.build_jackknife:
				# build jackknife with uniform randoms
				# use randoms jackknife to create galaxy jackknife
				# NEEDS DEBUG
				if self.corr_types[i] in ['wgp', 'wgg']:
					rrcol = self.rand_ra_col_proj
					rdcol = self.rand_dec_col_proj
					rzcol = self.rand_r_col
					zcol = self.r_col
				else:
					rrcol = self.ra_col
					rdcol = self.dec_col
					rzcol = None
					zcol = None
				rand_sk = skyknife.Jackknife(args.config_file, catpath=self.paths_rand1[i], ra_col=rrcol, dec_col=rdcol, z_col=rzcol)
				data_sk = skyknife.Jackknife(args.config_file, catpath=self.paths_data1[i], ra_col=self.ra_col, dec_col=self.dec_col, z_col=zcol)
				rand_groups = rand_sk.create_jackknife()
				data_groups = data_sk.create_jackknife(rand_groups)
				if not auto:
					rand2_sk = skyknife.Jackknife(args.config_file, catpath=self.paths_rand2[i], ra_col=rrcol, dec_col=rdcol, z_col=rzcol)
					data2_sk = skyknife.Jackknife(args.config_file, catpath=self.paths_data2[i], ra_col=self.ra_col, dec_col=self.dec_col, z_col=zcol)
					rand2_groups = rand2_sk.create_jackknife(rand_groups)
					data2_groups = data2_sk.create_jackknife(rand_groups)

			if args.verbosity >= 1:
				print "\n== Correlation: ", self.outfiles[i], '(jackknife #%s)'%jk_number

			try:
				d1 = fits.open(self.paths_data1[i])[1].data
			except IOError:
				print("==== %s not found! Skipping.."%self.paths_data1[i])
				continue
			try:
				r1 = fits.open(self.paths_rand1[i])[1].data
			except IOError:
				print("==== %s not found! Skipping.."%self.paths_rand1[i])
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
			for cat, cut in zip(fits_cats, cat_cuts):
				if args.verbosity >= 1: print(next(piter))
				cuts = cut[i].split('&')
				w = np.ones(len(cat), dtype=bool)
				if cuts[0] == 'none':
					wcols.append(w)
					continue

				if run_jackknife: # if doing jackknife, remove jackknife_ID == jk_number; will loop over all jk_numbers
					self.Njk = len(set(cat['jackknife_ID']))
					if jk_number == 0:
						jk_number = 1
					w &= (cat['jackknife_ID'] != 0) # always exclude ID = 0 -- these galaxies were lost in the jackknife routine
					w &= (cat['jackknife_ID'] != jk_number)
					if all(cat['jackknife_ID'] != jk_number):
						print "======== JACKKNIFE #%s FAILED TO EXCLUDE ANY GALAXIES -- must manually exclude this correlation"%(jk_number)
					if args.verbosity >= 1: print('==== jackknife #%s / %s excluded for %.1f%% losses'%(jk_number, len(set(cat['jackknife_ID'])), (~w).sum()*100./len(w)))

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
				wcols.append(w)

			if args.verbosity >= 1: print '== Applying cuts..'
			if args.verbosity >= 1: print '== Auto-correlation = ', auto
			# apply the cuts to data/randoms
			if auto:
				d1 = d1[wcols[0]]
				r1 = r1[wcols[1]]
				d2 = d1
				r2 = r1
			else:
				d1 = d1[wcols[0]]
				d2 = d2[wcols[1]]
				r1 = r1[wcols[2]]
				r2 = r2[wcols[3]]

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
				print "==== data1: %s galaxies"%len(d1)
				print "==== data2: %s galaxies"%len(d2)
				print "==== rand1: %s galaxies"%len(r1)
				print "==== rand2: %s galaxies"%len(r2)

			# optionally, save the treated catalogues for inspection
			if args.save_cats:
				td1 = Table(d1)
				tr1 = Table(r1)
				td2 = Table(d2)
				tr2 = Table(r2)
				td1.write(self.outfiles[i].replace('.dat', '_d1.fits'), overwrite=1, format='fits')
				tr1.write(self.outfiles[i].replace('.dat', '_r1.fits'), overwrite=1, format='fits')
				td2.write(self.outfiles[i].replace('.dat', '_d2.fits'), overwrite=1, format='fits')
				tr2.write(self.outfiles[i].replace('.dat', '_r2.fits'), overwrite=1, format='fits')

			# determine galaxy weighting in similar fashion to evaluation of cuts above
			if self.data_weights1[i] in ['ones', 'none']:
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
			if self.data_weights2[i] in ['ones', 'none']:
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

			# edit .dat suffix if saving down jackknife measurements
			if run_jackknife:
				outfile = self.outfiles[i].replace('.dat', '.jk%s'%jk_number)
			else:
				outfile = self.outfiles[i]

			# perform correlation
			if self.corr_types[i] == 'wth':
				self.compute_wtheta(d1, r1, outfile, auto=auto, d2=d2, r2=r2, wcol1=wcol1, wcol2=wcol2)
			if self.corr_types[i] == 'wgp':
				self.compute_wgplus(d1, r1, d2, r2, outfile, wcol1=wcol1, wcol2=wcol2)
			if self.corr_types[i] == 'wgg':
				self.compute_wgg(d1, r1, outfile, auto=auto, d2=d2, r2=r2, wcol1=wcol1, wcol2=wcol2)

		# if doing jackknife, +1 to the jk_number and run again
		if run_jackknife:
			if jk_number < self.Njk:
				self.run_loop(args, run_jackknife=1, jk_number=jk_number + 1)
			else:
				jk_number = 1

	def collect_jackknife(self, column=3):
		# collect-up jackknife measurements and construct
		# jackknife covariance/add jackknife mean and stderr
		# columns into main measurement output files
		for i in tqdm(self.loop, ascii=True, desc='Collecting covariances', ncols=100):
			if any((self.paths_data1[i] == '',
					self.paths_rand1[i] == '')):
				continue
			if not hasattr(self, 'Njk'):
				self.Njk = len(set(fits.open(self.paths_data1[i])[1].data['jackknife_ID'])) - 1 # exclude zero!
			jk_numbers = range(1, self.Njk + 1)
			#try:
			jk_data = [self.outfiles[i].replace('.dat', '.jk%s'%jkn) for jkn in jk_numbers]
			data_arr = [np.loadtxt(jk, usecols=column) for jk in jk_data]
			data_arr = np.array(data_arr)
			cov = np.cov(data_arr, rowvar=0) * (self.Njk - 1.)**2. / self.Njk # jackknife re-norm without weights for now -- assume samples are not too diverse
			stdev = np.sqrt(np.diag(cov))
			mean = data_arr.mean(axis=0)
			data = ascii.read(self.outfiles[i])
			data['jackknife_err'] = stdev
			data['jackknife_mean'] = mean
			data.write(self.outfiles[i], format='ascii.commented_header', overwrite=1)
			np.savetxt(self.outfiles[i].replace('.dat', '.cov'), cov)
			#except:
				#print '== Jackknife collection failed for corr # %s: '%i, self.paths_data1[i], '--', self.paths_data2[i]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'config_file',
		help='path to configuration file for correlations')
	parser.add_argument(
		'-bin_slop',
		type=float,
		default=0.4,
		help='specify treecorr bin_slop parameter (float, default=0.4)')
	parser.add_argument(
		'-num_threads',
		type=int,
		default=16,
		help='specify number of processors available (default=16)')
	parser.add_argument(
		'-save_cats',
		type=int,
		default=0,
		help='save out fits catalogues (per correlation) for inspection')
#	parser.add_argument(
#		'-remask_randoms',
#		type=int,
#		default=0,
#		help='DO NOT USE; causing bias by masking underdensities -- rewrite to take fits-mask file. Creates nside=2048 Healpix mask from data for application to randoms, set to 0 for no masking (default)')
	parser.add_argument(
		'-verbosity',
		type=int,
		default=0,
		help='specify treecorr verbosity (int[0,3], default=0)')
	parser.add_argument(
		'-down',
		type=float,
		default=10.,
		help='specify factor = Nrandoms / Ngalaxies (float, default=10. // set to 0. for no downsampling)')
	parser.add_argument(
		'-p',
		type=str,
		nargs='*',
		help='override config-file parameters e.g. -p section.param=value (make sure no trailing slashes in paths)')
	parser.add_argument(
		'-index',
		type=int,
		nargs='*',
		help='give indices of correlations in the config file that you desire to run -- others will be skipped. E.g. -index 0 will run only the first correlation')
	args = parser.parse_args()

	Corr = Correlate(args)
	if Corr.run_jackknife == 1:
		Corr.run_loop(args, run_jackknife=1)
		Corr.collect_jackknife()
	elif Corr.run_jackknife == 2:
		Corr.run_loop(args)
		Corr.run_loop(args, run_jackknife=1)
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










#		if args.remask_randoms:
#			#import healpy as hp
#			#nside = 2048
#			#npix = hp.nside2npix(nside)
#			#gpid = hp.ang2pix(nside, d1[ra_col], d1[dec_col], lonlat=True)
#			#rpid = hp.ang2pix(nside, r1[ra_col], r1[dec_col], lonlat=True)
#			#gmap = np.bincount(gpid, minlength=npix)
#			#pixcut = np.where(gmap[rpid] == 0, False, True)
#			#r1 = r1[pixcut]
#			#if not auto:
#			#	gpid = hp.ang2pix(nside, d2[ra_col], d2[dec_col], lonlat=True)
#			#	rpid = hp.ang2pix(nside, r2[ra_col], r2[dec_col], lonlat=True)
#			#	gmap = np.bincount(gpid, minlength=npix)
#			#	pixcut = np.where(gmap[rpid] == 0, False, True)
#			#	r2 = r2[pixcut]
