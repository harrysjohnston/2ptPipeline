# coding: utf-8
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import gc
from astropy.io import fits
from astropy.table import Table
import configparser
from os.path import join, expandvars, basename
cp = configparser.ConfigParser()
def betwixt(ra1, ra2, dec1, dec2):
	def make_cut(cat):
		return ((cat.T[0] >= ra1) &
				(cat.T[0] < ra2) &
				(cat.T[1] >= dec1) &
				(cat.T[1] < dec2))
	return make_cut
def betwixt_wrap(ra1, ra2, dec1, dec2):
	def make_cut(cat1):
		cat = cat1.copy()
		cat[:, 0] = np.where(cat[:, 0] < ra1, cat[:, 0] + 360, cat[:, 0])
		return ((cat.T[0] >= ra1) &
				(cat.T[0] < ra2) &
				(cat.T[1] >= dec1) &
				(cat.T[1] < dec2))
	return make_cut
def betwixt_z(z1, z2):
	def make_cut(z):
		return (z >= z1) & (z < z2)
	return make_cut

class Jackknife:
	def __init__(self, config, catpath=None, ra_col=None, dec_col=None, r_col=None, z_col=None, shiftra=None,
					do_3d=None, min_depth=None, zlims=None, sc_tol=None, sz_tol=None, min_scale_deg=None, plot=None):
		cp.read(config)
		self.catpath = expandvars(cp.get('jackknife', 'catalog'))
		if catpath is not None:
			self.catpath = expandvars(catpath)
		self.cat = fits.open(self.catpath)[1].data
		self.ra_col = cp.get('jackknife', 'ra_col')
		self.dec_col = cp.get('jackknife', 'dec_col')
		self.z_col = cp.get('jackknife', 'z_col')
		self.r_col = cp.get('jackknife', 'r_col')
		self.do_3d = int(cp.get('jackknife', 'do_3d'))
		self.min_depth = float(cp.get('jackknife', 'minimum_depth'))
		self.zlims = [float(zl) for zl in cp.get('jackknife', 'zlims').split(' ')]
		self.sz_tol = float(cp.get('jackknife', 'quarter_tol'))
		self.sc_tol = float(cp.get('jackknife', 'scale_tol'))
		self.min_scale_deg = float(cp.get('jackknife', 'minimum_scale'))
		self.plot = cp.get('jackknife', 'plot', fallback=None)
		self.shiftra = float(cp.get('jackknife', 'shiftra', fallback=None))
		if ra_col is not None:
			self.ra_col = ra_col
		if dec_col is not None:
			self.dec_col = dec_col
		if z_col is not None:
			self.z_col = z_col
		if do_3d is not None:
			self.do_3d = int(do_3d)
		if zlims is not None:
			self.zlims = zlims
		if sc_tol is not None:
			self.sc_tol = float(sc_tol)
		if sz_tol is not None:
			self.sz_tol = float(sz_tol)
		if min_scale_deg is not None:
			self.min_scale_deg = float(min_scale_deg)
		if min_depth is not None:
			self.min_depth = float(min_depth)
		if plot is not None:
			self.plot = int(plot)
		if (shiftra is not None or
			self.shiftra is not None):
			self.shiftra = float(shiftra or self.shiftra)
			self.mod = lambda ra: np.where(ra > self.shiftra, ra - 360., ra)
		else:
			self.mod = lambda ra: 1. * ra

		exportlist = expandvars(cp.get('jackknife', 'exportto', fallback='')).replace('\n', '').replace(' //', '//')
		if exportlist != '':
			# initialise exports of the jackknife to other catalogues
			exportsets = exportlist.split('//')
			self.exports = {}
			for cat, racol, decol, zcol in [i.split(' ') for i in exportsets if i != '']:
				self.exports[cat] = racol, decol, zcol

	def create_jackknife(self, groups=None):
		if groups is None:
			groups = self.define_initial_grouping()
			stop = False
			while not stop:
				groups, stop = self.iterate_quarter_group(groups)
			print("==== %s (2D) jackknife samples " % len(groups))

			if self.do_3d:
				groups = self.slice_jackknife(groups, zbound=self.zlims)
			plot = True
		else:
			plot = False

		# assign a jackknife ID to each galaxy in the catalogue
		ra = self.mod(self.cat[self.ra_col])
		dec = self.cat[self.dec_col]
		cat_radec = np.column_stack((ra, dec))
		jackknife_IDs = np.zeros_like(ra, dtype=int)
		for i, group in enumerate(groups):

			if group['wrap']:
				make_galaxy_cut = betwixt_wrap(group['gal_ra1'], group['gal_ra2'], group['gal_dec1'], group['gal_dec2'])
			else:
				make_galaxy_cut = betwixt(group['gal_ra1'], group['gal_ra2'], group['gal_dec1'], group['gal_dec2'])

			galaxy_cut = make_galaxy_cut(cat_radec)

			if self.do_3d:
				make_zslice = betwixt_z(group['gal_z1'], group['gal_z2'])
				z = self.cat[self.z_col]
				redshift_slice = make_zslice(z)
				galaxy_cut &= redshift_slice

			jackknife_IDs[galaxy_cut] = (i + 1)

		if 'jackknife_ID' in self.cat.columns.names:
			self.cat.columns.del_col('jackknife_ID')
		cols = fits.ColDefs([fits.Column(array=jackknife_IDs, format='K', name='jackknife_ID')]) + self.cat.columns
		hdu = fits.BinTableHDU.from_columns(cols)
		hdu.writeto(self.catpath, overwrite=1)

		if self.plot and plot:
			#try:
			self.plot_jackknife()
			#except:
			#	print('== %s plotting failed?'%self.catpath)

		return groups

	def plot_jackknife(self, **kwargs):
		f, ax = plt.subplots(figsize=(16,9))
		cat = fits.open(self.catpath)[1].data
		lencat = len(cat)
		if lencat > 1e5:
			f = 1e5 / float(lencat)
			cat = cat[np.random.rand(lencat) <= f]

		unique_IDs = np.unique(cat['jackknife_ID'])
		unique_IDs = unique_IDs[unique_IDs!=0]

		if self.do_3d:
			cat_z = cat[self.z_col]
			jack_z = [cat_z[cat['jackknife_ID']==i].mean() for i in unique_IDs]
			if len(unique_IDs) // self.nzbin >= 2:
				small_z = np.sort(jack_z)[:len(unique_IDs)//self.nzbin]
			else:
				small_z = np.sort(jack_z)
		else:
			jack_z = np.ones_like(unique_IDs)

		for i, iz in zip(unique_IDs, jack_z):
			if i == 0: continue
			if self.do_3d:
				if iz not in small_z: continue

			cut = cat['jackknife_ID'] == i
			try:
				ra = self.mod(cat['ra'][cut])
				dec = cat['dec'][cut]
			except:
				ra = self.mod(cat[self.ra_col][cut])
				dec = cat[self.dec_col][cut]
				
			count_percentage = 100*len(ra)/float(len(cat))
			ra_range = 'ra:%.2f'%(ra.max() - ra.min())
			dec_range = 'dec:%.2f'%(dec.max() - dec.min())

			if 's' not in kwargs.keys():
				kwargs['s'] = 1
			plt.scatter(ra, dec, **kwargs)
			plt.annotate('\n'.join(('#%s'%i, '%.1f%%'%count_percentage, ra_range, dec_range)),
						xy=(ra.mean(), dec.mean()), xycoords='data', fontsize=12, ha='center', va='center')
			plt.xlabel('RA')
			plt.ylabel('DEC')
			plt.title(self.catpath)
			plt.tight_layout()
			plt.savefig(self.catpath.replace('.fits','.jk.png'), bbox_inches='tight')
			plt.show()

	def slice_jackknife(self, groups, zbound=None, nbin=None):
		print('== Attempting to slice samples in redshift..')
		z = self.cat[self.z_col].copy()
		r = self.cat[self.r_col].copy()

		if zbound is None:
			zmin, zmax = np.percentile(z, [0.5, 99.5])
		else:
			zmin, zmax = zbound

		cut = (z > zmin) & (z < zmax)
		z = z[cut]
		r = r[cut]
		sz, sr = np.sort(z), np.sort(r)
		zpdf, zbin = np.histogram(z, bins='auto')
		zpdf = zpdf / float(zpdf.sum())
		zcdf = np.cumsum(zpdf)

		if nbin is None:
			nbin = 30 # start over-thin
		fracs = np.linspace(0., 1., nbin + 1)
		zedge = np.interp(fracs, zcdf, zbin[:-1])
		#zedge = np.append(zedge, zmax)
		redge = np.interp(zedge, sz, sr)
		depth = np.diff(redge).min()

		while float(depth) <= self.min_depth:
			nbin -= 1
			fracs = np.linspace(0., 1., nbin + 1)
			zedge = np.interp(fracs, zcdf, zbin[:-1])
			#zedge = np.append(zedge, zmax)
			redge = np.interp(zedge, sz, sr)
			depth = np.diff(redge).min()

		final_groups = []
		for group in groups:
			z = self.get_group_galaxies(group)[self.z_col]
			for i in range(len(zedge)-1):
				zgroup = group.copy()
				z_in_group = z[(z > zedge[i]) & (z < zedge[i+1])]
				zgroup['gal_z1'] = z_in_group.min()
				zgroup['gal_z2'] = z_in_group.max()
				zgroup['count'] = len(z_in_group)
				final_groups.append(zgroup)

		self.nzbin = nbin
		print('==== %s slices; %s (3D) jackknife samples' % (nbin, len(final_groups)))
		print('== redshift slice edges: ', np.round(zedge, 3))
		print('== comoving slice edges: ', np.round(redge, 1))
		print('== comoving slice depths', np.round(np.diff(redge), 1))
		return final_groups

	def define_initial_grouping(self, discont_tol_ra=1., discont_tol_dec=1.):
		# discont_tolerances give the minimum gap [in deg] to
		# be considered a gap in the survey footprint
		print('== finding initial grouping..')
		ra, dec = self.mod(self.cat[self.ra_col]), self.cat[self.dec_col]
		cat_radec = np.column_stack((ra, dec))
		sra = np.sort(ra)
		sdec = np.sort(dec)
		dra = np.diff(sra)
		ddec = np.diff(sdec)
		discont_ra = np.where(dra > discont_tol_ra)[0]
		discont_dec = np.where(ddec > discont_tol_dec)[0]
		jumpsra = np.array([ra.min(), ra.max()])
		jumpsdec = np.array([dec.min(), dec.max()])
		jumpsra = np.append(jumpsra, sra[discont_ra])
		jumpsdec = np.append(jumpsdec, sdec[discont_dec])
		jumpsra = np.append(jumpsra, sra[discont_ra+1])
		jumpsdec = np.append(jumpsdec, sdec[discont_dec+1])
		jumpsra = np.sort(jumpsra)
		jumpsdec = np.sort(jumpsdec)
		jumpsra = np.insert(jumpsra, len(jumpsra), 360.+jumpsra[0])
		#jumpsdec = np.insert(jumpsdec, 0, -90.)
		#jumpsdec = np.insert(jumpsdec, len(jumpsdec), 90.)
		nra = len(jumpsra) - 1
		ndec = len(jumpsdec) - 1

		subset_counts = np.zeros([nra, ndec], dtype=int)
		for i in range(nra):
			for j in range(ndec):
				if i == nra-1:
					make_cut = betwixt_wrap(jumpsra[i], jumpsra[i+1], jumpsdec[j], jumpsdec[j+1])
				else:
					make_cut = betwixt(jumpsra[i], jumpsra[i+1], jumpsdec[j], jumpsdec[j+1])
				cut = make_cut(cat_radec)
				subset_counts[i, j] += int(cut.sum())

		groups = []
		for i in range(subset_counts.shape[0]):
			for j in range(subset_counts.shape[1]):
				if subset_counts[i, j] != 0:
					if subset_counts[i, j] < 0.1*subset_counts.max():
						continue
					group_dict = {'ra1':jumpsra[i],
								  'ra2':jumpsra[i+1],
								  'dec1':jumpsdec[j],
								  'dec2':jumpsdec[j+1],
								  'count':subset_counts[i, j],
								  'wrap':jumpsra[i+1]>360.}
					groups.append(group_dict)

		return groups

	def iterate_quarter_group(self, groups):
		print('== %s groups; attempting to quarter..'%len(groups))
		final_groups = []
		for group in groups:
			new_groups = self.quarter_group(group) # will return 1, 2 or 4 group dicts with ~same dims
			ms = []
			for ng in new_groups:
				ra_scale, dec_scale = self.get_group_dims(ng)
				mean_scale_i = (ra_scale + dec_scale) / 2.
				ms.append(mean_scale_i)
			mean_scale = np.array(ms).mean()
			if ( (mean_scale >= self.min_scale_deg) &
				 (ra_scale >= self.sc_tol * self.min_scale_deg) &
				 (dec_scale >= self.sc_tol * self.min_scale_deg) ):
				 #(abs(ra_scale - dec_scale) < 0.5*max(ra_scale, dec_scale)) ):
				#print('%.2f, %.2f, %.2f'%(mean_scale, ra_scale, dec_scale))
				for ng in new_groups:
					final_groups.append(ng)
			else:
				final_groups.append(group)
		if len(final_groups) == len(groups):
			stop = True
		else:
			stop = False

		return final_groups, stop

	def quarter_group(self, group_dict):
		gd = group_dict
		ra_scale, dec_scale = self.get_group_dims(gd)
		half_ra = gd['gal_ra1'] + ra_scale/2.
		half_dec = gd['gal_dec1'] + dec_scale/2.
		# define quadrants clockwise from 'top-left' of an ra, dec square
		dims_q1 = gd['gal_ra1'], half_ra, half_dec, gd['gal_dec2']
		dims_q2 = half_ra, gd['gal_ra2'], half_dec, gd['gal_dec2']
		dims_q3 = half_ra, gd['gal_ra2'], gd['gal_dec1'], half_dec
		dims_q4 = gd['gal_ra1'], half_ra, gd['gal_dec1'], half_dec

		if gd['wrap']:
			mq1 = betwixt_wrap(*dims_q1)
			mq2 = betwixt_wrap(*dims_q2)
			mq3 = betwixt_wrap(*dims_q3)
			mq4 = betwixt_wrap(*dims_q4)
		else:
			mq1 = betwixt(*dims_q1)
			mq2 = betwixt(*dims_q2)
			mq3 = betwixt(*dims_q3)
			mq4 = betwixt(*dims_q4)

		gg = self.get_group_galaxies(gd)
		gg_radec = np.column_stack((self.mod(gg[self.ra_col]), gg[self.dec_col]))
		q1 = gg[mq1(gg_radec)]
		q2 = gg[mq2(gg_radec)]
		q3 = gg[mq3(gg_radec)]
		q4 = gg[mq4(gg_radec)]

		if ( (abs((len(q1)+len(q4)) - (len(q2)+len(q3))) < self.sz_tol * len(gg)) &
			(ra_scale/2. > self.sc_tol * self.min_scale_deg) ):
			divide_ra = True
		else:
			divide_ra = False
		if ( (abs((len(q1)+len(q2)) - (len(q3)+len(q4))) < self.sz_tol * len(gg)) &
			(dec_scale/2. > self.sc_tol * self.min_scale_deg) ):
			divide_dec = True
		else:
			divide_dec = False

		if divide_ra & divide_dec:
			new_groups = [{'gal_ra1':dq[0],
						   'gal_ra2':dq[1],
						   'gal_dec1':dq[2],
						   'gal_dec2':dq[3],
						   'count':len(q),
						   'wrap':gd['wrap']} for (q, dq) in [(q1, dims_q1),
															  (q2, dims_q2),
															  (q3, dims_q3),
															  (q4, dims_q4)]]
		elif divide_ra & (not divide_dec):
			dims_q14 = dims_q1[0], dims_q1[1], dims_q4[2], dims_q1[3]
			dims_q23 = dims_q2[0], dims_q2[1], dims_q3[2], dims_q2[3]
			new_groups = [{'gal_ra1':dq[0],
						   'gal_ra2':dq[1],
						   'gal_dec1':dq[2],
						   'gal_dec2':dq[3],
						   'count':len(q),
						   'wrap':gd['wrap']} for (q, dq) in [(range(len(q1)+len(q4)), dims_q14),
															  (range(len(q2)+len(q3)), dims_q23)]]
		elif (not divide_ra) & divide_dec:
			dims_q12 = dims_q1[0], dims_q2[1], dims_q1[2], dims_q1[3]
			dims_q34 = dims_q4[0], dims_q3[1], dims_q3[2], dims_q3[3]
			new_groups = [{'gal_ra1':dq[0],
						   'gal_ra2':dq[1],
						   'gal_dec1':dq[2],
						   'gal_dec2':dq[3],
						   'count':len(q),
						   'wrap':gd['wrap']} for (q, dq) in [(range(len(q1)+len(q2)), dims_q12),
															  (range(len(q3)+len(q4)), dims_q34)]]
		elif (not divide_ra) & (not divide_dec):
			new_groups = [gd.copy()]

		return new_groups

	def get_group_dims(self, group_dict):
		gd = group_dict
		try:
			if gd['wrap']:
				make_catalog_cut = betwixt_wrap(gd['gal_ra1'], gd['gal_ra2'], gd['gal_dec1'], gd['gal_dec2'])
			else:
				make_catalog_cut = betwixt(gd['gal_ra1'], gd['gal_ra2'], gd['gal_dec1'], gd['gal_dec2'])
		except KeyError:
			if gd['wrap']:
				make_catalog_cut = betwixt_wrap(gd['ra1'], gd['ra2'], gd['dec1'], gd['dec2'])
			else:
				make_catalog_cut = betwixt(gd['ra1'], gd['ra2'], gd['dec1'], gd['dec2'])

		ra = self.mod(self.cat[self.ra_col])
		dec = self.cat[self.dec_col]
		cat_radec = np.column_stack((ra, dec))
		group_galaxy_cut = make_catalog_cut(cat_radec)

		group_galaxies = self.cat[group_galaxy_cut]
		gra = self.mod(group_galaxies[self.ra_col])
		gdec = group_galaxies[self.dec_col]

		if gd['wrap']:
			gd['gal_ra1'] = np.where(gra > gd['ra1'], gra, gra + 360).min()
			gd['gal_ra2'] = np.where(gra < gd['ra1'], gra, gra - 360).max()
			ra_scale = gd['gal_ra2'] - gd['gal_ra1'] + 360.
		else:
			gd['gal_ra1'] = gra.min()
			gd['gal_ra2'] = gra.max()
			ra_scale = gd['gal_ra2'] - gd['gal_ra1']
		gd['gal_dec1'] = gdec.min()
		gd['gal_dec2'] = gdec.max()
		dec_scale = gd['gal_dec2'] - gd['gal_dec1']

		return ra_scale, dec_scale

	def get_group_galaxies(self, group_dict):
		gd = group_dict
		if gd['wrap']:
			make_cut = betwixt_wrap(gd['gal_ra1'], gd['gal_ra2'], gd['gal_dec1'], gd['gal_dec2'])
		else:
			make_cut = betwixt(gd['gal_ra1'], gd['gal_ra2'], gd['gal_dec1'], gd['gal_dec2'])
		cat_radec = np.column_stack((self.mod(self.cat[self.ra_col]), self.cat[self.dec_col]))
		group_galaxies = self.cat[make_cut(cat_radec)]

		return group_galaxies

	def kmeans(self, ncen=40):
		randoms = self.cat.copy()
		X = np.column_stack((self.mod(randoms[self.ra_col]), randoms[self.dec_col]))

		km = kmeans_sample(X, ncen, maxiter=100, tol=1.0e-5, nsample=len(X))
		jk_labels = km.labels + 1
		if self.do_3d:
			jk_labels, zedge = self.slice_kmeans(jk_labels, self.cat, self.z_col, zbound=self.zlims)
			rand_z, rand_r = self.cat[self.z_col], self.cat[self.r_col]
			zsort = np.argsort(rand_z)
			rand_z, rand_r = rand_z[zsort], rand_r[zsort]
			self.comoving = lambda z: np.interp(z, rand_z, rand_r)
		t = Table(self.cat)
		t['jackknife_ID'] = jk_labels
		t.write(self.catpath, overwrite=1)

		if hasattr(self, 'exports'):
			for cat in self.exports.keys():
				print('== exporting jackknife to', cat)
				racol, decol, zcol = self.exports[cat]
				t1 = Table.read(cat)
				X2 = np.column_stack((self.mod(t1[racol]), t1[decol]))
				jk_labels2 = km.find_nearest(X2) + 1
				if self.do_3d:
					jk_labels2, zedge = self.slice_kmeans(jk_labels2, t1, zcol, zbound=self.zlims, zedge=zedge)
				t1['jackknife_ID'] = jk_labels2
				t1.write(cat, overwrite=1)
				del X2, t1
				gc.collect()

		if self.plot:
			#try:
			self.plot_jackknife()
			#except:
			#	print('== %s plotting failed?'%self.catpath)

	def slice_kmeans(self, jk_labels, cat, z_col, zbound=None, nbin=None, zedge=None):
		if zedge is None:
			print('== Attempting to slice samples in redshift..')
		assert len(jk_labels) == len(cat), "==== kmeans redshift slicing broken!"
		if cat is self.cat:
			z1 = self.cat[self.z_col].copy()
			r1 = self.cat[self.r_col].copy()
		else:
			z1 = cat[z_col].copy()
			r1 = self.comoving(z1)

		if zbound is None:
			zmin, zmax = np.percentile(z, [0.5, 99.5])
		else:
			zmin, zmax = zbound

		cut = (z1 > zmin) & (z1 < zmax)
		z = z1[cut]
		r = r1[cut]
		jk_labels[~cut] = 0
		sz, sr = np.sort(z), np.sort(r)
		zpdf, zbin = np.histogram(z, bins='auto')
		zpdf = zpdf / float(zpdf.sum())
		zcdf = np.cumsum(zpdf)

		if zedge is not None:
			nbin = len(zedge) - 1
			redge = np.interp(zedge, sz, sr)
			depth = np.diff(redge).min()
			report = False
		else:
			report = True
			if nbin is None:
				nbin = 30 # start over-thin
			fracs = np.linspace(0., 1., nbin + 1)
			zedge = np.interp(fracs, zcdf, zbin[:-1])
			#zedge = np.append(zedge, zmax)
			redge = np.interp(zedge, sz, sr)
			depth = np.diff(redge).min()

			while float(depth) <= self.min_depth:
				nbin -= 1
				fracs = np.linspace(0., 1., nbin + 1)
				zedge = np.interp(fracs, zcdf, zbin[:-1])
				#zedge = np.append(zedge, zmax)
				redge = np.interp(zedge, sz, sr)
				depth = np.diff(redge).min()

		n_jk = nbin * len(set(jk_labels[jk_labels!=0]))
		jk_labels_sliced = jk_labels.copy()
		i = 1
		for jki in set(jk_labels[jk_labels!=0]):
			for zi in range(nbin):
				jk_labels_sliced[(z1 > zedge[zi]) & (z1 <= zedge[zi+1]) & (jk_labels == jki)] = i
				i += 1
		assert i - 1 == n_jk, "==== kmeans redshift slicing broken!"

		self.nzbin = nbin
		if report:
			print('==== %s slices; %s (3D) jackknife samples' % (nbin, n_jk))
			print('== redshift slice edges: ', np.round(zedge, 3))
			print('== comoving slice edges: ', np.round(redge, 1))
			print('== comoving slice depths', np.round(np.diff(redge), 1))
		return jk_labels_sliced, zedge

import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'config',
		help='Path to skyknife config file')
	parser.add_argument(
		'-kmeans',
		type=int,
		default=0,
		help='Give desired Njk to define jackknife with kmeans_radec, with/out equal-numbers redshift slicing as per config do_3d= argument -- default=0; use skyknife iterative quartering method')
	parser.add_argument(
		'-p',
		nargs='*',
		type=str,
		default = [''],
		help='Override config file arguments, with syntax "-p <arg1>=<value1>.." (beware trailing slashes, and space-separated argments may not work)')
	args = parser.parse_args()
	kw = {}
	for arg in args.p:
		try:
			section, value = arg.split('=')
		except:
			continue
		kw[section] = value
	sk = Jackknife(args.config, **kw)

	if args.kmeans != 0:
		import kmeans_radec
		from kmeans_radec import KMeans, kmeans_sample
		sk.kmeans(ncen=args.kmeans)
	else:
		rand_groups = sk.create_jackknife()
		if hasattr(sk, 'exports'):
			if hasattr(sk.exports, 'iteritems'): details = sk.exports.iteritems()
			else: details = sk.exports.items()
			for cat, (ra, dec, z) in details:
				print('== Exporting to %s..'%cat)
				sk_cat = Jackknife(args.config, catpath=cat,
									ra_col=ra, dec_col=dec, z_col=z)
				sk_cat.create_jackknife(rand_groups)
				del sk_cat
				gc.collect()
			print('== done!')







