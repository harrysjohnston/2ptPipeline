# coding: utf-8
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
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
	def __init__(self, config, catpath=None, ra_col=None, dec_col=None, r_col=None, z_col=None,
					do_3d=None, min_depth=None, sc_tol=None, sz_tol=None, min_scale_deg=None, plot=None):
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
		self.sz_tol = float(cp.get('jackknife', 'quarter_tol'))
		self.sc_tol = float(cp.get('jackknife', 'scale_tol'))
		self.min_scale_deg = float(cp.get('jackknife', 'minimum_scale'))
		self.plot = cp.get('jackknife', 'plot', fallback=None)
		if ra_col is not None:
			self.ra_col = ra_col
		if dec_col is not None:
			self.dec_col = dec_col
		if z_col is not None:
			self.z_col = z_col
		if do_3d is not None:
			self.do_3d = int(do_3d)
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

	def create_jackknife(self, groups=None):
		if groups is None:
			groups = self.define_initial_grouping()
			stop = False
			while not stop:
				groups, stop = self.iterate_quarter_group(groups)
			print "==== %s (2D) jackknife samples " % len(groups)

			if self.do_3d:
				groups = self.slice_jackknife(groups)

		# assign a jackknife ID to each galaxy in the catalogue
		ra = self.cat[self.ra_col]
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

		if self.plot:
			self.plot_jackknife()

		return groups

	def plot_jackknife(self, **kwargs):
		f, ax = plt.subplots(figsize=(10,8))
		cat = fits.open(self.catpath)[1].data
		lencat = len(cat)
		if lencat > 1e5:
			f = 1e5 / float(lencat)
			cat = cat[np.random.rand(lencat) <= f]
		for i in np.unique(cat['jackknife_ID']):
			if i == 0: continue
			cut = cat['jackknife_ID'] == i
			ra = cat['ra'][cut]
			dec = cat['dec'][cut]
			if 's' not in kwargs.keys():
				kwargs['s'] = 1
			plt.scatter(ra, dec, **kwargs)
			count_percentage = 100*len(ra)/float(len(cat))
			ra_range = 'ra:%.2f'%(ra.max() - ra.min())
			dec_range = 'dec:%.2f'%(dec.max() - dec.min())
			plt.annotate('\n'.join(('#%s'%i, '%.1f%%'%count_percentage, ra_range, dec_range)),
						xy=(ra.mean(), dec.mean()), xycoords='data', fontsize=12, ha='center', va='center')
			plt.xlabel('RA')
			plt.ylabel('DEC')
			#plt.savefig(self.plot, bbox_inches='tight')
			plt.show()
			plt.tight_layout()

	def slice_jackknife(self, groups, zbound=None, nbin=None):
		print '== Attempting to slice samples in redshift..'
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

		print '==== %s slices; %s (3D) jackknife samples' % (nbin, len(final_groups))
		print '== redshift slice edges: ', zedge
		print '== comoving slice edges: ', redge
		print '== comoving slice depths', np.diff(redge)
		return final_groups

	def define_initial_grouping(self, discont_factor_ra=100, discont_factor_dec=100):
		ra, dec = self.cat[self.ra_col], self.cat[self.dec_col]
		cat_radec = np.column_stack((ra, dec))
		sra = np.sort(ra)
		sdec = np.sort(dec)
		dra = np.diff(sra)
		ddec = np.diff(sdec)
		discont_ra = np.where(dra > discont_factor_ra*dra.mean())[0]
		discont_dec = np.where(ddec > discont_factor_dec*ddec.mean())[0]
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
		print '== %s groups; attempting to quarter..'%len(groups)
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
				 (abs(ra_scale - dec_scale) < 0.5*max(ra_scale, dec_scale)) ):
				#print '%.2f, %.2f, %.2f'%(mean_scale, ra_scale, dec_scale)
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
		gg_radec = np.column_stack((gg[self.ra_col], gg[self.dec_col]))
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
						   'wrap':gd['wrap']} for (q, dq) in [(xrange(len(q1)+len(q4)), dims_q14),
															  (xrange(len(q2)+len(q3)), dims_q23)]]
		elif (not divide_ra) & divide_dec:
			dims_q12 = dims_q1[0], dims_q2[1], dims_q1[2], dims_q1[3]
			dims_q34 = dims_q4[0], dims_q3[1], dims_q3[2], dims_q3[3]
			new_groups = [{'gal_ra1':dq[0],
						   'gal_ra2':dq[1],
						   'gal_dec1':dq[2],
						   'gal_dec2':dq[3],
						   'count':len(q),
						   'wrap':gd['wrap']} for (q, dq) in [(xrange(len(q1)+len(q2)), dims_q12),
															  (xrange(len(q3)+len(q4)), dims_q34)]]
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

		ra = self.cat[self.ra_col]
		dec = self.cat[self.dec_col]
		cat_radec = np.column_stack((ra, dec))
		group_galaxy_cut = make_catalog_cut(cat_radec)

		group_galaxies = self.cat[group_galaxy_cut]
		gra = group_galaxies[self.ra_col]
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
		cat_radec = np.column_stack((self.cat[self.ra_col], self.cat[self.dec_col]))
		group_galaxies = self.cat[make_cut(cat_radec)]

		return group_galaxies

import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'config',
		help='path to skyknife config file')
	parser.add_argument(
		'-data',
		type=str,
		nargs=4,
		help='give details of data catalogue corresponding to the randoms:'
				'(path, ra_colname, dec_colname, z_colname) -- will port jackknife sampling to the data')
	parser.add_argument(
		'-p',
		nargs='*',
		type=str,
		default = [''],
		help='with syntax "-p <arg1>=<value1>.." specify any of the following args: catpath, ra_col, dec_col, z_col, r_col, do_3d, sc_tol, sz_tol, min_scale_deg, plot')
	args = parser.parse_args()
	#argsp = [ap.split(' ') for ap in args.p]
	kw = {}
	for arg in args.p:
		try:
			section, value = arg.split('=')
		except:
			continue
		kw[section] = value
	sk = Jackknife(args.config, **kw)
	rand_groups = sk.create_jackknife()

	if args.data is not None:
		catpath, ra_col, dec_col, z_col = args.data
		sk_data = Jackknife(args.config, catpath=catpath,
							ra_col=ra_col, dec_col=dec_col, z_col=z_col **kw)
		data_groups = sk_data.create_jackknife(rand_groups)












# OLD CODE
		#hra, bra = np.histogram(ra, bins='auto')
		#hdec, bdec = np.histogram(dec, bins='auto')
		#if ( (abs(bra[1]-bra[0]) < self.min_scale_deg) |
		#	 (abs(bdec[1]-bdec[0]) < self.min_scale_deg) ):
		#	nbin_ra = (ra.max() - ra.min()) // (self.min_scale_deg * 2)
		#	nbin_dec = (dec.max() - dec.min()) // (self.min_scale_deg * 2)
		#	if nbin_ra < 1:
		#		nbin_ra = (ra.max() - ra.min()) // self.min_scale_deg
		#		if nbin_ra < 1:
		#			nbin_ra = 1
		#	if nbin_dec < 1:
		#		nbin_dec = (dec.max() - dec.min()) // self.min_scale_deg
		#		if nbin_dec < 1:
		#			nbin_dec = 1
		#	hra, bra = np.histogram(ra, bins=int(nbin_ra))
		#	hdec, bdec = np.histogram(dec, bins=int(nbin_dec))
		#iedgera = np.where(hra != 0)[0]
		#iedgedec = np.where(hdec != 0)[0]
		#dhra = np.diff(hra)
		#ddec = np.diff(hdec)
		#for iera in iedgera[:-1]:
		#	if any(hra[iera:iera+1] < abs(dhra[iera])):
		#		iedgera = np.delete(iedgera, np.where(iedgera==iera)[0])
		#for iedec in iedgedec[:-1]:
		#	if any(hdec[iedec:iedec+1] < abs(dhdec[iedec])):
		#		iedgedec = np.delete(iedgedec, np.where(iedgedec==iedec)[0])
		#edgera = np.concatenate((bra[1:][iedgera], bra[:-1][iedgera]))
		#edgedec = np.concatenate((bdec[1:][iedgedec], bdec[:-1][iedgedec]))
		#edgera = np.unique(edgera)
		#edgedec = np.unique(edgedec)
		#dedgera = np.diff(edgera)
		#dedgedec = np.diff(edgedec)
		##if any(dedgera > 1.01*dedgera.min()):
		##	jumpsra = np.concatenate((edgera[1:][dedgera > 1.01*dedgera.min()], edgera[:-1][dedgera > 1.01*dedgera.min()]))
		##else:
		#jumpsra = edgera.copy()
		##if any(dedgedec > 1.01*dedgedec.min()):
		##	jumpsdec = np.concatenate((edgedec[1:][dedgedec > 1.01*dedgedec.min()], edgedec[:-1][dedgedec > 1.01*dedgedec.min()]))
		##else:
		#jumpsdec = edgedec.copy()
		#jumpsra = np.unique(jumpsra)
		#jumpsdec = np.unique(jumpsdec)
