# coding: utf-8
import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 'large'
import matplotlib.pyplot as plt
from astropy.io import ascii

rpp = 0.4
f, ax = plt.subplots(3, figsize=(6, 7))
for i, f in enumerate(('OUTPUTS/wgp.dat','OUTPUTS/wgg.dat','OUTPUTS/wth.dat')):
	dat = ascii.read(f)
	if i == 0:
		r, wgp, wgx = dat['rnom'], dat['wgplus'], dat['wgcross']
		wgp_err, wgx_err = dat['wgplus_jackknife_err'], dat['wgcross_jackknife_err']
		ax[2].errorbar(r*0.99, r**rpp*wgp ,r**rpp*wgp_err, fmt='.-', capsize=1, label='$w_{g+}$')
		ax[2].errorbar(r*1.01, r**rpp*wgx ,r**rpp*wgx_err, fmt='x:', capsize=1, label='$w_{g\\times}$')
	elif i == 1:
		r, wgg = dat['rnom'], dat['wgg']
		wgg_err = dat['wgg_jackknife_err']
		ax[1].errorbar(r, r*wgg, r*wgg_err, fmt='.-', capsize=1)
		#ax[1].errorbar(r, wgg, wgg_err, fmt='.-', capsize=1)
		#ax[1].set_yscale('log')
	elif i == 2:
		t, w, err = dat['r_nom'], dat['xi'], dat['xi_jackknife_err']
		ax[0].errorbar(t, t*w, t*err, fmt='.-', capsize=1)
for a in ax:
	a.axhline(0, ls=':', c='k')
ax[2].legend()
ax[2].set_ylabel('$r_{p}^{%s}w(r_{p})$'%rpp)
ax[1].set_ylabel('$r_{p}w_{gg}(r_{p})$')
ax[0].set_ylabel('$\\vartheta{w}(\\vartheta)$')
for a in ax:
	a.set_xscale('log')
ax[0].set_xlabel('$\\vartheta\,[\\rm{arcmin}]$')
for a in ax[1:]:
	a.set_xlabel('$r_{p}\,[\\rm{Mpc}/h]$')
plt.tight_layout()
plt.show()

