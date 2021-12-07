# coding: utf-8
import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 'large'
import matplotlib.pyplot as plt
from astropy.io import ascii

f, ax = plt.subplots(2, sharex=True)
for i, f in enumerate(('OUTPUTS/wgp.dat','OUTPUTS/wgg.dat')):
	dat = ascii.read(f)
	if i == 0:
		r, wgp, wgx = dat['rnom'], dat['wgplus'], dat['wgcross']
		wgp_err, wgx_err = dat['wgplus_jackknife_err'], dat['wgcross_jackknife_err']
		ax[0].errorbar(r, r**0.8*wgp ,r**0.8*wgp_err, fmt='.-', capsize=1, label='$w_{g+}$')
		ax[0].errorbar(r, r**0.8*wgx ,r**0.8*wgx_err, fmt='x:', capsize=1, label='$w_{g\\times}$')
	elif i == 1:
		r, wgg = dat['rnom'], dat['wgg']
		wgg_err = dat['wgg_jackknife_err']
		ax[1].errorbar(r, r*wgg, r*wgg_err, fmt='.-', capsize=1)
ax[0].axhline(0, ls=':', c='k')
ax[0].legend()
ax[0].set_ylabel('$r_{p}^{0.8}w(r_{p})$')
ax[1].set_ylabel('$r_{p}w_{gg}(r_{p})$')
ax[1].set_xlabel('$r_{p}\,[\\rm{Mpc}/h]$')
plt.xscale('log')
plt.tight_layout()
plt.show()

