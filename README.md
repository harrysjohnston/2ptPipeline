# 2ptPipeline
2-point statistics pipeline, using TreeCorr, allowing for many correlations in fewer configs, with different cuts/weighting/other choices. Primarily intended for measurement of projected correlations $w(r_p)$, but can also measure angular clustering $w(\theta)$, and recently included non-projected clustering/IA $\xi(r)$ for simulation boxes. skyknife.py defines jackknife regions in catalogues, for covariance estimation. Only FITS catalogue formats are supported.



Requirements: TreeCorr, astropy, configparser

# Usage

Basic usage is:
> python w_pipeline.py <config/file> <command-line arguments>

For details of possible command-line arguments, do:
> python w_pipeline.py -h

See also the comments in the config.ini file.

# Jackknife usage

Using same config file as for w_pipeline.py, run skyknife with:
> python skyknife.py <path/to/config/file> <command-line arguments>

And similarly for details, run:
> python skyknife.py -h

Skyknife should be run against a uniform randoms catalogue before running w_pipeline. It is important that the catalogue used for jackknife definition be uniformly distributed, so that regions are driven by area and not density. Jackknife regions so defined can be exported to other catalogues using the exportto= argument in the config file, as well as column names for 3D coordinates in those files. See additional comments in config.

For use of the kmeans clustering routine, clone into this repository https://github.com/esheldon/kmeans_radec and follow the setup instructions, before running skyknife.py with the -kmeans command-line argument.
  
Unfortunately never got round to doing something more clever with jackknifes than just looping over the samples and re-measuring each correlation - lots of duplicated computations. If one writes code to somehow track the jackknife_IDs of galaxy pairs, then the jackknife measurements could be assembled after a single run agains the catalogue.

# Test installation

Do ./run_tests.sh and expect to see a little plot of w(theta), w_g+, w_gx, and w_gg for a small test sample, after about a minute of computation (16 threads). Should match the image in the repository (values not meaningful).
