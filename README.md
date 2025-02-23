# 2ptPipeline
2-point statistics pipeline, using TreeCorr, allowing for many correlations in fewer configs, with different cuts/weighting/other choices. Primarily intended for measurement of projected correlations $w(r_p)$, but can also measure angular clustering $w(\theta)$, and recently included non-projected clustering/IA $\xi(r)$ for simulation boxes. skyknife.py defines jackknife regions in catalogues, for covariance estimation. Only FITS catalogue formats are supported.

Requirements: TreeCorr, astropy, configparser

# Usage

Basic usage is:
> python w_pipeline.py <config/file> <command-line arguments>

For details of possible command-line arguments, do:
> python w_pipeline.py -h

See also the numerous comments in the config.ini file.

# Jackknife usage

Using same config file as for w_pipeline.py, run skyknife with:
> python skyknife.py <path/to/config/file> <command-line arguments>

And similarly for details, run:
> python skyknife.py -h

Skyknife should be run against a uniform randoms catalogue before running w_pipeline. It is important that the catalogue used for jackknife definition be uniformly distributed, so that regions are driven by area and not density. Jackknife regions so defined can be exported to other catalogues using the exportto= argument in the config file, as well as column names for 3D coordinates in those files. See additional comments in config.

For use of the kmeans clustering routine, clone into this repository https://github.com/esheldon/kmeans_radec and follow the setup instructions, before running skyknife.py with the -kmeans command-line argument.
  
Unfortunately never got round to doing something more clever with jackknifes than just looping over the samples and re-measuring each correlation - lots of duplicated computations. If one writes code to somehow track the jackknife_IDs of galaxy pairs, then the jackknife measurements could be assembled after a single run against the catalogue.

TreeCorr also implements a jackknife routine (which is collected much more efficiently), which is supported within this pipeline by the config argument jackknife.treejack=N -- requests N subsamples to be defined via kmeans, in 2D (3D) for RADEC (XYZ). This argument will override the jackknife.run= config argument that otherwise governs the jackknife covariance computation, but treejack is only supported for angular correlations w(theta) and Cartesian 3D clustering xi_gg(r) -- this is because the other 2-point functions involve auxiliary computations that will render the TreeCorr covariance output meaningless.

# Test installation

Do ./run_tests.sh and expect to see a little plot of w(theta), w_g+, w_gx, and w_gg for a small test sample, after about a minute of computation (16 threads). Should match the image in the repository (values not meaningful).

# Credit

This pipeline is released under a CC-BY 4.0 license: https://creativecommons.org/licenses/by/4.0/ .

If making use of the pipeline, please cite:
```
@ARTICLE{2019A&A...624A..30J,
       author = {{Johnston}, Harry and {Georgiou}, Christos and {Joachimi}, Benjamin and {Hoekstra}, Henk and {Chisari}, Nora Elisa and {Farrow}, Daniel and {Fortuna}, Maria Cristina and {Heymans}, Catherine and {Joudaki}, Shahab and {Kuijken}, Konrad and {Wright}, Angus},
        title = "{KiDS+GAMA: Intrinsic alignment model constraints for current and future weak lensing cosmology}",
      journal = {\aap},
     keywords = {gravitational lensing: weak, cosmology: observations, large-scale structure of Universe, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2019,
        month = apr,
       volume = {624},
          eid = {A30},
        pages = {A30},
          doi = {10.1051/0004-6361/201834714},
archivePrefix = {arXiv},
       eprint = {1811.09598},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019A&A...624A..30J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
