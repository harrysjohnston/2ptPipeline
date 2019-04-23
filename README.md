# 2ptPipeline
For development and testing of a 2-point statistics pipeline, using TreeCorr

# Usage: python w_pipeline.py <path/to/config/file> ...

optional arguments:

  -h, --help    show this help message and exit
  
  -bin_slop BIN_SLOP    specify treecorr bin_slop parameter (float, default=0.4)
                        
  -num_threads NUM_THREADS    specify number of processors available (default=16)
                       
  -save_cats SAVE_CATS    save out fits catalogues (per correlation) for inspection
  
  -verbosity VERBOSITY  specify treecorr verbosity (int[0,3], default=0)
  
  -down DOWN            specify factor = Nrandoms / Ngalaxies (float,
                        default=10. // set to 0. for no downsampling)
                        
  -p [P [P ...]]    override config-file parameters e.g. -p section.param=value (make sure no trailing slashes in paths)
                        
  -index [INDEX [INDEX ...]]    give indices of correlations in the config file that you desire to run -- others will be skipped. E.g. -index 0 will run only the first correlation
                        
See also the comments in the example_clustering_config.ini file

# Jackknife: python skyknife.py <path/to/config/file> ...

Run this on each randoms/data catalogue-pair BEFORE running the correlation w_pipeline.py. See notes in [jackknife] section of config file for details.

optional arguments:

  
  -data [4x args]    give data catalogue (1) path, (2) RA column, (3) declination column, (4) redshift column, space-separated -- will port jackknife sampling from randoms into data
                        
  -p [P [P ...]]    override config-file parameters e.g. -p section.param=value (make sure no trailing slashes in paths)

# Note: currently require separate config files for angular/projected stats
