#!/bin/bash
set -e

python skyknife.py test_config.ini

python w_pipeline.py test_config.ini -verbosity 1
python w_pipeline.py test_config.ini -verbosity 1 -p catalogs.corr_types=wth output.out_corrs=wth

python plot_test.py

