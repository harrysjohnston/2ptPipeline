#!/bin/bash
set -e

python skyknife.py test_config.ini

python w_pipeline.py test_config.ini -verbosity 1

python plot_test.py

