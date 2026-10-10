#!/usr/bin/env bash
# Data collection. Several steps need API keys or files downloaded by hand (see README).
# Run the steps one at a time the first time and read each output.
set -uo pipefail
PY=${PY:-python3}
cd collect
run() { echo; echo "=== $* ==="; $PY "$@" || echo "!!! step failed or skipped: $*"; }
run c01_acled_api.py                  # ACLED_USERNAME, ACLED_PASSWORD
run c02b_gdelt_bulk.py                # exam-leak candidates; verify each one by hand
run c00_census_download.py            # Census 2011 C-13 state files
run c00b_census_pca_download.py       # Census 2011 PCA (district covariates)
run c03_census.py
run c10_acled_spatial_xwalk.py        # needs geopandas; run after a first 02_acled.py
run c04_courts_s3.py                  # large download
run c04b_court_states.py
run c06_youtube_comments.py           # YOUTUBE_API_KEY
run c08_trends_pull.py                # slow
run c09_classify_llm.py acled         # ANTHROPIC_API_KEY
run c09_classify_llm.py petitioners
run c09_classify_llm.py comments
