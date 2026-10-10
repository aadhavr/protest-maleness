#!/usr/bin/env bash
# Runs every step reported in the write-up, in order.
# Steps whose inputs are missing are skipped with a note.
set -uo pipefail
PY=${PY:-python3}
export PM_EVENT_DATE=exam_date
run() { echo; echo "=== $* ==="; $PY "$@" || echo "!!! step failed or skipped: $*"; }

# Part 1
run 01_exam_shocks.py
run 02_acled.py
run 03_exposure.py
run 04_stack.py
run 04_stack.py --placebo
for y in n_exam_strict_nonparty n_exam_strict_party n_exam_strict n_exam_strict_nonparty_broad; do
  run 05_event_study.py "$y" --total
  run 05b_randomization.py "$y"
done
run 05_event_study.py n_exam_strict_nonparty --placebo --total
run 05c_descriptive.py n_exam_strict_nonparty
run 17_robustness.py n_exam_strict_nonparty
run 24_effect_size.py n_exam_strict_nonparty

# State panel: courts (26-week window) and Google Trends
run 07_trends_build.py
( export PM_WINDOW_POST=26
  run 08_state_panel.py
  run 09_state_event_study.py n_orders_body )
run 08_state_panel.py
for y in trends_control trends_manosphere trends_mra trends_gender_jobs trends_hindi trends_index; do
  run 09_state_event_study.py "$y"
done

# Part 2
run 16_repertoire_compare.py
run 18_reporting.py
run 20_regional.py

# Part 3
run 12_online_frames.py
run 22_online_extra.py
run 23_surveys.py

# Figures
run 26_story_figures.py
