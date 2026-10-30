#!/usr/bin/env bash
# Runs the whole project in order. The first run downloads the data into data/.
set -e
for s in part0_data part1_xt part1b_chart part2_league part2b_chart part3_how part3b_chart part3c_fast_attacks \
         part4_players part4b_chart checks; do
  echo "== $s"
  python "$s.py"
done
