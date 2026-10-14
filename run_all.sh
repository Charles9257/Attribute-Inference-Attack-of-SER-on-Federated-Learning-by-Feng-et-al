#!/usr/bin/env bash
# Full reproduction pipeline (CPU is enough; ~1 h per setting on 4 cores).
set -e
FEAT=features/emobase
# 1. data (see README "Datasets") -> data/crema-d/{wav,VideoDemographics.csv}, data/ravdess/Actor_*, EmoDB via audb
python3 src/extract_features.py --dataset crema-d --data_dir data/crema-d --out_dir $FEAT
python3 src/extract_features.py --dataset ravdess --data_dir data/ravdess --out_dir $FEAT
python3 src/extract_features.py --dataset emodb   --data_dir data/emodb/wav --out_dir $FEAT
# 2. FL training + shadow training + attack, for every private dataset and FL algorithm
# (settings used for the reported CPU results; see README for the paper's full budget)
for norm in speaker global; do
  for mode in fed_sgd fed_avg; do
    for dp in crema-d ravdess emodb; do
      python3 src/pipeline.py --private $dp --mode $mode --norm $norm --epochs_first 8 --shadow_per_model 500 --private_every 2 --feat_dir $FEAT --out_dir results
    done
  done
  python3 src/summarize.py --results results/${norm}_norm
done
python3 src/probe_indomain.py --feat_dir $FEAT --out results/indomain_probe.csv
