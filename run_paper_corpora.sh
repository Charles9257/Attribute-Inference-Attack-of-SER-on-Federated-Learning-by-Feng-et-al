#!/usr/bin/env bash
# Exact-corpus replication: IEMOCAP + CREMA-D + MSP-Improv with the paper's full budget (GPU recommended).
# IEMOCAP_DIR / MSPIMPROV_DIR = folders holding the official release layout
#   IEMOCAP:    Session{1..5}/dialog/EmoEvaluation/*.txt and Session*/sentences/wav/**.wav
#   MSP-Improv: Evalution.txt and Audio/**/MSP-IMPROV-*.wav
# e.g. with kagglehub:  python -c "import kagglehub; print(kagglehub.dataset_download('columbine/iemocap'))"
set -e
IEMOCAP_DIR=${IEMOCAP_DIR:?set IEMOCAP_DIR}; MSPIMPROV_DIR=${MSPIMPROV_DIR:?set MSPIMPROV_DIR}
FEAT=features/emobase
python3 src/extract_features.py --dataset iemocap    --data_dir "$IEMOCAP_DIR"    --out_dir $FEAT
python3 src/extract_features.py --dataset msp-improv --data_dir "$MSPIMPROV_DIR" --out_dir $FEAT
python3 src/extract_features.py --dataset crema-d    --data_dir data/crema-d      --out_dir $FEAT
for mode in fed_sgd fed_avg; do
  for dp in iemocap crema-d msp-improv; do
    python3 src/pipeline.py --private $dp --mode $mode --norm speaker --corpora iemocap,crema-d,msp-improv \
      --shadow_per_model 100000 --epochs_first 50 --epochs_other 50 --private_every 1 \
      --feat_dir $FEAT --out_dir results_paper_corpora
  done
done
