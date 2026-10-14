# Reproduction: Attribute Inference Attack of Speech Emotion Recognition in Federated Learning (Feng et al.)

This repository reproduces

> T. Feng, H. Hashemi, R. Hebbar, M. Annavaram, S. Narayanan. *Attribute Inference Attack of Speech
> Emotion Recognition in Federated Learning Settings.* arXiv:2112.13416 (2021/2022).
> Official code: <https://github.com/usc-sail/fed-ser-leakage>

The paper trains a speech emotion recognition (SER) model with federated learning (FedSGD or FedAvg).
It then shows that an honest-but-curious server can infer each client's **gender** from the model
updates the client shares. The attacker trains **shadow FL models** on public SER corpora (`D_s`) and
records their updates. It then trains a **CNN attack model** on those updates and applies it to the
updates of the private training (`D_p`).

Everything here was re-implemented from the paper and the official code, and it runs on CPU.

## What was reproduced

| Component | Paper / official code | This reproduction |
|---|---|---|
| Features | openSMILE **emobase** (988-d) + 5 SSL models (APC, Vq-APC, NPC, DeCoAR 2.0, TERA) | openSMILE **emobase** (988-d) only. The SSL checkpoints are hosted on Hugging Face/Dropbox, which this environment cannot reach |
| Corpora | IEMOCAP, CREMA-D, MSP-Improv (4 classes: neu/hap/sad/ang) | **CREMA-D** (same corpus). **EmoDB** stands in for IEMOCAP (10 speakers) and **RAVDESS** for MSP-Improv (12 → 24 speakers). IEMOCAP and MSP-Improv require a signed licence |
| Rotation | each corpus is `D_p` once, the other two are `D_s` | same |
| Feature normalisation | z-norm within each speaker | same (`speaker_norm`). We also report `global_norm` (see Finding 1) |
| Clients | CREMA-D: 1 speaker = 1 client. IEMOCAP/MSP: each speaker split into 10 clients | CREMA-D: 1 speaker = 1 client. RAVDESS/EmoDB: each speaker split into **5** clients (only ~30 utterances per speaker) |
| Splits | 5 speaker folds (20% test speakers); 80/20 local train/val in each client | same |
| SER model | 988 → 256 → 128 → 4, ReLU, dropout 0.2 | same (`src/models.py`) |
| FL | 200 global rounds, 10% of clients per round, batch 20. FedSGD lr 0.05 (one-batch gradient). FedAvg Adam lr 5e-4, 1 local epoch, pseudo-gradient `(w_g − w_l)/(lr·steps)` | same (`src/fl_train.py`) |
| Shadow models | 5 (one per fold of `D_s`), every update of every round | 5. To fit CPU memory and time, about 500 updates are kept per shadow model (every k-th round) |
| Attack model | 3-layer CNN on the weight update plus the bias update. Adam lr 1e-4, wd 1e-4, batch 20, 80/20 split, best val | same architecture (`src/models.py`). **8 epochs** for the first layer (the paper uses about 50), 30 for the other layers |
| Metric | success rate per client, averaged over clients (10 random updates per client) | per-update accuracy and UAR, plus the paper's per-client success rate (`*_client_asr`). The attack is evaluated on every 2nd round of all 5 private folds |

## Results

To browse every table and plot, open [`notebooks/results_viewer.ipynb`](notebooks/results_viewer.ipynb). It is already executed and renders on GitHub. To re-run it locally: `pip install jupyter matplotlib && jupyter notebook notebooks/results_viewer.ipynb`.

All numbers are means over the 5 private speaker folds. The paper reports the success rate
(≈ accuracy); we report UAR. Per-update accuracy and the paper-style per-client success rate are
in each `private_results.csv`.

### Main result (paper protocol: within-speaker z-norm)

#### SER performance of the federated model (test speakers, mean of 5 folds), Acc / UAR %

| FL | D_p | ours | paper (emobase) |
|---|---|---|---|
| FedSGD | EmoDB (paper: IEMOCAP) | 84.01 / 83.40 | 62.94 / 61.62 |
| FedSGD | CREMA-D (paper: CREMA-D) | 77.04 / 76.81 | 69.81 / 64.24 |
| FedSGD | RAVDESS (paper: MSP-Improv) | 73.54 / 70.97 | 54.17 / 47.23 |
| FedAvg | EmoDB (paper: IEMOCAP) | 86.90 / 86.90 | 61.22 / 61.25 |
| FedAvg | CREMA-D (paper: CREMA-D) | 79.22 / 79.04 | 71.26 / 66.38 |
| FedAvg | RAVDESS (paper: MSP-Improv) | 72.68 / 69.38 | 54.37 / 47.67 |

#### Gender inference attack, first-layer updates (UAR %, mean of 5 private folds)

| FL | D_p | D_s (shadow) | ours | paper (emobase) |
|---|---|---|---|---|
| FedSGD | EmoDB | CREMA-D+RAVDESS | 46.64 | 89.33 (IEMOCAP) |
| FedSGD | CREMA-D | RAVDESS+EmoDB | 45.55 | 73.87 (CREMA-D) |
| FedSGD | RAVDESS | CREMA-D+EmoDB | 51.94 | 91.36 (MSP-Improv) |
| FedAvg | EmoDB | CREMA-D+RAVDESS | 43.22 | 87.45 (IEMOCAP) |
| FedAvg | CREMA-D | RAVDESS+EmoDB | 49.78 | 81.98 (CREMA-D) |
| FedAvg | RAVDESS | CREMA-D+EmoDB | 50.59 | 82.86 (MSP-Improv) |

#### Which layer leaks most (attack UAR %)

| FL | D_p | first | second | last |
|---|---|---|---|---|
| FedSGD | EmoDB | 46.64 | 48.48 | 49.84 |
| FedSGD | CREMA-D | 45.55 | 52.56 | 50.29 |
| FedSGD | RAVDESS | 51.94 | 51.23 | 49.72 |
| FedAvg | EmoDB | 43.22 | 45.90 | 49.18 |
| FedAvg | CREMA-D | 49.78 | 51.36 | 48.25 |
| FedAvg | RAVDESS | 50.59 | 51.01 | 52.04 |

### Same experiment with global z-norm (statistics of the training speakers)

#### SER performance of the federated model (test speakers, mean of 5 folds), Acc / UAR %

| FL | D_p | ours | paper (emobase) |
|---|---|---|---|
| FedSGD | EmoDB (paper: IEMOCAP) | 79.91 / 78.95 | 62.94 / 61.62 |
| FedSGD | CREMA-D (paper: CREMA-D) | 68.41 / 68.17 | 69.81 / 64.24 |
| FedSGD | RAVDESS (paper: MSP-Improv) | 65.00 / 62.06 | 54.17 / 47.23 |
| FedAvg | EmoDB (paper: IEMOCAP) | 82.90 / 81.30 | 61.22 / 61.25 |
| FedAvg | CREMA-D (paper: CREMA-D) | 69.10 / 68.78 | 71.26 / 66.38 |
| FedAvg | RAVDESS (paper: MSP-Improv) | 65.46 / 61.22 | 54.37 / 47.67 |

#### Gender inference attack, first-layer updates (UAR %, mean of 5 private folds)

| FL | D_p | D_s (shadow) | ours | paper (emobase) |
|---|---|---|---|---|
| FedSGD | EmoDB | CREMA-D+RAVDESS | 47.37 | 89.33 (IEMOCAP) |
| FedSGD | CREMA-D | RAVDESS+EmoDB | 54.12 | 73.87 (CREMA-D) |
| FedSGD | RAVDESS | CREMA-D+EmoDB | 54.77 | 91.36 (MSP-Improv) |
| FedAvg | EmoDB | CREMA-D+RAVDESS | 55.84 | 87.45 (IEMOCAP) |
| FedAvg | CREMA-D | RAVDESS+EmoDB | 49.76 | 81.98 (CREMA-D) |
| FedAvg | RAVDESS | CREMA-D+EmoDB | 50.68 | 82.86 (MSP-Improv) |

#### Which layer leaks most (attack UAR %)

| FL | D_p | first | second | last |
|---|---|---|---|---|
| FedSGD | EmoDB | 47.37 | 52.90 | 55.57 |
| FedSGD | CREMA-D | 54.12 | 55.49 | 57.95 |
| FedSGD | RAVDESS | 54.77 | 54.49 | 58.77 |
| FedAvg | EmoDB | 55.84 | 57.65 | 54.60 |
| FedAvg | CREMA-D | 49.76 | 55.54 | 54.99 |
| FedAvg | RAVDESS | 50.68 | 53.25 | 57.01 |

### Supplementary: same-corpus, speaker-disjoint linear probe (`results/indomain_probe.csv`)

Here the attacker has shadow updates from the *same* corpus, but from different speakers. A logistic
regression on the updates (PCA-200 for the first layer) is trained on half of the speakers and
tested on the other half (2-fold, UAR %, private FL fold 1):

| corpus | norm | FL | first | second | last |
|---|---|---|---|---|---|
| CREMA-D (72 train spk) | speaker | FedSGD | 66.5 | 67.1 | 64.0 |
| CREMA-D (72 train spk) | speaker | FedAvg | 66.1 | 69.0 | 68.1 |
| CREMA-D (72 train spk) | global | FedSGD | 66.0 | 63.4 | 60.9 |
| CREMA-D (72 train spk) | global | FedAvg | 70.6 | 78.2 | 68.9 |
| RAVDESS (19 train spk) | speaker | FedSGD | 56.5 | 61.2 | 56.8 |
| RAVDESS (19 train spk) | speaker | FedAvg | 59.5 | 67.9 | 64.5 |
| RAVDESS (19 train spk) | global | FedSGD | 57.2 | 54.3 | 52.3 |
| RAVDESS (19 train spk) | global | FedAvg | 78.0 | 80.0 | 67.0 |
| EmoDB (8 train spk) | speaker | FedSGD | 55.7 | 55.7 | 54.2 |
| EmoDB (8 train spk) | speaker | FedAvg | 53.4 | 49.6 | 48.1 |
| EmoDB (8 train spk) | global | FedSGD | 54.4 | 50.0 | 48.0 |
| EmoDB (8 train spk) | global | FedAvg | 78.8 | 76.0 | 59.6 |

### Findings

1. **SER (federated emotion recognition) reproduces.** FedSGD and FedAvg reach similar accuracy, as in
   the paper. On CREMA-D (the one corpus we share with the paper) we get 77–79% UAR with
   within-speaker z-norm, against 64–66% in the paper. The paper took CREMA-D labels from the
   crowd-sourced majority vote, while we use the acted labels, which probably explains part of the gap.
   EmoDB and RAVDESS are acted corpora and much easier than IEMOCAP and MSP-Improv.
2. **The cross-corpus attack does not reproduce.** With shadow models trained on the other two
   corpora, the first-layer attack stays at chance (43–56% UAR). The paper reports 74–91%. The
   attack model reaches 64–73% on its own shadow validation split, but that split is a random 80/20
   split of updates (paper protocol), so the same clients appear in training and validation. That
   score largely measures client memorisation, not generalisation to new speakers.
3. **The updates still leak gender.** When the shadow data comes from the same corpus (supplementary
   probe), even a linear model infers the gender of *unseen* CREMA-D speakers from first-layer
   updates with 66–71% UAR. That is close to the paper's CREMA-D figures (74% FedSGD, 82% FedAvg).
   The privacy risk the paper describes is real. What failed here is the transfer from our small
   public shadow corpora (RAVDESS + EmoDB: about 1,000 utterances) to the private corpus.
4. **Within-speaker z-norm removes most first-order gender information.** A gender classifier on the
   normalised emobase features gets 42–56% (linear) and 63–74% (gradient boosting), against 88–93%
   on raw features. The remaining signal is non-linear, so the attacker needs much more shadow
   data and training than our CPU budget allowed.
5. **Layer ablation.** In the cross-corpus setting no layer clearly beats the others, because all are
   near chance. In the in-domain probe the first and second layers leak most, consistent with the
   paper's claim that the first layer is the most informative.

### Why the numbers differ from the paper (in order of likely impact)

* **Different corpora.** IEMOCAP and MSP-Improv are licensed. EmoDB (339 4-class utterances) and
  RAVDESS (672) are 10–20× smaller than what the paper's shadow models trained on, and they are
  acted read speech.
* **Compute.** The first-layer attack ran for 8 epochs on about 2.5k shadow updates. The paper
  used about 50 epochs on every update (about 15–20k). The validation curves were still rising
  when training stopped (`attack_train_first.csv`).
* **Features.** Only emobase. The SSL features (APC, NPC, DeCoAR 2.0, ...) gave the paper's highest
  leakage, but their checkpoints could not be downloaded here.
* **Clients.** RAVDESS and EmoDB speakers are split into 5 clients (the paper split into 10), so
  each client has 3–7 training utterances.


## How to run

```bash
pip install -r requirements.txt
# data (all openly downloadable)
#  CREMA-D : https://github.com/CheyneyComputerScience/CREMA-D  (AudioWAV/*.wav + VideoDemographics.csv) -> data/crema-d/{wav,VideoDemographics.csv}
#  RAVDESS : https://zenodo.org/record/1188976 (Audio_Speech_Actors_01-24.zip)                           -> data/ravdess/Actor_*/
#  EmoDB   : python -c "import audb; audb.load('emodb', version='1.4.1', format='wav', sampling_rate=16000)"  -> data/emodb/wav
./run_all.sh   # features, then 12 FL+attack settings (results/<norm>_norm/summary.md), then the in-domain probe

```

`src/pipeline.py` has flags that bring back the paper's full budget on a GPU machine:
`--shadow_per_model 100000 --epochs_first 50 --private_every 1`.

## Repository layout

```
src/extract_features.py  openSMILE emobase functionals for CREMA-D / RAVDESS / EmoDB
src/data.py              speaker folds, speaker- or global z-norm, client construction
src/models.py            SER MLP and attack CNN (ported from the official code)
src/fl_train.py          FedSGD / FedAvg training that records the shared updates
src/attack.py            normalisation, attack training and evaluation
src/pipeline.py          one (D_p, FL algorithm, normalisation) setting end-to-end
src/summarize.py         builds results/summary.md next to the paper's numbers
src/probe_indomain.py    supplementary: speaker-disjoint, same-corpus linear probe
notebooks/results_viewer.ipynb  tables and plots of all saved results vs the paper
results/<norm>/<fl>/<D_p>/  private_results.csv (per fold), attack_train_*.csv, meta.json, log.txt
```
