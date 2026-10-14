"""Supplementary diagnostic (not in the paper): does a single corpus' shared updates leak gender
to an attacker who has shadow data from the *same* corpus?

One private FL run per (corpus, normalisation, FL algorithm). The recorded client updates are split
by speaker into two disjoint halves (2-fold, swapped); a linear probe (standardise -> PCA(200) for
the first layer -> logistic regression) is trained on one half and tested on the other.
Reports balanced accuracy (= UAR) per layer.
"""
import argparse, warnings
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from data import build_clients
from fl_train import run_fl

warnings.filterwarnings('ignore')
LR = {'fed_sgd': 0.05, 'fed_avg': 0.0005}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat_dir', default='features/emobase')
    ap.add_argument('--out', default='results/indomain_probe.csv')
    ap.add_argument('--datasets', default='crema-d,ravdess,emodb')
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows = []
    for ds in a.datasets.split(','):
        for norm in ['speaker', 'global']:
            for mode in ['fed_sgd', 'fed_avg']:
                c, t = build_clients(a.feat_dir, [ds], 0, norm)
                _, u = run_fl(c, t, mode, LR[mode], log=lambda *x: None)
                spk = np.array([s.rsplit('_', 1)[0] for s in u['client']]); y = u['gender']
                # gender-balanced speaker halves
                halves = [[], []]
                for g in [0, 1]:
                    s = np.unique(spk[y == g]); np.random.RandomState(0).shuffle(s)
                    halves[0] += list(s[::2]); halves[1] += list(s[1::2])
                row = dict(dataset=ds, norm=norm, mode=mode, n_updates=len(y), n_speakers=len(np.unique(spk)))
                for k, layer in [('w0', 'first'), ('w1', 'second'), ('w2', 'last')]:
                    ba = []
                    for h in [0, 1]:
                        tr = np.isin(spk, halves[h]); te = ~tr
                        X = u[k].reshape(len(y), -1).astype(np.float32)
                        X = StandardScaler().fit(X[tr]).transform(X)
                        if X.shape[1] > 2000: X = PCA(200, random_state=0).fit(X[tr]).transform(X)
                        m = LogisticRegression(C=0.01, max_iter=3000).fit(X[tr], y[tr])
                        ba.append(balanced_accuracy_score(y[te], m.predict(X[te])))
                    row[f'{layer}_uar'] = float(np.mean(ba))
                print(row, flush=True); rows.append(row)
                pd.DataFrame(rows).to_csv(a.out, index=False)
