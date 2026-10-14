"""Federated data preparation, following preprocess_data/ and read_data_dict_by_client()
of the official code:
  * 5-fold speaker-independent split per dataset (KFold, no shuffle) -> train / test speakers
  * speaker-level z-normalisation of every feature
  * CREMA-D: one client per speaker.  Small corpora (IEMOCAP/MSP-Improv in the paper,
    RAVDESS/EmoDB here): each speaker's data is split into several clients.
  * inside every client 80% of the data is used for local training, 20% for local validation.
"""
import numpy as np
from sklearn.model_selection import KFold

# number of clients each speaker is split into (paper: 10 for IEMOCAP / MSP-Improv).
# RAVDESS and EmoDB only have ~28-34 4-class utterances per speaker, so we use 5.
SPLITS = {'crema-d': 1, 'ravdess': 5, 'emodb': 5}


def load(feat_dir, dataset, norm='speaker', train_speakers=None):
    """norm='speaker': z-normalise within every speaker (as in the paper / official code).
    norm='global' : z-normalise with mean/std of the training speakers of the corpus."""
    d = np.load(f'{feat_dir}/{dataset}.npz')
    x = np.nan_to_num(d['feat'].astype(np.float32))
    spk = np.array([f'{dataset}:{s}' for s in d['speaker']])
    if norm == 'speaker':
        for s in np.unique(spk):
            m = spk == s
            x[m] = (x[m] - x[m].mean(0)) / (x[m].std(0) + 1e-5)
    else:
        m = np.isin(spk, list(train_speakers)) if train_speakers is not None else np.ones(len(x), bool)
        x = (x - x[m].mean(0)) / (x[m].std(0) + 1e-5)
    return dict(x=x, y=d['emo'].astype(np.int64), gender=d['gender'], speaker=spk)


def build_clients(feat_dir, datasets, fold, norm='speaker', seed=8):
    """Return (clients, test) for one fold. clients: list of dicts with
    train_x, train_y, val_x, val_y, gender, name."""
    rng = np.random.RandomState(seed)
    clients, tx, ty = [], [], []
    for ds in datasets:
        speakers = np.array(sorted(set(f'{ds}:{s}' for s in np.load(f'{feat_dir}/{ds}.npz')['speaker'])))
        train_idx, test_idx = list(KFold(5).split(speakers))[fold]
        d = load(feat_dir, ds, norm, speakers[train_idx])
        test_spk = set(speakers[test_idx])
        m = np.isin(d['speaker'], list(test_spk))
        tx.append(d['x'][m]); ty.append(d['y'][m])
        for s in speakers[train_idx]:
            idx = rng.permutation(np.where(d['speaker'] == s)[0])
            for k, part in enumerate(np.array_split(idx, SPLITS[ds])):
                part = rng.permutation(part)
                n_tr = int(0.8 * len(part))
                clients.append(dict(name=f'{s}_{k}', gender=d['gender'][part[0]],
                                    train_x=d['x'][part[:n_tr]], train_y=d['y'][part[:n_tr]],
                                    val_x=d['x'][part[n_tr:]], val_y=d['y'][part[n_tr:]]))
    return clients, dict(x=np.concatenate(tx), y=np.concatenate(ty))
