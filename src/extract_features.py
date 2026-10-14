"""Extract openSMILE emobase functionals (988-d) for the open SER datasets.

Mirrors feature_extraction/opensmile_feature_extraction.py of the official code
(usc-sail/fed-ser-leakage), using the same 4 emotion classes (neu, hap, sad, ang).

Output: <out_dir>/<dataset>.npz with arrays feat, emo, gender, speaker, files.
"""
import argparse, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import opensmile

EMO = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
_smile = None


def _feat(path):
    global _smile
    if _smile is None:
        _smile = opensmile.Smile(feature_set=opensmile.FeatureSet.emobase,
                                 feature_level=opensmile.FeatureLevel.Functionals)
    return _smile.process_file(str(path)).values[0].astype(np.float32)


def list_crema(root):
    # 1001_DFA_ANG_XX.wav ; gender from VideoDemographics.csv
    demo = pd.read_csv(Path(root, 'VideoDemographics.csv'), index_col=0)
    m = {'ANG': 'ang', 'HAP': 'hap', 'NEU': 'neu', 'SAD': 'sad'}
    rows = []
    for f in sorted(Path(root, 'wav').glob('*.wav')):
        spk, _, emo, _ = f.stem.split('_')
        if emo not in m: continue
        g = 'M' if demo.loc[int(spk), 'Sex'] == 'Male' else 'F'
        rows.append((f, m[emo], g, spk))
    return rows


def list_ravdess(root):
    # 03-01-EE-II-SS-RR-AA.wav ; EE: 01 neutral 03 happy 04 sad 05 angry ; odd actor = male
    m = {'01': 'neu', '03': 'hap', '04': 'sad', '05': 'ang'}
    rows = []
    for f in sorted(Path(root).glob('Actor_*/*.wav')):
        p = f.stem.split('-')
        if p[0] != '03' or p[1] != '01' or p[2] not in m: continue
        actor = int(p[6])
        rows.append((f, m[p[2]], 'M' if actor % 2 == 1 else 'F', 'A%02d' % actor))
    return rows


def list_emodb(root):
    # 03a01Fa.wav ; speaker = first 2 chars ; emotion letter at idx 5
    m = {'N': 'neu', 'F': 'hap', 'T': 'sad', 'W': 'ang'}
    male = {'03', '10', '11', '12', '15'}
    rows = []
    for f in sorted(Path(root).glob('*.wav')):
        e = f.stem[5]
        if e not in m: continue
        spk = f.stem[:2]
        rows.append((f, m[e], 'M' if spk in male else 'F', spk))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['crema-d', 'ravdess', 'emodb'])
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default='features/emobase')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    rows = {'crema-d': list_crema, 'ravdess': list_ravdess, 'emodb': list_emodb}[a.dataset](a.data_dir)
    with Pool(a.workers) as p:
        feats = p.map(_feat, [r[0] for r in rows], chunksize=16)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.out_dir, a.dataset + '.npz'), feat=np.stack(feats),
             emo=np.array([EMO[r[1]] for r in rows]), gender=np.array([r[2] for r in rows]),
             speaker=np.array([r[3] for r in rows]), files=np.array([r[0].name for r in rows]))
    print(a.dataset, len(rows), 'utterances,', len({r[3] for r in rows}), 'speakers')
