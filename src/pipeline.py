"""End-to-end reproduction for one (private dataset D_p, FL algorithm) setting:
  1. attacker trains 5 shadow FL models on D_s (the two other corpora) and records updates
  2. trains one attack model per leak layer (first / second / last)
  3. private FL training on D_p (5 speaker folds): SER accuracy/UAR + attack accuracy/UAR
     on every update shared by the participating clients.
Results are written to results/<mode>/<D_p>/.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import build_clients
from fl_train import run_fl
from attack import norm_stats, train_attack, predict, scores

ALL = ['crema-d', 'ravdess', 'emodb', 'iemocap', 'msp-improv']
LR = {'fed_sgd': 0.05, 'fed_avg': 0.0005}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--private', required=True, choices=ALL)
    ap.add_argument('--corpora', default='crema-d,ravdess,emodb',
                    help='the three corpora of the rotation; paper: iemocap,crema-d,msp-improv')
    ap.add_argument('--mode', required=True, choices=['fed_sgd', 'fed_avg'])
    ap.add_argument('--feat_dir', default='features/emobase')
    ap.add_argument('--out_dir', default='results')
    ap.add_argument('--rounds', type=int, default=200)
    ap.add_argument('--shadow_per_model', type=int, default=600,
                    help='approx. number of updates recorded from each of the 5 shadow models')
    ap.add_argument('--epochs_first', type=int, default=10)
    ap.add_argument('--epochs_other', type=int, default=30)
    ap.add_argument('--layers', default='first,second,last')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--norm', default='speaker', choices=['speaker', 'global'])
    ap.add_argument('--private_every', type=int, default=1, help='evaluate the attack on every n-th private round')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out_dir, f'{a.norm}_norm', a.mode, a.private); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'log.txt', 'w')
    def log(*s):
        msg = ' '.join(map(str, s)); print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    shadow_ds = [d for d in a.corpora.split(',') if d != a.private]
    lr = LR[a.mode]
    log(f'== D_p={a.private}  D_s={"+".join(shadow_ds)}  {a.mode}  lr={lr}')

    # 1. shadow training
    shadow_parts, shadow_ser = [], []
    for s in range(5):
        clients, test = build_clients(a.feat_dir, shadow_ds, s, a.norm)
        per_round = max(1, int(0.1 * len(clients)))
        every = max(1, math.ceil(a.rounds * per_round / a.shadow_per_model))
        log(f'shadow model {s}: {len(clients)} clients, record every {every} rounds')
        best, upd = run_fl(clients, test, a.mode, lr, rounds=a.rounds, record_every=every, log=log)
        shadow_parts.append(upd); shadow_ser.append(best)
    shadow = {k: np.concatenate([p[k] for p in shadow_parts]) for k in shadow_parts[0]}
    del shadow_parts
    log(f'shadow updates: {len(shadow["gender"])} (male share {shadow["gender"].mean():.2f})')
    st = norm_stats(shadow)

    # 2. attack models
    models, attack_val = {}, {}
    for layer in a.layers.split(','):
        ep = a.epochs_first if layer == 'first' else a.epochs_other
        models[layer], attack_val[layer], hist = train_attack(shadow, layer, st, ep, log=log)
        pd.DataFrame(hist).to_csv(out / f'attack_train_{layer}.csv', index=False)
        log(f'  best val ({layer}): {attack_val[layer]}')
    del shadow

    # 3. private training + attack
    rows = []
    for f in range(5):
        clients, test = build_clients(a.feat_dir, [a.private], f, a.norm)
        log(f'private fold {f}: {len(clients)} clients')
        best, upd = run_fl(clients, test, a.mode, lr, rounds=a.rounds, record_every=a.private_every, log=log)
        row = dict(fold=f + 1, n_updates=len(upd['gender']), ser_acc=best['test_acc'], ser_uar=best['test_uar'])
        for layer, m in models.items():
            pred = predict(m, upd, layer, st)
            sc = scores(upd['gender'], pred)
            # paper's success rate: share of correct predictions per client, averaged over clients
            per_client = pd.Series(pred == upd['gender']).groupby(upd['client']).mean().mean()
            row[f'attack_{layer}_acc'], row[f'attack_{layer}_uar'] = sc['acc'], sc['uar']
            row[f'attack_{layer}_client_asr'] = per_client
        log('  ', row); rows.append(row)
        del upd
    df = pd.DataFrame(rows)
    df.loc[len(df)] = dict(fold='average', **df.drop(columns='fold').mean().to_dict())
    df.to_csv(out / 'private_results.csv', index=False)
    json.dump(dict(norm=a.norm, private=a.private, shadow=shadow_ds, mode=a.mode, lr=lr, rounds=a.rounds,
                   shadow_ser=shadow_ser, attack_val=attack_val), open(out / 'meta.json', 'w'), indent=1, default=float)
    log(df.to_string())
