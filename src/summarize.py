"""Collect results/<mode>/<D_p>/private_results.csv into results/summary.md, next to the paper's numbers."""
import argparse
from pathlib import Path
import pandas as pd

# Paper (emobase rows), Tables of arXiv 2112.13416 / official README.
PAPER_SER = {  # (acc, uar)
    'fed_sgd': {'IEMOCAP': (62.94, 61.62), 'CREMA-D': (69.81, 64.24), 'MSP-Improv': (54.17, 47.23)},
    'fed_avg': {'IEMOCAP': (61.22, 61.25), 'CREMA-D': (71.26, 66.38), 'MSP-Improv': (54.37, 47.67)}}
PAPER_ATTACK = {'fed_sgd': {'IEMOCAP': 89.33, 'CREMA-D': 73.87, 'MSP-Improv': 91.36},
                'fed_avg': {'IEMOCAP': 87.45, 'CREMA-D': 81.98, 'MSP-Improv': 82.86}}
# our open substitute for each paper corpus (similar #speakers / client construction)
SUBST = {'crema-d': 'CREMA-D', 'emodb': 'IEMOCAP', 'ravdess': 'MSP-Improv'}
NAME = {'crema-d': 'CREMA-D', 'emodb': 'EmoDB', 'ravdess': 'RAVDESS'}
MODE = {'fed_sgd': 'FedSGD', 'fed_avg': 'FedAvg'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results'); a = ap.parse_args()
    ser, att, lay = [], [], []
    for mode in ['fed_sgd', 'fed_avg']:
        for dp in ['emodb', 'crema-d', 'ravdess']:
            f = Path(a.results, mode, dp, 'private_results.csv')
            if not f.exists(): continue
            r = pd.read_csv(f).set_index('fold').loc['average']
            p = SUBST[dp]; others = '+'.join(NAME[d] for d in ['crema-d', 'ravdess', 'emodb'] if d != dp)
            ser.append(f'| {MODE[mode]} | {NAME[dp]} (paper: {p}) | {100*r.ser_acc:.2f} / {100*r.ser_uar:.2f} | '
                       f'{PAPER_SER[mode][p][0]:.2f} / {PAPER_SER[mode][p][1]:.2f} |')
            att.append(f'| {MODE[mode]} | {NAME[dp]} | {others} | {100*r.attack_first_uar:.2f} | {PAPER_ATTACK[mode][p]:.2f} ({p}) |')
            lay.append(f'| {MODE[mode]} | {NAME[dp]} | ' + ' | '.join(
                f'{100*r[f"attack_{l}_uar"]:.2f}' if f'attack_{l}_uar' in r else '-' for l in ['first', 'second', 'last']) + ' |')
    md = ['## SER performance of the federated model (test speakers, mean of 5 folds), Acc / UAR %', '',
          '| FL | D_p | ours | paper (emobase) |', '|---|---|---|---|', *ser, '',
          '## Gender inference attack, first-layer updates (UAR %, mean of 5 private folds)', '',
          '| FL | D_p | D_s (shadow) | ours | paper (emobase) |', '|---|---|---|---|---|', *att, '',
          '## Which layer leaks most (attack UAR %)', '',
          '| FL | D_p | first | second | last |', '|---|---|---|---|---|', *lay, '']
    Path(a.results, 'summary.md').write_text('\n'.join(md)); print('\n'.join(md))
