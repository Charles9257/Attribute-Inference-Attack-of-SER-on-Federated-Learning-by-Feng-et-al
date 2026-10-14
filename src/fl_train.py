"""Federated SER training (FedSGD / FedAvg) that records the model updates shared by the
selected clients in every global round, following train/federated_ser_classifier.py.

FedSGD : client sends the gradient of one local batch (20 utterances); server does an SGD step.
FedAvg : client trains 1 local epoch with Adam and sends its weights; the recorded update is the
         'pseudo gradient' (w_global - w_local) / (lr * local_steps), as in the official code.
10% of the clients participate in each of the 200 global rounds.
"""
import copy, math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score
from models import SERClassifier

GENDER = {'F': 0, 'M': 1}


def _batches(x, y, bs, rng):
    idx = rng.permutation(len(x))
    for i in range(0, len(x), bs):
        j = idx[i:i + bs]
        yield torch.from_numpy(x[j]), torch.from_numpy(y[j])


def _evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(x)).argmax(1).numpy()
    return accuracy_score(y, p), recall_score(y, p, average='macro', zero_division=0)


def run_fl(clients, test, mode, lr, rounds=200, frac=0.1, batch_size=20, dropout=0.2,
           record_every=1, seed=8, log=print):
    """Train one federated SER model. Returns (ser_result, updates) where updates holds the
    per-client shared updates for every `record_every`-th round."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    crit = nn.NLLLoss()
    model = SERClassifier(test['x'].shape[1], dropout=dropout)
    n = len(clients)
    rec = {k: [] for k in ['w0', 'b0', 'w1', 'b1', 'w2', 'b2']}
    rec_gender, rec_round, rec_client = [], [], []
    best = dict(val_uar=-1)
    for r in range(rounds):
        sel = np.random.RandomState(r).choice(n, max(1, int(frac * n)), replace=False)
        g_state = copy.deepcopy(model.state_dict())
        ups, sizes = [], []
        for ci in sel:
            c = clients[ci]
            local = copy.deepcopy(model); local.train()
            if mode == 'fed_sgd':
                xb, yb = next(_batches(c['train_x'], c['train_y'], batch_size, rng))
                local.zero_grad()
                crit(local(xb), yb).backward()
                upd = [p.grad.detach().clone() for p in local.parameters()]
                n_samples = len(yb)
            else:
                opt = torch.optim.Adam(local.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-9)
                for xb, yb in _batches(c['train_x'], c['train_y'], batch_size, rng):
                    opt.zero_grad(); crit(local(xb), yb).backward(); opt.step()
                n_samples = len(c['train_y'])
                steps = int(n_samples / batch_size) + 1
                w_local = [p.detach().clone() for p in local.parameters()]
                upd = [(g.detach() - w) / (lr * steps) for g, w in zip(model.parameters(), w_local)]
                ups.append(w_local)
            if mode == 'fed_sgd':
                ups.append(upd)
            sizes.append(n_samples)
            if r % record_every == 0:
                for k, u in zip(rec, upd):
                    rec[k].append(u.numpy().astype(np.float16))
                rec_gender.append(GENDER[c['gender']]); rec_round.append(r); rec_client.append(c['name'])
        # server aggregation (weighted by number of samples)
        wts = np.array(sizes) / np.sum(sizes)
        agg = [sum(float(wts[i]) * ups[i][j] for i in range(len(ups))) for j in range(len(ups[0]))]
        with torch.no_grad():
            for p, a in zip(model.parameters(), agg):
                if mode == 'fed_sgd': p -= lr * a
                else: p.copy_(a)
        # validation over all clients' local validation sets (sample-weighted), test on held-out speakers
        accs, uars, ns = [], [], []
        for c in clients:
            if len(c['val_y']) == 0: continue
            a, u = _evaluate(model, c['val_x'], c['val_y'])
            accs.append(a); uars.append(u); ns.append(len(c['val_y']))
        ns = np.array(ns) / np.sum(ns)
        val_uar = float(np.dot(uars, ns))
        if r > rounds // 2 and val_uar > best['val_uar']:
            ta, tu = _evaluate(model, test['x'], test['y'])
            best = dict(val_uar=val_uar, test_acc=ta, test_uar=tu, round=r)
        if r % 50 == 0 or r == rounds - 1:
            ta, tu = _evaluate(model, test['x'], test['y'])
            log(f'  round {r:3d}  val_uar {val_uar:.3f}  test_acc {ta:.3f}  test_uar {tu:.3f}')
    updates = {k: np.stack(v) for k, v in rec.items()}
    updates.update(gender=np.array(rec_gender), round=np.array(rec_round), client=np.array(rec_client))
    return best, updates
