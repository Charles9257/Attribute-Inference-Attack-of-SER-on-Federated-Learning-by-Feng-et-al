"""Attribute (gender) inference attack on shared SER model updates, following
train/federated_attribute_attack.py and train/federated_attribute_attack_result.py.

* Shadow training data: updates recorded while the attacker trains 5 shadow FL models on
  public data D_s with the same hyper-parameters as the private training.
* Every element of an update is z-normalised with mean/std computed over all shadow updates.
* 80% / 20% train / validation split; Adam(lr=1e-4, wd=1e-4), ReduceLROnPlateau, batch 20;
  the checkpoint with the best validation UAR is kept.
* Test: every update shared by the clients of the private FL training on D_p.
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from models import AttackModel

LAYERS = {'first': ('w0', 'b0'), 'second': ('w1', 'b1'), 'last': ('w2', 'b2')}


def norm_stats(updates):
    st = {}
    for k in ['w0', 'b0', 'w1', 'b1', 'w2', 'b2']:
        a = updates[k].astype(np.float32)
        st[k] = (a.mean(0), a.std(0) + 1e-5)
    return st


def _tensor(updates, idx, k, st):
    return torch.from_numpy((updates[k][idx].astype(np.float32) - st[k][0]) / st[k][1])


def predict(model, updates, layer, st, bs=50):
    wk, bk = LAYERS[layer]
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(updates['gender']), bs):
            idx = np.arange(i, min(i + bs, len(updates['gender'])))
            out.append(model(_tensor(updates, idx, wk, st), _tensor(updates, idx, bk, st)).argmax(1).numpy())
    return np.concatenate(out)


def scores(y, p):
    return dict(acc=accuracy_score(y, p), uar=recall_score(y, p, average='macro'))


def train_attack(shadow, layer, st, epochs, lr=1e-4, bs=20, seed=8, log=print):
    torch.manual_seed(seed)
    wk, bk = LAYERS[layer]
    tr, va = train_test_split(np.arange(len(shadow['gender'])), test_size=0.2, random_state=0)
    model = AttackModel(layer, shadow[wk].shape[1:], shadow[bk].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.2, min_lr=1e-6)
    crit = nn.NLLLoss()
    y_all = torch.from_numpy(shadow['gender'])
    best, best_state, hist = None, None, []
    rng = np.random.RandomState(seed)
    va_sub = {k: shadow[k][va] for k in [wk, bk, 'gender']}
    for ep in range(epochs):
        t0 = time.time(); model.train(); losses = []
        order = rng.permutation(tr)
        for i in range(0, len(order), bs):
            idx = np.sort(order[i:i + bs])
            loss = crit(model(_tensor(shadow, idx, wk, st), _tensor(shadow, idx, bk, st)), y_all[idx])
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        # validation
        model.eval()
        with torch.no_grad():
            vl, vp = [], []
            for i in range(0, len(va), 50):
                idx = np.arange(i, min(i + 50, len(va)))
                o = model(_tensor(va_sub, idx, wk, st), _tensor(va_sub, idx, bk, st))
                vl.append(crit(o, torch.from_numpy(va_sub['gender'][idx])).item() * len(idx)); vp.append(o.argmax(1).numpy())
        vloss = sum(vl) / len(va); s = scores(va_sub['gender'], np.concatenate(vp))
        sched.step(vloss)
        hist.append(dict(epoch=ep, train_loss=float(np.mean(losses)), val_loss=vloss, **s))
        log(f'    [{layer}] epoch {ep:2d} train_loss {np.mean(losses):.3f} val_loss {vloss:.3f} '
            f'val_acc {s["acc"]:.3f} val_uar {s["uar"]:.3f} ({time.time() - t0:.0f}s)')
        if best is None or s['uar'] > best['uar']:
            best, best_state = dict(epoch=ep, **s), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best, hist
