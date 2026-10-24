"""Ablazioni del re-rank a palla residua (segue restball_rerank.py).

Domande:
  A. Da dove viene il +0.129? score = con + rest: ablazione con_only / rest_only.
     Predizione: con_only ~ base (e' la congiuntiva soft gia' falsificata, tetto
     +0.061); rest_only cattura quasi tutto il guadagno.
  B. M ottimale: coverage e dp_probe per M in {100, 200, 500, 1000}.
     Predizione: coverage sale con M ma il rumore di verifica pure; picco
     tra 200 e 500.
  C. Profilo sorgente indurito (bit a +-1 da soglia 0.5) vs morbido.
     Predizione: peggio del morbido (perde lo sconto sull'incertezza).
Tutto mirror, sonde e base identici al run precedente (stesso seed).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.paths import CELEBA, EVAL_JSON, CHECKPOINTS, load_features, results_dir
P2 = CHECKPOINTS
OUT = results_dir("predicate_rerank")
DEV = "cuda"
torch.manual_seed(0)

lines = (CELEBA / "list_attr_celeba.txt").read_text().splitlines()
ATTR_NAMES = lines[1].split()
NAME2COL = {n: i for i, n in enumerate(ATTR_NAMES)}
filenames, rows = [], []
for ln in lines[2:]:
    parts = ln.split()
    if parts:
        filenames.append(parts[0])
        rows.append([int(x) for x in parts[1:]])
ATTRS = np.array(rows, dtype=np.int8)
partition = {}
for ln in (CELEBA / "list_eval_partition.txt").read_text().splitlines():
    parts = ln.split()
    if parts:
        partition[parts[0]] = int(parts[1])
PART = np.array([partition[f] for f in filenames])
SPLIT_IDX = {s: np.nonzero(PART == c)[0] for s, c in
             (("train", 0), ("val", 1), ("test", 2))}
A_SPLIT = {s: ATTRS[SPLIT_IDX[s]] for s in SPLIT_IDX}

d = load_features()
fn2idx = {f: i for i, f in enumerate(d["filenames"])}
order = [fn2idx[f] for f in filenames]
E = F.normalize(d["embeddings"][order], dim=-1)
E_TRAIN = E[SPLIT_IDX["train"]].to(DEV)
V_VAL = E[SPLIT_IDX["val"]].to(DEV)
A_TRAIN = torch.from_numpy(A_SPLIT["train"].astype(np.float32)).to(DEV)
A_VAL = torch.from_numpy(A_SPLIT["val"].astype(np.float32)).to(DEV)


def parse_query(q):
    return [(1 if t.strip()[0] == "+" else -1, NAME2COL[t.strip()[1:].strip()])
            for t in q.split(",")]


def rebuild_ground_truth(query, A):
    conds = parse_query(query)
    q_cols = [c for _, c in conds]
    rest = np.array([c for c in range(40) if c not in q_cols])
    ok = np.ones(len(A), dtype=bool)
    viol = np.ones(len(A), dtype=bool)
    for sign, col in conds:
        ok &= A[:, col] == sign
        viol &= A[:, col] != sign
    cand = np.nonzero(ok)[0]
    A_cand = A[np.ix_(cand, rest)].astype(np.float32)
    A_rest = A[:, rest].astype(np.float32)
    targets = {}
    for lo in range(0, len(A), 4000):
        hi = min(lo + 4000, len(A))
        ham = (len(rest) - A_cand @ A_rest[lo:hi].T) / 2
        for j in range(lo, hi):
            t = cand[ham[:, j - lo] <= 2]
            t = t[t != j]
            if len(t) >= 5:
                targets[j] = t.tolist()
    return targets, viol


with open(EVAL_JSON) as f:
    BENCH = json.load(f)
bench_val, seen = [], set()
for entry in BENCH:
    if entry["query"] in seen:
        continue
    seen.add(entry["query"])
    targets, viol = rebuild_ground_truth(entry["query"], A_SPLIT["val"])
    gt = {str(j): t for j, t in targets.items() if viol[j]}
    bench_val.append({"query": entry["query"], "ground_truth": gt})

ck = torch.load(P2 / "fusion_text_s3000_lr0.0003.pt", map_location=DEV,
                weights_only=False)
E_pos, E_neg = ck["state"]["E_pos"], ck["state"]["E_neg"]


def qfn(v_ref, conds):
    delta = torch.stack([E_pos[c] if s > 0 else E_neg[c]
                         for s, c in conds]).sum(0)
    return F.normalize(v_ref + delta, dim=-1)


def train_logreg(c, steps=400, lr=0.05, wd=1e-4):
    y = (A_TRAIN[:, c] > 0).float()
    w = torch.zeros(512, device=DEV, requires_grad=True)
    b = torch.zeros((), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        z = E_TRAIN @ w + b
        loss = F.binary_cross_entropy_with_logits(z, y) + wd * (w * w).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


t0 = time.time()
W = torch.zeros(40, 512, device=DEV)
B = torch.zeros(40, device=DEV)
for c in range(40):
    W[c], B[c] = train_logreg(c)
P_VAL = torch.sigmoid(V_VAL @ W.T + B).clamp(1e-5, 1 - 1e-5)
print(f"sonde pronte ({time.time() - t0:.0f}s)", flush=True)

KS = (1, 5, 10)
MS = (100, 200, 500, 1000)


def pb_tail_le2(agree):
    B_, M_, R_ = agree.shape
    st = torch.zeros(B_, M_, 3, device=agree.device)
    st[:, :, 0] = 1.0
    for i in range(R_):
        a = agree[:, :, i]
        m = 1.0 - a
        st = torch.stack([st[:, :, 0] * a,
                          st[:, :, 1] * a + st[:, :, 0] * m,
                          st[:, :, 2] * a + st[:, :, 1] * m], dim=-1)
    return torch.log(st.sum(-1).clamp_min(1e-30))


def eval_entry(scores, gt_lists):
    topk = scores.topk(max(KS), dim=1).indices.cpu().numpy()
    out = {}
    for K in KS:
        r = p = 0.0
        for i, gt in enumerate(gt_lists):
            hits = len(set(topk[i, :K].tolist()) & gt)
            r += hits > 0
            p += hits / K
        out[f"R@{K}"] = r / len(gt_lists)
        out[f"P@{K}"] = p / len(gt_lists)
    return out


VAR = ["cov", "full", "rest_only", "con_only", "refhard"]
res = {v: {M: [] for M in MS} for v in VAR}
for entry in bench_val:
    conds = parse_query(entry["query"])
    ccols = [c for _, c in conds]
    csigns = torch.tensor([s for s, _ in conds], device=DEV, dtype=torch.float32)
    rest = [c for c in range(40) if c not in ccols]
    src_idx = np.array([int(k) for k in entry["ground_truth"]])
    gt_lists = [set(v_) for v_ in entry["ground_truth"].values()]
    agg = {v: {M: {f"{m}@{K}": 0.0 for m in "RP" for K in KS} for M in MS}
           for v in VAR}
    for lo in range(0, len(src_idx), 1024):
        sl = slice(lo, min(lo + 1024, len(src_idx)))
        idx = torch.from_numpy(src_idx[sl]).to(DEV)
        nb = len(idx)
        q = qfn(V_VAL[idx], conds)
        base = q @ V_VAL.T
        base[torch.arange(nb, device=DEV), idx] = -torch.inf
        w = (sl.stop - sl.start) / len(src_idx)
        topall = base.topk(max(MS), dim=1).indices
        for M in MS:
            top = topall[:, :M]
            p_cand = P_VAL[top]
            a_cand = A_VAL[top]
            pc = p_cand[:, :, ccols]
            target_pos = (csigns > 0).view(1, 1, -1)
            lg_con = torch.log(torch.where(target_pos, pc, 1 - pc)).sum(-1)
            p_r = p_cand[:, :, rest]
            pref = P_VAL[idx][:, None, rest]
            pref_h = (pref > 0.5).float()
            lg_rest = pb_tail_le2(p_r * pref + (1 - p_r) * (1 - pref))
            lg_rest_h = pb_tail_le2(p_r * pref_h + (1 - p_r) * (1 - pref_h))
            rel = torch.zeros(nb, M, device=DEV)
            for i, gt in enumerate(gt_lists[sl]):
                gt_t = torch.tensor(list(gt), device=DEV)
                rel[i] = torch.isin(top[i], gt_t).float()
            scores = {"cov": rel, "full": lg_con + lg_rest,
                      "rest_only": lg_rest, "con_only": lg_con,
                      "refhard": lg_con + lg_rest_h}
            for v, sc in scores.items():
                fullsc = torch.full_like(base, -torch.inf)
                fullsc.scatter_(1, top, sc + 1e-4 * base.gather(1, top))
                part = eval_entry(fullsc, gt_lists[sl])
                for k in agg[v][M]:
                    agg[v][M][k] += part[k] * w
    for v in VAR:
        for M in MS:
            res[v][M].append({"query": entry["query"], **agg[v][M]})
    print(f"{entry['query']:44s} " + " | ".join(
        f"M{M}: cov {agg['cov'][M]['R@10']:.3f} full {agg['full'][M]['R@10']:.3f}"
        for M in (200, 500)), flush=True)


def macro(t, k="R@10"):
    return float(np.mean([r[k] for r in t]))


print("\n=== MACRO R@10 mirror ===")
print(f"{'variante':12s}" + "".join(f"  M={M:<5d}" for M in MS))
for v in VAR:
    print(f"{v:12s}" + "".join(f"  {macro(res[v][M]):.3f} " for M in MS))
with open(OUT / "restball_ablation_mirror.json", "w") as f:
    json.dump({v: {str(M): res[v][M] for M in MS} for v in VAR}, f, indent=1)
print(f"salvato ({time.time() - t0:.0f}s totali)")
