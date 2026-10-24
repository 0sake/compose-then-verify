"""Proposta 5, tetto oracolo: penalita' = popolarita' VERA (N_10 z-scored).

Deliberatamente circolare (N_10 viene dalle stesse query del mirror): limita
dall'alto qualunque correzione di popolarita' bank-based. Se il massimo su
gamma e' ~= 0, la proposta e' morta indipendentemente dallo stimatore.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.paths import CELEBA, EVAL_JSON, VETO_JSON, CHECKPOINTS, load_features, results_dir
P2 = CHECKPOINTS
OUT = results_dir("p5_hubness")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

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
V_VAL = E[SPLIT_IDX["val"]].to(DEV)


def parse_query(q):
    out = []
    for tok in q.split(","):
        tok = tok.strip()
        out.append((1 if tok[0] == "+" else -1, NAME2COL[tok[1:].strip()]))
    return out


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
veto_ref = {row["query"]: row["n_src"] for row in json.load(open(VETO_JSON))["base"]}
assert all(len(e["ground_truth"]) == veto_ref[e["query"]] for e in bench_val)

ck = torch.load(P2 / "fusion_text_s3000_lr0.0003.pt", map_location=DEV,
                weights_only=False)
E_pos, E_neg = ck["state"]["E_pos"], ck["state"]["E_neg"]

N10 = np.load(OUT / "N10_counts.npy").astype(np.float64)
P = torch.tensor((N10 - N10.mean()) / N10.std(), dtype=torch.float32,
                 device=DEV)

KS = (1, 5, 10)


def eval_entry(scores, gt_lists):
    topk = scores.topk(max(KS), dim=1).indices.cpu().numpy()
    out = {}
    for K in KS:
        r = 0.0
        for i, gt in enumerate(gt_lists):
            r += len(set(topk[i, :K].tolist()) & gt) > 0
        out[f"R@{K}"] = r / len(gt_lists)
    return out


def evaluate_pen(gamma, batch=2048):
    table = []
    for entry in bench_val:
        conds = parse_query(entry["query"])
        delta = torch.stack([E_pos[c] if s > 0 else E_neg[c]
                             for s, c in conds]).sum(0)
        src_idx = np.array([int(kk) for kk in entry["ground_truth"]])
        gt_lists = [set(v) for v in entry["ground_truth"].values()]
        agg = {f"R@{K}": 0.0 for K in KS}
        for lo in range(0, len(src_idx), batch):
            sl = slice(lo, min(lo + batch, len(src_idx)))
            idx = torch.from_numpy(src_idx[sl]).to(DEV)
            q = F.normalize(V_VAL[idx] + delta, dim=-1)
            scores = q @ V_VAL.T - gamma * P[None, :]
            scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
            part = eval_entry(scores, gt_lists[sl])
            w = (sl.stop - sl.start) / len(src_idx)
            for kk in agg:
                agg[kk] += part[kk] * w
        table.append({"query": entry["query"], **agg})
    return table


def macro(table, key):
    return float(np.mean([row[key] for row in table]))


GAMMAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)
res = {}
print(f"{'gamma':>7} {'macro R@10':>11}")
for g in GAMMAS:
    t = evaluate_pen(g)
    res[g] = macro(t, "R@10")
    print(f"{g:>7.3f} {res[g]:>11.3f}", flush=True)
g_best = max(res, key=res.get)
print(f"\ntetto oracolo popolarita': gamma*={g_best:g}, "
      f"R@10 {res[g_best]:.3f} (base {res[0.0]:.3f}, "
      f"delta {res[g_best] - res[0.0]:+.3f})")
json.dump({f"{g:g}": v for g, v in res.items()},
          open(OUT / "oracle_popularity_headroom.json", "w"), indent=1)
