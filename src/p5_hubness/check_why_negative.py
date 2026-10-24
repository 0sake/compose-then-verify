"""Proposta 5, autopsia del risultato negativo (solo mirror).

Due domande:
  1. La banca stima la popolarita'? Spearman(ACT, N_10) sulla galleria.
  2. La popolarita' e' informativa? hit rate degli slot top-10 per decile
     di N_10 (GT densa -> se cresce col decile, sottrarla DEVE far male).
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

N10 = np.load(OUT / "N10_counts.npy")
ACT = torch.load(OUT / "qbnorm_bank.pt", map_location="cpu",
                 weights_only=False)["ACT"].numpy()


def spearman(x, y):
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


print(f"1) Spearman(ACT_banca, N_10) = {spearman(ACT, N10.astype(float)):.3f}")

# hit per slot, come nella diagnosi
HIT = np.zeros(len(N10), dtype=np.int64)
N10b = np.zeros(len(N10), dtype=np.int64)
for e in bench_val:
    conds = parse_query(e["query"])
    delta = torch.stack([E_pos[c] if s > 0 else E_neg[c]
                         for s, c in conds]).sum(0)
    src = torch.tensor([int(i) for i in e["ground_truth"]], device=DEV)
    gt_sets = [set(v) for v in e["ground_truth"].values()]
    for lo in range(0, len(src), 2048):
        sl = slice(lo, min(lo + 2048, len(src)))
        idx = src[sl]
        q = F.normalize(V_VAL[idx] + delta, dim=-1)
        scores = q @ V_VAL.T
        scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
        topk = scores.topk(10, dim=1).indices.cpu().numpy()
        for i, row in enumerate(topk):
            gt = gt_sets[lo + i]
            for x in row:
                N10b[x] += 1
                HIT[x] += x in gt
assert (N10b == N10).all(), "N10 non riproducibile"

print("\n2) hit rate degli slot per decile di N_10 (solo immagini con N_10>0):")
pos = N10 > 0
vals = N10[pos]
edges = np.quantile(vals, np.linspace(0, 1, 11))
print(f"{'decile':>7} {'N10 range':>15} {'n_img':>7} {'slot':>8} {'hit rate':>9}")
for k in range(10):
    lo, hi = edges[k], edges[k + 1]
    m = pos & (N10 >= lo) & (N10 <= hi if k == 9 else N10 < hi)
    if m.sum() == 0:
        continue
    slots, hits = int(N10[m].sum()), int(HIT[m].sum())
    print(f"{k + 1:>7} {f'{int(lo)}-{int(hi)}':>15} {int(m.sum()):>7} "
          f"{slots:>8} {hits / max(slots, 1):>9.3f}")
