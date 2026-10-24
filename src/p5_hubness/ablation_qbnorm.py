"""Proposta 5, ablazione minima: DirectionFusion congelato +- QB-Norm sul mirror.

Correzione lato galleria (ammissibile: evaluate_retrieval prende indici):
    s'(q, x) = cos(q, x) - gamma * mean_{q' in Q}[cos(q', x)]
Q = banca di query sintetiche generate con la stessa regola del training
(k in {1,2,3}, condizioni casuali con >=1 sorgente violatrice e >=5
satisfier nel train, sorgenti dal train), composte con i delta del
DirectionFusion congelato (text-init, lr 3e-4, mirror 0.296). Sweep di gamma
su 5 valori, selezione sul mirror. Il test non viene toccato.
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
SEED = 0
torch.manual_seed(SEED)

# --- dati (identico alla pipeline) -----------------------------------------
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
assert {p: int((PART == p).sum()) for p in (0, 1, 2)} == \
    {0: 162770, 1: 19867, 2: 19962}, "split non canonico"
SPLIT_IDX = {s: np.nonzero(PART == c)[0] for s, c in
             (("train", 0), ("val", 1), ("test", 2))}
A_SPLIT = {s: ATTRS[SPLIT_IDX[s]] for s in SPLIT_IDX}

d = load_features()
fn2idx = {f: i for i, f in enumerate(d["filenames"])}
order = [fn2idx[f] for f in filenames]
E = F.normalize(d["embeddings"][order], dim=-1)
V_VAL = E[SPLIT_IDX["val"]].to(DEV)
V_TRAIN = E[SPLIT_IDX["train"]].to(DEV)
A_TRAIN = torch.from_numpy(A_SPLIT["train"].astype(np.float32)).to(DEV)


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
print(f"mirror rigenerato e verificato: {len(bench_val)} query", flush=True)

# --- DirectionFusion congelato ---------------------------------------------
ck = torch.load(P2 / "fusion_text_s3000_lr0.0003.pt", map_location=DEV,
                weights_only=False)
E_pos, E_neg = ck["state"]["E_pos"], ck["state"]["E_neg"]
print(f"DirectionFusion mirror R@10 {ck['mirror_R10']:.3f}", flush=True)

# --- banca di query sintetiche (generatore del training, sorgenti train) ---
GEN = torch.Generator().manual_seed(SEED)
BANK_SIZE = 2000
bank = []
while len(bank) < BANK_SIZE:
    k = int(torch.randint(1, 4, (1,), generator=GEN))
    cols = torch.randperm(40, generator=GEN)[:k]
    signs = torch.randint(0, 2, (k,), generator=GEN) * 2.0 - 1.0
    cols_d, signs_d = cols.to(DEV), signs.to(DEV)
    sat = (A_TRAIN[:, cols_d] == signs_d).all(1)
    viol = (A_TRAIN[:, cols_d] == -signs_d).all(1)
    if int(sat.sum()) < 5 or int(viol.sum()) < 1:
        continue
    viol_idx = torch.nonzero(viol, as_tuple=True)[0]
    src = viol_idx[int(torch.randint(0, len(viol_idx), (1,), generator=GEN))]
    delta = torch.stack([E_pos[c] if s > 0 else E_neg[c]
                         for c, s in zip(cols.tolist(), signs.tolist())]).sum(0)
    bank.append(F.normalize(V_TRAIN[src] + delta, dim=-1))
Q_BANK = torch.stack(bank)
ACT = (Q_BANK @ V_VAL.T).mean(0)          # attivazione media per immagine
print(f"banca: {BANK_SIZE} query sintetiche; attivazione galleria "
      f"mean {ACT.mean():.4f} std {ACT.std():.4f} "
      f"min {ACT.min():.4f} max {ACT.max():.4f}", flush=True)

# --- valutazione con s' = cos - gamma * ACT --------------------------------
KS = (1, 5, 10)


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


def evaluate_qb(gamma, batch=2048):
    table = []
    for entry in bench_val:
        conds = parse_query(entry["query"])
        delta = torch.stack([E_pos[c] if s > 0 else E_neg[c]
                             for s, c in conds]).sum(0)
        src_idx = np.array([int(kk) for kk in entry["ground_truth"]])
        gt_lists = [set(v) for v in entry["ground_truth"].values()]
        agg = {f"{m}@{K}": 0.0 for m in "RP" for K in KS}
        for lo in range(0, len(src_idx), batch):
            sl = slice(lo, min(lo + batch, len(src_idx)))
            idx = torch.from_numpy(src_idx[sl]).to(DEV)
            q = F.normalize(V_VAL[idx] + delta, dim=-1)
            scores = q @ V_VAL.T - gamma * ACT[None, :]
            scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
            part = eval_entry(scores, gt_lists[sl])
            w = (sl.stop - sl.start) / len(src_idx)
            for kk in agg:
                agg[kk] += part[kk] * w
        table.append({"query": entry["query"], "n_src": len(src_idx), **agg})
    return table


def macro(table, key):
    return float(np.mean([row[key] for row in table]))


GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
tables = {}
print(f"\n{'gamma':>6} {'macro R@10':>11} {'R@5':>7} {'R@1':>7}")
for g in GAMMAS:
    t = evaluate_qb(g)
    tables[g] = t
    print(f"{g:>6.2f} {macro(t, 'R@10'):>11.3f} {macro(t, 'R@5'):>7.3f} "
          f"{macro(t, 'R@1'):>7.3f}", flush=True)

g_best = max(GAMMAS, key=lambda g: macro(tables[g], "R@10"))
base_t, best_t = tables[0.0], tables[g_best]
print(f"\ngamma* = {g_best} (selezionato sul mirror)")

print(f"\n{'query':<46} {'n_src':>6} {'base':>7} {'qbnorm':>7} {'delta':>7}")
for i, e in enumerate(bench_val):
    b, q = base_t[i]["R@10"], best_t[i]["R@10"]
    mark = "  <- negativa" if "-" in e["query"] else ""
    print(f"{e['query']:<46} {e['n_src'] if 'n_src' in e else len(e['ground_truth']):>6} "
          f"{b:>7.3f} {q:>7.3f} {q - b:>+7.3f}{mark}")
print(f"{'MACRO':<46} {'':>6} {macro(base_t, 'R@10'):>7.3f} "
      f"{macro(best_t, 'R@10'):>7.3f} "
      f"{macro(best_t, 'R@10') - macro(base_t, 'R@10'):>+7.3f}")

neg_q = [i for i, e in enumerate(bench_val) if "-" in e["query"]]
negdom = [i for i, e in enumerate(bench_val)
          if all(s < 0 for s, _ in parse_query(e["query"]))]
per_k = {}
for i, e in enumerate(bench_val):
    per_k.setdefault(len(parse_query(e["query"])), []).append(i)
for label, idxs in (("solo query con negazioni", neg_q),
                    ("solo negative-dominanti", negdom),
                    *[(f"k={k}", v) for k, v in sorted(per_k.items())]):
    b = np.mean([base_t[i]["R@10"] for i in idxs])
    q = np.mean([best_t[i]["R@10"] for i in idxs])
    print(f"{label:<53} {b:>7.3f} {q:>7.3f} {q - b:>+7.3f}")

json.dump({"gammas": {f"{g:g}": macro(tables[g], "R@10") for g in GAMMAS},
           "gamma_best": g_best,
           "base_table": base_t, "qbnorm_table": best_t,
           "bank_size": BANK_SIZE},
          open(OUT / "qbnorm_ablation_mirror.json", "w"), indent=1)
torch.save({"Q_bank": Q_BANK.cpu(), "ACT": ACT.cpu()}, OUT / "qbnorm_bank.pt")
print("\nsalvati qbnorm_ablation_mirror.json e qbnorm_bank.pt", flush=True)
