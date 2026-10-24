"""Proposta 5, diagnosi anticipata: hubness della galleria sotto le query composte.

Tesi: le query composte sono OOD rispetto alla galleria e il ranking e'
dominato dagli hub. Diagnosi (due numeri, decisione binaria):
  1. skewness della distribuzione delle k-occurrence N_10(x) sulla galleria
     di validation sotto TUTTE le (query, sorgente) del mirror con
     DirectionFusion (text-init, lr 3e-4, mirror 0.296);
  2. overlap hub/GT: per gli hub (top-1% e top-0.1% per N_10), frazione delle
     loro apparizioni nei top-10 che sono hit (x nel GT della sorgente),
     confrontata con la frazione hit di tutti gli slot top-10.
Regola dichiarata nella proposta: skewness alta E overlap basso -> QB-Norm ha
qualcosa da rimuovere; hub legittimi (overlap ~= frazione globale) -> lasciar
perdere. Zero training, solo mirror.
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
torch.manual_seed(0)

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
N_GAL = V_VAL.shape[0]


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

# --- DirectionFusion (text-init, lr 3e-4) ----------------------------------
ck = torch.load(P2 / "fusion_text_s3000_lr0.0003.pt", map_location=DEV,
                weights_only=False)
E_pos, E_neg = ck["state"]["E_pos"], ck["state"]["E_neg"]
print(f"DirectionFusion mirror R@10 {ck['mirror_R10']:.3f}", flush=True)

# --- top-10 di tutte le (query, sorgente); N_10 e hit per immagine ---------
N10 = np.zeros(N_GAL, dtype=np.int64)      # apparizioni nei top-10
HIT = np.zeros(N_GAL, dtype=np.int64)      # apparizioni che sono hit
n_slots = 0
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
                N10[x] += 1
                HIT[x] += x in gt
            n_slots += 10

n_src_tot = n_slots // 10
print(f"(query, sorgente) totali: {n_src_tot}, slot top-10: {n_slots}",
      flush=True)

# --- numeri della decisione -------------------------------------------------
mu, sd = N10.mean(), N10.std()
skew = float(((N10 - mu) ** 3).mean() / sd ** 3)
occupied = int((N10 > 0).sum())
hit_rate_all = HIT.sum() / n_slots

print(f"\ndistribuzione N_10 sulla galleria ({N_GAL} immagini):")
print(f"  media {mu:.3f}  std {sd:.3f}  max {N10.max()}  "
      f"immagini con N_10>0: {occupied} ({100 * occupied / N_GAL:.1f}%)")
print(f"  skewness = {skew:.2f}")

report = {"skewness": skew, "mean": float(mu), "std": float(sd),
          "max": int(N10.max()), "occupied_frac": occupied / N_GAL,
          "hit_rate_all_slots": float(hit_rate_all), "hubs": {}}
print(f"\n  frazione hit su TUTTI gli slot top-10: {hit_rate_all:.3f}")
for frac in (0.01, 0.001):
    n_hub = max(1, int(N_GAL * frac))
    hub_idx = np.argsort(-N10)[:n_hub]
    slots_hub = int(N10[hub_idx].sum())
    hits_hub = int(HIT[hub_idx].sum())
    share = slots_hub / n_slots
    hr = hits_hub / max(slots_hub, 1)
    print(f"  hub top-{100 * frac:g}% ({n_hub} img): occupano "
          f"{100 * share:.1f}% degli slot, frazione hit {hr:.3f} "
          f"(vs {hit_rate_all:.3f} globale)")
    report["hubs"][f"top{frac:g}"] = {
        "n_images": n_hub, "slot_share": share, "hit_rate": hr,
        "N10_min": int(N10[hub_idx].min()), "N10_max": int(N10[hub_idx].max())}

# top-20 hub singoli, per ispezione nel report
top20 = np.argsort(-N10)[:20]
val_files = [filenames[i] for i in SPLIT_IDX["val"]]
print(f"\n  top-20 hub (val_idx, file, N_10, hit_rate):")
for i in top20:
    print(f"    {i:>6} {val_files[i]} N10={N10[i]:>5} "
          f"hit={HIT[i] / max(N10[i], 1):.3f}")
report["top20"] = [{"val_idx": int(i), "file": val_files[i],
                    "N10": int(N10[i]),
                    "hit_rate": float(HIT[i] / max(N10[i], 1))}
                   for i in top20]

OUT.mkdir(exist_ok=True)
json.dump(report, open(OUT / "hubness_diagnostic_mirror.json", "w"), indent=1)
np.save(OUT / "N10_counts.npy", N10)
print("\nsalvati hubness_diagnostic_mirror.json e N10_counts.npy", flush=True)
