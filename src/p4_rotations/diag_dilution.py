"""Proposta 4, diagnosi anticipata (gratis): diluizione multi-condizione.

Tesi: con q = normalize(v_ref + somma_c delta_c) lo spostamento effettivo per
condizione decade ~1/sqrt(k). Misura sul mirror, per B3 (D_text, alpha/beta
ottimi) e DirectionFusion (text-init, lr 3e-4, mirror 0.296):

  per ogni (query, condizione c): rapporto di diluizione
      r_c = mean_src[(q_full_hat - v_hat) . d_hat_c] /
            mean_src[(q_alone_hat - v_hat) . d_hat_c]
  con d_hat_c = versore del delta applicato (E_pos/E_neg o +-alpha/beta d_text)
  e q_alone = stessa condizione applicata da sola alle stesse sorgenti.

Controlla l'identita' dell'attributo (ogni condizione confrontata con se
stessa), quindi il decadimento in k non e' confuso dal mix di attributi.
Predizione della tesi: r(k=2) ~ 0.707, r(k=3) ~ 0.577. Se r non decade,
l'idea e' morta. Zero training, solo mirror.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.paths import CELEBA, EVAL_JSON, VETO_JSON, CHECKPOINTS, load_features, results_dir
P2 = CHECKPOINTS
OUT = results_dir("p4_rotations")
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

# --- delta per (attributo, segno) dei due metodi ---------------------------
DIRS = torch.load(P2 / "directions_all_sets.pt", map_location=DEV,
                  weights_only=False)
B3RES = json.load(open(results_dir("p2_paired_directions") / "b3_direction_ablation_mirror.json"))
a_t, b_t = B3RES["text"]["alpha"], B3RES["text"]["beta"]
D_text = DIRS["D_text"].to(DEV)

ck = torch.load(P2 / "fusion_text_s3000_lr0.0003.pt", map_location=DEV,
                weights_only=False)
E_pos, E_neg = ck["state"]["E_pos"], ck["state"]["E_neg"]
print(f"B3: (alpha, beta) = ({a_t}, {b_t}); "
      f"DirectionFusion mirror R@10 {ck['mirror_R10']:.3f}", flush=True)

METHODS = {
    "B3": lambda s, c: (a_t * D_text[c]) if s > 0 else (-b_t * D_text[c]),
    "DirectionFusion": lambda s, c: E_pos[c] if s > 0 else E_neg[c],
}

# --- diagnosi --------------------------------------------------------------
rows_out = []
for name, delta_fn in METHODS.items():
    per_k = defaultdict(list)
    for e in bench_val:
        conds = parse_query(e["query"])
        k = len(conds)
        src = torch.tensor([int(i) for i in e["ground_truth"]], device=DEV)
        v = V_VAL[src]                                   # gia' normalizzati
        delta_full = torch.stack([delta_fn(s, c) for s, c in conds]).sum(0)
        q_full = F.normalize(v + delta_full, dim=-1)
        ratios, moves_alone, moves_full = [], [], []
        for s, c in conds:
            dvec = delta_fn(s, c)
            dhat = F.normalize(dvec, dim=-1)
            q_alone = F.normalize(v + dvec, dim=-1)
            m_alone = ((q_alone - v) @ dhat).mean().item()
            m_full = ((q_full - v) @ dhat).mean().item()
            ratios.append(m_full / m_alone)
            moves_alone.append(m_alone)
            moves_full.append(m_full)
        ang = torch.rad2deg(torch.arccos(
            (q_full * v).sum(-1).clamp(-1, 1))).mean().item()
        row = {"method": name, "query": e["query"], "k": k,
               "ratio": float(np.mean(ratios)),
               "move_alone": float(np.mean(moves_alone)),
               "move_full": float(np.mean(moves_full)),
               "angle_deg": ang}
        rows_out.append(row)
        per_k[k].append(row["ratio"])

    print(f"\n[{name}] rapporto di diluizione per query "
          f"(move_full/move_alone, media sulle condizioni):")
    print(f"{'query':<46} {'k':>2} {'ratio':>7} {'m_alone':>8} "
          f"{'m_full':>8} {'ang':>6}")
    for r in [r for r in rows_out if r["method"] == name]:
        print(f"{r['query']:<46} {r['k']:>2} {r['ratio']:>7.3f} "
              f"{r['move_alone']:>8.4f} {r['move_full']:>8.4f} "
              f"{r['angle_deg']:>6.2f}")
    print(f"  media per k: " + "  ".join(
        f"k={k}: {np.mean(per_k[k]):.3f} (n={len(per_k[k])})"
        for k in sorted(per_k)))
    print(f"  predizione 1/sqrt(k):  k=1: 1.000  k=2: {1/np.sqrt(2):.3f}  "
          f"k=3: {1/np.sqrt(3):.3f}")

OUT.mkdir(exist_ok=True)
json.dump(rows_out, open(OUT / "dilution_diagnostic_mirror.json", "w"),
          indent=1)
print("\nsalvato dilution_diagnostic_mirror.json", flush=True)
