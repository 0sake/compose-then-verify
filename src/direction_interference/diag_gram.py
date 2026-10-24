"""Idea 2, diagnosi preliminare dichiarata: Gram delle direzioni appaiate.

Tesi da verificare PRIMA di implementare il solve a margini: l'addizione
sovrappone direzioni che interferiscono — se cos(d_a, d_b) e' alto, l'edit
`+d_a - d_b` si autodistrugge (spiegherebbe lipstick 0.000 e -Male,-Mustache).

Predizioni dichiarate in documents/round2_proposals.md:
  - |cos|(Wearing_Lipstick, Heavy_Makeup) >= 0.6
  - |cos|(Male, Mustache) alto
Morte della tesi: TUTTE le coppie co-occorrenti nel benchmark |cos| <= 0.3
(l'interferenza non spiega nulla; del problema resta solo la diluizione).

Letture:
  1. Gram |cos| delle direzioni appaiate sui 12 attributi benchmark;
  2. coppie co-occorrenti (query con k>=2) con il cross-term FIRMATO
     dell'edit B3-paired (alpha, beta) = (0.75, 0.50): interferenza reale
     = g_i g_j cos(d_i, d_j) coi coefficienti del metodo;
  3. rapporto di cancellazione per query: ||sum g_i d_i|| / sqrt(sum g_i^2)
     (1 = ortogonale, <1 distruttivo, >1 costruttivo);
  4. riferimento: stesse coppie per D_b5 (= direzioni delle sonde medie di
     classe, cioe' la Gram del solve k x k) e D_text; condizionamento
     kappa(W W^T) del solve per le query multi-condizione.

Zero training, zero GPU: solo il file directions_all_sets.pt e le query.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.paths import EVAL_JSON, CHECKPOINTS, results_dir
P2 = CHECKPOINTS
OUT = results_dir("direction_interference")

d = torch.load(P2 / "directions_all_sets.pt", map_location="cpu",
               weights_only=False)
ATTR_NAMES = d["ATTR_NAMES"]
NAME2COL = {n: i for i, n in enumerate(ATTR_NAMES)}
SETS = {name: F.normalize(d[key].float(), dim=-1)
        for name, key in (("paired", "D_pair"), ("b5_sonde", "D_b5"),
                          ("text", "D_text"))}
ALPHA, BETA = 0.75, 0.50                      # B3-paired ottimi dalla griglia

# --- query benchmark e coppie co-occorrenti ---------------------------------
with open(EVAL_JSON) as f:
    BENCH = json.load(f)
queries, seen = [], set()
for e in BENCH:
    if e["query"] not in seen:
        seen.add(e["query"])
        conds = []
        for tok in e["query"].split(","):
            tok = tok.strip()
            conds.append((1 if tok[0] == "+" else -1, NAME2COL[tok[1:].strip()]))
        queries.append((e["query"], conds))
bench_cols = sorted({c for _, conds in queries for _, c in conds})
multi = [(q, conds) for q, conds in queries if len(conds) >= 2]
print(f"query: {len(queries)}, attributi benchmark: {len(bench_cols)}, "
      f"query multi-condizione: {len(multi)}\n")

# --- 1. Gram |cos| delle appaiate sui 12 attributi --------------------------
Dp = SETS["paired"][bench_cols]
G = (Dp @ Dp.T).numpy()
names = [ATTR_NAMES[c] for c in bench_cols]
print("Gram |cos| direzioni appaiate (12 attributi benchmark):")
hdr = "                    " + "".join(f"{n[:7]:>8s}" for n in names)
print(hdr)
for i, n in enumerate(names):
    print(f"  {n:18s}" + "".join(f"{abs(G[i, j]):8.2f}" for j in range(len(names))))

offdiag = [(abs(G[i, j]), names[i], names[j])
           for i in range(len(names)) for j in range(i + 1, len(names))]
offdiag.sort(reverse=True)
print("\ntop-8 coppie piu' allineate (qualunque, non solo co-occorrenti):")
for v, a, b in offdiag[:8]:
    print(f"  |cos|({a}, {b}) = {v:.3f}")

# --- 2-3. coppie co-occorrenti e cancellazione per query --------------------
def gram_of(set_name, ci, cj):
    D = SETS[set_name]
    return float(D[ci] @ D[cj])


print("\npredizioni dichiarate (appaiate):")
lip_mk = abs(gram_of("paired", NAME2COL["Wearing_Lipstick"],
                     NAME2COL["Heavy_Makeup"]))
male_mu = abs(gram_of("paired", NAME2COL["Male"], NAME2COL["Mustache"]))
print(f"  |cos|(Wearing_Lipstick, Heavy_Makeup) = {lip_mk:.3f}   "
      f"[predetto >= 0.6]")
print(f"  |cos|(Male, Mustache)                 = {male_mu:.3f}   "
      f"[predetto alto]")

print(f"\ncoppie co-occorrenti nel benchmark — |cos| per set e cross-term "
      f"firmato dell'edit B3-paired (g = +{ALPHA} / -{BETA}):")
print(f"{'query':<46} {'coppia':<28} {'paired':>7} {'b5':>6} {'text':>6} "
      f"{'g_i*g_j*cos':>11}")
coocc_rows = []
for q, conds in multi:
    for (si, ci), (sj, cj) in combinations(conds, 2):
        gi = ALPHA if si > 0 else -BETA
        gj = ALPHA if sj > 0 else -BETA
        cp = gram_of("paired", ci, cj)
        cross = gi * gj * cp
        row = {"query": q, "pair": (ATTR_NAMES[ci], ATTR_NAMES[cj]),
               "cos_paired": cp, "cos_b5": gram_of("b5_sonde", ci, cj),
               "cos_text": gram_of("text", ci, cj), "signed_cross": cross}
        coocc_rows.append(row)
        pair_s = f"{ATTR_NAMES[ci][:12]}, {ATTR_NAMES[cj][:12]}"
        print(f"{q:<46} {pair_s:<28} {abs(cp):>7.3f} "
              f"{abs(row['cos_b5']):>6.3f} {abs(row['cos_text']):>6.3f} "
              f"{cross:>+11.3f}")

print(f"\nrapporto di cancellazione per query multi-condizione "
      f"(||sum g_i d_i|| / sqrt(sum g_i^2); 1 = ortogonale):")
canc_rows = []
for q, conds in multi:
    out = {"query": q}
    for set_name in SETS:
        D = SETS[set_name]
        e = torch.zeros(512)
        g2 = 0.0
        for s, c in conds:
            g = ALPHA if s > 0 else -BETA
            e += g * D[c]
            g2 += g * g
        out[set_name] = float(e.norm() / np.sqrt(g2))
    canc_rows.append(out)
    print(f"  {q:<46} paired {out['paired']:.3f}  b5 {out['b5_sonde']:.3f}  "
          f"text {out['text']:.3f}")

# --- 4. condizionamento del solve k x k (sonde = direzioni b5) --------------
print(f"\ncondizionamento kappa(W W^T) del solve a margini per query "
      f"(W = direzioni sonda dei k attributi):")
cond_rows = []
for q, conds in multi:
    for set_name in ("b5_sonde", "paired"):
        W = SETS[set_name][[c for _, c in conds]]
        gram = (W @ W.T).numpy()
        k = np.linalg.cond(gram)
        if set_name == "b5_sonde":
            cond_rows.append({"query": q, "kappa_b5": float(k)})
        else:
            cond_rows[-1]["kappa_paired"] = float(k)
    print(f"  {q:<46} b5 {cond_rows[-1]['kappa_b5']:8.2f}   "
          f"paired {cond_rows[-1]['kappa_paired']:8.2f}")

# --- verdetto dichiarato -----------------------------------------------------
coocc_max = max(abs(r["cos_paired"]) for r in coocc_rows)
alive = coocc_max > 0.3
print(f"\nmax |cos| appaiate sulle coppie co-occorrenti: {coocc_max:.3f}   "
      f"[morte della tesi se TUTTE <= 0.3]")
verdict = ("TESI VIVA: interferenza presente sulle coppie co-occorrenti"
           if alive else
           "TESI MORTA: nessuna coppia co-occorrente sopra 0.3 — "
           "l'interferenza non spiega i fallimenti, resta la diluizione")
print(f"VERDETTO DIAGNOSI: {verdict}")

OUT.mkdir(exist_ok=True)
json.dump({"gram_paired_benchmark":
           {f"{names[i]}|{names[j]}": float(G[i, j])
            for i in range(len(names)) for j in range(i + 1, len(names))},
           "predicted": {"lipstick_makeup": lip_mk, "male_mustache": male_mu},
           "cooccurring": [{**r, "pair": list(r["pair"])} for r in coocc_rows],
           "cancellation": canc_rows, "solve_conditioning": cond_rows,
           "coocc_max_abs_cos": coocc_max, "verdict": verdict},
          open(OUT / "gram_diagnostic.json", "w"), indent=1)
print("\nsalvato gram_diagnostic.json")
