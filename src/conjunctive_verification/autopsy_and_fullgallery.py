"""Idea 1, seconda tappa: autopsia della griglia piatta + estensione full-gallery.

La griglia top-M della prima tappa e' piatta (+0.000, 3/294 punti sopra il base
per rumore) nonostante i violatori nel top-10 scendano (25.5% -> 22.7%). Il
tetto oracolo del veto (+0.027) era pero' calcolato sulla GALLERY PIENA, e il
suo guadagno era concentrato su -Male,-Mustache (+0.143).

Parte A — autopsia (decomposizione del collo di bottiglia, solo oracoli):
  A1. R@K del base per K in {10, 50, 100, 200}: copertura del GT nei top-M
      (tetto assoluto di QUALUNQUE re-ranking ristretto ai top-M).
  A2. veto oracolo (label vere, attributi negati) ristretto ai top-200 vs
      gallery piena: quanto del +0.027 sopravvive alla restrizione.
  A3. oracolo congiuntivo (TUTTE le condizioni via label, bonus sui
      soddisfacenti) ristretto ai top-200 vs gallery piena: tetto della
      verifica congiuntiva nelle due modalita'.
  Predizione P_A: l'oracolo in-M recupera una frazione piccola del tetto
  full-gallery -> la restrizione top-M spiega la griglia piatta.

Parte B — estensione dichiarata: stesso scorer a sonde della prima tappa, ma
sulla gallery piena (M = intera validation, 19867):
  S(x) = cos(q_DF, x) + w+ * sum_pos logsig(+sc_c(x)) + w- * sum_neg logsig(-sc_c(x))
E' il "veto coi pesi scelti a griglia invece che con InfoNCE" della proposta,
generalizzato alle positive; l'harness accetta la matrice di score (il veto
lo faceva gia'). Griglia dichiarata: w+, w- in {0, 0.01, 0.02, 0.05, 0.1,
0.2, 0.5, 1.0}, varianti cal / znorm. Ancora: (0,0) = base esatto.
  Predizione P_B: recupera quasi tutto il tetto su -Male,-Mustache (sonda
  Male AUC 0.999) e poco sui casi diffusi (sonde 0.91-0.95).

Regole invariate: kill se tutta la griglia <= base; adozione solo se
delta macro >= +0.010; tutto solo sul mirror; zero training.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.paths import CELEBA, EVAL_JSON, VETO_JSON, CHECKPOINTS, load_features, results_dir
P2 = CHECKPOINTS
OUT = results_dir("conjunctive_verification")
DEV = "cuda"
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
E_TRAIN = E[SPLIT_IDX["train"]].to(DEV)
V_VAL = E[SPLIT_IDX["val"]].to(DEV)
N_GAL = V_VAL.shape[0]
A_TRAIN = torch.from_numpy(A_SPLIT["train"].astype(np.float32)).to(DEV)
A_VAL = torch.from_numpy(A_SPLIT["val"].astype(np.float32)).to(DEV)


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

KS = (1, 5, 10)

# --- DirectionFusion congelato ----------------------------------------------
ck = torch.load(P2 / "fusion_text_s3000_lr0.0003.pt", map_location=DEV,
                weights_only=False)
E_pos, E_neg = ck["state"]["E_pos"], ck["state"]["E_neg"]


def qfn(v_ref, conds):
    delta = torch.stack([E_pos[c] if s > 0 else E_neg[c]
                         for s, c in conds]).sum(0)
    return F.normalize(v_ref + delta, dim=-1)


# --- sonde calibrate (formula identica a killtest_veto.py) ------------------
probes, mids = [], []
for i in range(40):
    pos = A_TRAIN[:, i] > 0
    dd = E_TRAIN[pos].mean(0) - E_TRAIN[~pos].mean(0)
    dd = dd / dd.norm()
    m_pos, m_neg = (E_TRAIN[pos] @ dd).mean(), (E_TRAIN[~pos] @ dd).mean()
    scale = 2.0 / (m_pos - m_neg)
    probes.append(dd * scale)
    mids.append(((m_pos + m_neg) / 2 * scale).item())
P = torch.stack(probes)
mid = torch.tensor(mids, device=DEV)
SC = V_VAL @ P.T - mid
SCZ = (SC - SC.mean(0)) / SC.std(0)

# regola AUC >= 0.75 (prima tappa: tutti i 12 attributi del benchmark passano)
def auc(scores, labels):
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64,
                                device=scores.device)
    pos = labels > 0
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


bench_cols = sorted({c for e in bench_val for _, c in parse_query(e["query"])})
mask = {c: auc(SC[:, c], A_VAL[:, c]) >= 0.75 for c in bench_cols}
assert all(mask.values()), "attributo sotto soglia AUC: rivedere la regola"

# --- strutture per query -----------------------------------------------------
MMAX = 200
BATCH = 2048
t0 = time.time()


def query_structs(entry):
    conds = parse_query(entry["query"])
    src_idx = np.array([int(k) for k in entry["ground_truth"]])
    gt_arrays = [np.array(v) for v in entry["ground_truth"].values()]
    sat = torch.ones(N_GAL, dtype=torch.bool, device=DEV)
    violneg = torch.zeros(N_GAL, dtype=torch.bool, device=DEV)
    for s, c in conds:
        sat &= (A_VAL[:, c] * s) > 0
        if s < 0:
            violneg |= A_VAL[:, c] > 0
    tp = {"cal": torch.zeros(N_GAL, device=DEV),
          "znorm": torch.zeros(N_GAL, device=DEV)}
    tn = {"cal": torch.zeros(N_GAL, device=DEV),
          "znorm": torch.zeros(N_GAL, device=DEV)}
    for tag, sc in (("cal", SC), ("znorm", SCZ)):
        for s, c in conds:
            if s > 0:
                tp[tag] += F.logsigmoid(sc[:, c])
            else:
                tn[tag] += F.logsigmoid(-sc[:, c])
    return conds, src_idx, gt_arrays, sat, violneg, tp, tn


def base_scores(src_slice, conds):
    idx = torch.from_numpy(src_slice).to(DEV)
    q = qfn(V_VAL[idx], conds)
    scores = q @ V_VAL.T
    scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
    return scores


def gt_mask_batch(gt_arrays, lo, hi):
    rows = np.concatenate([np.full(len(gt_arrays[i]), i - lo)
                           for i in range(lo, hi)])
    cols = np.concatenate([gt_arrays[i] for i in range(lo, hi)])
    m = torch.zeros(hi - lo, N_GAL, dtype=torch.bool, device=DEV)
    m[torch.from_numpy(rows).long().to(DEV),
      torch.from_numpy(cols).long().to(DEV)] = True
    return m


# =========================== PARTE A: autopsia ===============================
BIG = 1e4
KS_COV = (10, 50, 100, 200)
rowsA = []
for e in bench_val:
    conds, src_idx, gt_arrays, sat, violneg, tp, tn = query_structs(e)
    n = len(src_idx)
    agg = {f"R@{K}cov": 0.0 for K in KS_COV}
    agg.update({"base": 0.0, "ovetoM": 0.0, "osatM": 0.0,
                "ovetoF": 0.0, "osatF": 0.0})
    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        scores = base_scores(src_idx[lo:hi], conds)
        gtm = gt_mask_batch(gt_arrays, lo, hi)
        vals, top = scores.topk(MMAX, dim=1)
        gt_top = torch.gather(gtm, 1, top)                    # (b, 200)
        w = (hi - lo) / n
        for K in KS_COV:
            agg[f"R@{K}cov"] += float(gt_top[:, :K].any(1).float().mean()) * w
        agg["base"] += float(gt_top[:, :10].any(1).float().mean()) * w
        # oracoli ristretti ai top-200 (re-rank di vals)
        for key, adj in (("ovetoM", -BIG * violneg[top].float()),
                         ("osatM", BIG * sat[top].float())):
            ord10 = (vals + adj).topk(10, dim=1).indices
            hit = torch.gather(gt_top, 1, ord10)
            agg[key] += float(hit.any(1).float().mean()) * w
        # oracoli sulla gallery piena
        for key, adj in (("ovetoF", torch.where(violneg, -torch.inf, 0.0)),
                         ("osatF", BIG * sat.float())):
            ord10 = (scores + adj).topk(10, dim=1).indices
            hit = torch.gather(gtm, 1, ord10)
            agg[key] += float(hit.any(1).float().mean()) * w
    rowsA.append({"query": e["query"], "n_src": n, **agg})
    print(f"  A {e['query']:44s} ({time.time() - t0:.0f}s)", flush=True)

m_base = float(np.mean([r["base"] for r in rowsA]))
print(f"\n=== PARTE A — autopsia (macro su 13 query) ===", flush=True)
print(f"copertura GT del base: " + "  ".join(
    f"R@{K} {np.mean([r[f'R@{K}cov'] for r in rowsA]):.3f}" for K in KS_COV))
print(f"tetto assoluto di ogni re-rank in-M: R@M_cov (sopra)")
names = {"base": "base (R@10)",
         "ovetoM": "veto oracolo   in top-200",
         "ovetoF": "veto oracolo   gallery piena",
         "osatM": "oracolo congiuntivo in top-200",
         "osatF": "oracolo congiuntivo gallery piena"}
for k, lbl in names.items():
    m = float(np.mean([r[k] for r in rowsA]))
    print(f"  {lbl:34s} {m:.3f}  ({m - m_base:+.3f})")

print(f"\nper-query (base | oveto M/F | osat M/F | R@200cov):")
for r in rowsA:
    print(f"  {r['query']:44s} {r['base']:.3f} | {r['ovetoM']:.3f}/"
          f"{r['ovetoF']:.3f} | {r['osatM']:.3f}/{r['osatF']:.3f} | "
          f"{r['R@200cov']:.3f}")

# ====================== PARTE B: sonde su gallery piena ======================
WS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
TAGS = ("cal", "znorm")
nW = len(WS)
R10 = np.zeros((len(TAGS), nW, nW, len(bench_val)))   # griglia R@10 per query

for qi, e in enumerate(bench_val):
    conds, src_idx, gt_arrays, sat, violneg, tp, tn = query_structs(e)
    n = len(src_idx)
    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        scores = base_scores(src_idx[lo:hi], conds)
        gtm = gt_mask_batch(gt_arrays, lo, hi)
        w = (hi - lo) / n
        for ti, tag in enumerate(TAGS):
            for wi, wp in enumerate(WS):
                for wj, wn in enumerate(WS):
                    S = scores + wp * tp[tag] + wn * tn[tag]
                    ord10 = S.topk(10, dim=1).indices
                    hit = torch.gather(gtm, 1, ord10)
                    R10[ti, wi, wj, qi] += float(hit.any(1).float().mean()) * w
    print(f"  B {e['query']:44s} ({time.time() - t0:.0f}s)", flush=True)

macroG = R10.mean(axis=3)                              # (tag, wp, wn)
anchor = macroG[0, 0, 0]
print(f"\n=== PARTE B — sonde su gallery piena ===")
print(f"ancora (w+=0, w-=0): {anchor:.6f} vs base {m_base:.6f}", flush=True)
assert abs(anchor - m_base) < 1e-6, "il punto nullo non riproduce il base"

n_above = int((macroG > m_base).sum())
print(f"punti griglia sopra il base: {n_above}/{macroG.size}")
print(f"\ngriglia macro R@10 (variante cal), righe w+, colonne w-:")
print("        " + "".join(f"{wn:>8g}" for wn in WS))
for wi, wp in enumerate(WS):
    print(f"  {wp:>5g} " + "".join(f"{macroG[0, wi, wj]:8.3f}"
                                   for wj in range(nW)))
print(f"\ngriglia macro R@10 (variante znorm):")
print("        " + "".join(f"{wn:>8g}" for wn in WS))
for wi, wp in enumerate(WS):
    print(f"  {wp:>5g} " + "".join(f"{macroG[1, wi, wj]:8.3f}"
                                   for wj in range(nW)))

ti, wi, wj = np.unravel_index(macroG.argmax(), macroG.shape)
tag_b, wp_b, wn_b = TAGS[ti], WS[wi], WS[wj]
m_best = float(macroG[ti, wi, wj])
print(f"\nscelto sul mirror: variante {tag_b}, w+={wp_b:g}, w-={wn_b:g} "
      f"-> macro R@10 {m_best:.3f} ({m_best - m_base:+.3f})", flush=True)

# passo finale al punto scelto: metriche complete + tabella + check violatori
rowsB, viol_fracs_base, viol_fracs_rr = [], [], []
for e in bench_val:
    conds, src_idx, gt_arrays, sat, violneg, tp, tn = query_structs(e)
    n = len(src_idx)
    negs = [c for s, c in conds if s < 0]
    negdom = len(negs) > len(conds) - len(negs)
    agg = {f"{m}@{K}": 0.0 for m in "RP" for K in KS}
    vb = {c: 0.0 for c in negs}
    vr = {c: 0.0 for c in negs}
    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        scores = base_scores(src_idx[lo:hi], conds)
        gtm = gt_mask_batch(gt_arrays, lo, hi)
        w = (hi - lo) / n
        S = scores + wp_b * tp[tag_b] + wn_b * tn[tag_b]
        ord10 = S.topk(10, dim=1).indices
        hit = torch.gather(gtm, 1, ord10).float()
        for K in KS:
            agg[f"R@{K}"] += float((hit[:, :K].sum(1) > 0).float().mean()) * w
            agg[f"P@{K}"] += float(hit[:, :K].mean()) * w
        if negdom:
            b10 = scores.topk(10, dim=1).indices
            for c in negs:
                vb[c] += float((A_VAL[b10.reshape(-1), c] > 0)
                               .float().mean()) * w
                vr[c] += float((A_VAL[ord10.reshape(-1), c] > 0)
                               .float().mean()) * w
    rowsB.append({"query": e["query"], "n_src": n, **agg})
    if negdom:
        viol_fracs_base += list(vb.values())
        viol_fracs_rr += list(vr.values())

print(f"\n{'query':<46} {'n_src':>6} {'base':>7} {'sonde':>7} {'delta':>8}")
for rA, rB, e in zip(rowsA, rowsB, bench_val):
    conds = parse_query(e["query"])
    n_neg = sum(1 for s, _ in conds if s < 0)
    tagq = (" NEG-DOM" if n_neg > len(conds) - n_neg
            else (" neg" if n_neg else ""))
    print(f"{e['query']:<46} {rB['n_src']:>6} {rA['base']:>7.3f} "
          f"{rB['R@10']:>7.3f} {rB['R@10'] - rA['base']:>+8.3f}{tagq}")
print(f"{'MACRO (13 query)':<46} {'':>6} {m_base:>7.3f} {m_best:>7.3f} "
      f"{m_best - m_base:>+8.3f}")
for name, cond in (("solo query con negazioni",
                    lambda c: any(s < 0 for s, _ in c)),
                   ("solo negative-dominanti",
                    lambda c: sum(s < 0 for s, _ in c) >
                              sum(s > 0 for s, _ in c))):
    sel = [(rA["base"], rB["R@10"]) for rA, rB, e in
           zip(rowsA, rowsB, bench_val) if cond(parse_query(e["query"]))]
    b = float(np.mean([x[0] for x in sel]))
    r = float(np.mean([x[1] for x in sel]))
    print(f"{name:<46} {'':>6} {b:>7.3f} {r:>7.3f} {r - b:>+8.3f}")
macro_best = {k: float(np.mean([r[k] for r in rowsB])) for k in
              rowsB[0] if k not in ("query", "n_src")}
print(f"\nmetriche macro al punto scelto: " +
      "  ".join(f"{k} {v:.3f}" for k, v in macro_best.items()))

v_base = float(np.mean(viol_fracs_base))
v_rr = float(np.mean(viol_fracs_rr))
print(f"\ncheck meccanicistico — %violatori top-10 (negative-dominanti): "
      f"base {100 * v_base:.1f}% -> sonde {100 * v_rr:.1f}%", flush=True)

verdict = ("MORTA (kill-test: nessun punto sopra il base)" if n_above == 0
           else f"delta {m_best - m_base:+.3f}"
                + (" (>= +0.010, ADOTTABILE)" if m_best - m_base >= 0.010
                   else " (< +0.010, rumore: non adottare)"))
print(f"\nVERDETTO parte B: {verdict}", flush=True)

json.dump({"partA": rowsA,
           "partA_macro": {k: float(np.mean([r[k] for r in rowsA]))
                           for k in rowsA[0] if k not in ("query", "n_src")},
           "partB_grid": {t: {f"wp{wp:g}_wn{wn:g}":
                              float(macroG[ti2, wi2, wj2])
                              for wi2, wp in enumerate(WS)
                              for wj2, wn in enumerate(WS)}
                          for ti2, t in enumerate(TAGS)},
           "partB_best": {"variant": tag_b, "wp": wp_b, "wn": wn_b,
                          "macro": macro_best},
           "partB_per_query": rowsB,
           "violator_rate": {"base": v_base, "rerank": v_rr},
           "n_above_base": n_above, "verdict": verdict},
          open(OUT / "autopsy_fullgallery_mirror.json", "w"), indent=1)
print(f"salvato autopsy_fullgallery_mirror.json ({time.time() - t0:.0f}s)",
      flush=True)
