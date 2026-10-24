"""Round 2, Idea 1: verifica congiuntiva con sonde (retrieve-then-verify).

Smettere di sintetizzare la query perfetta; verificare i vincoli sul candidato.
DirectionFusion (text-init, lr 3e-4, mirror 0.296) recupera i top-M, poi
re-ranking con:

  S(x) = cos(q_DF, x) + w+ * sum_{c in pos} logsig(+sc_c(x))
                      + w- * sum_{c in neg} logsig(-sc_c(x))

dove sc_c(x) = P_c . x - mid_c sono le sonde calibrate del veto experiment
(medie di classe del train, scala +-1, formula identica a killtest_veto.py).
Regola dichiarata (ereditata dal veto): termine attivo solo se AUC della sonda
su validation >= 0.75.

Griglia dichiarata in anticipo, tunata SOLO sul mirror:
  w+, w- in {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5},  M in {50, 100, 200}.
Variante (b) della proposta: stessa griglia coi logit z-normalizzati sulla
gallery (calibrazione transduttiva senza label, spirito QB-Norm).

Ancore di correttezza:
  (0) riproduzione — evaluate(DF) sul mirror deve dare il mirror_R10 del
      checkpoint (assert, altrimenti stack non riproducibile);
  (1) il punto (w+=0, w-=0, M=200) deve coincidere col base per costruzione.

Kill-test dichiarato: macro R@10 <= base su TUTTA la griglia = idea morta.
Adozione solo se delta >= +0.010. Check meccanicistico: tasso violatori nel
top-10 sulle query negative-dominanti (definizione killtest_veto) deve
scendere sotto il valore base (misurato 25.5%).

Zero training: 3 scalari (w+, w-, M) scelti a griglia sul mirror.
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


def evaluate(query_fn, bench, batch=2048):
    table = []
    for entry in bench:
        conds = parse_query(entry["query"])
        src_idx = np.array([int(k) for k in entry["ground_truth"]])
        gt_lists = [set(v) for v in entry["ground_truth"].values()]
        agg = {f"{m}@{K}": 0.0 for m in "RP" for K in KS}
        for lo in range(0, len(src_idx), batch):
            sl = slice(lo, min(lo + batch, len(src_idx)))
            idx = torch.from_numpy(src_idx[sl]).to(DEV)
            q = query_fn(V_VAL[idx], conds)
            scores = q @ V_VAL.T
            scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
            part = eval_entry(scores, gt_lists[sl])
            w = (sl.stop - sl.start) / len(src_idx)
            for k in agg:
                agg[k] += part[k] * w
        table.append({"query": entry["query"], "n_src": len(src_idx), **agg})
    return table


def macro(table, key):
    return float(np.mean([row[key] for row in table]))


# --- (0) DirectionFusion congelato, riproduzione ---------------------------
ck = torch.load(P2 / "fusion_text_s3000_lr0.0003.pt", map_location=DEV,
                weights_only=False)
E_pos, E_neg = ck["state"]["E_pos"], ck["state"]["E_neg"]


def qfn(v_ref, conds):
    delta = torch.stack([E_pos[c] if s > 0 else E_neg[c]
                         for s, c in conds]).sum(0)
    return F.normalize(v_ref + delta, dim=-1)


with torch.no_grad():
    base_table = evaluate(qfn, bench_val)
m_base = macro(base_table, "R@10")
print(f"(0) riproduzione — mirror R@10 DirectionFusion: {m_base:.3f} "
      f"(checkpoint dichiarava {ck['mirror_R10']:.3f})", flush=True)
assert abs(m_base - ck["mirror_R10"]) < 1e-6, "stack non riproducibile, fermarsi"

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
SC = V_VAL @ P.T - mid                       # (n_val, 40) punteggi calibrati
SCZ = (SC - SC.mean(0)) / SC.std(0)          # variante (b): z-norm su gallery


def auc(scores, labels):
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64,
                                device=scores.device)
    pos = labels > 0
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


bench_cols = sorted({c for e in bench_val for _, c in parse_query(e["query"])})
probe_auc = {c: auc(SC[:, c], A_VAL[:, c]) for c in bench_cols}
mask = {c: probe_auc[c] >= 0.75 for c in bench_cols}
print(f"\nAUC sonde (train->validation) sugli attributi del benchmark "
      f"[regola: termine attivo solo se >= 0.75]:", flush=True)
for c in bench_cols:
    flag = "  <-- ESCLUSO" if not mask[c] else ""
    print(f"  {ATTR_NAMES[c]:20s} AUC {probe_auc[c]:.3f}{flag}")
print(f"  media: {np.mean(list(probe_auc.values())):.3f}", flush=True)

# --- cache per query: top-Mmax base + termini sonda + membership GT ---------
MMAX = 200
WS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
MS = (50, 100, 200)


def build_cache(entry, batch=2048):
    conds = parse_query(entry["query"])
    src_idx = np.array([int(k) for k in entry["ground_truth"]])
    gt_lists = [np.array(v) for v in entry["ground_truth"].values()]
    terms = {}
    for tag, sc in (("cal", SC), ("znorm", SCZ)):
        tp = torch.zeros(N_GAL, device=DEV)
        tn = torch.zeros(N_GAL, device=DEV)
        for s, c in conds:
            if not mask[c]:
                continue
            if s > 0:
                tp += F.logsigmoid(sc[:, c])
            else:
                tn += F.logsigmoid(-sc[:, c])
        terms[tag] = (tp, tn)
    vals_l, top_l = [], []
    with torch.no_grad():
        for lo in range(0, len(src_idx), batch):
            idx = torch.from_numpy(src_idx[lo:lo + batch]).to(DEV)
            q = qfn(V_VAL[idx], conds)
            scores = q @ V_VAL.T
            scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
            v, ix = scores.topk(MMAX, dim=1)
            vals_l.append(v)
            top_l.append(ix)
    vals, top = torch.cat(vals_l), torch.cat(top_l)
    top_np = top.cpu().numpy()
    gtm = np.zeros(top_np.shape, dtype=bool)
    for i, g in enumerate(gt_lists):
        gtm[i] = np.isin(top_np[i], g)
    cache = {"conds": conds, "vals": vals, "top": top,
             "gt": torch.from_numpy(gtm).to(DEV)}
    for tag, (tp, tn) in terms.items():
        cache[tag] = (tp[top], tn[top])
    return cache


def rerank_metrics(cache, tag, wp, wn, M):
    TP, TN = cache[tag]
    S = cache["vals"][:, :M] + wp * TP[:, :M] + wn * TN[:, :M]
    ord10 = S.topk(10, dim=1).indices
    h = torch.gather(cache["gt"][:, :M].float(), 1, ord10)
    out = {}
    for K in KS:
        out[f"R@{K}"] = float((h[:, :K].sum(1) > 0).float().mean())
        out[f"P@{K}"] = float(h[:, :K].mean())
    return out


t0 = time.time()
caches = []
for e in bench_val:
    caches.append(build_cache(e))
    print(f"  cache {e['query']:44s} n_src {len(e['ground_truth']):6d} "
          f"({time.time() - t0:.0f}s)", flush=True)

# ancora (1): (0, 0, 200) deve coincidere col base
anchor = np.mean([rerank_metrics(c, "cal", 0.0, 0.0, MMAX)["R@10"]
                  for c in caches])
print(f"\n(1) ancora (w+=0, w-=0, M=200): {anchor:.6f} vs base {m_base:.6f}",
      flush=True)
assert abs(anchor - m_base) < 1e-6, "il punto nullo non riproduce il base"

# --- griglia ----------------------------------------------------------------
grids = {}          # grids[tag][f"M{M}"][f"wp{wp}_wn{wn}"] = macro R@10
per_query = {}      # per_query[(tag, M, wp, wn)] = [R@10 per query]
for tag in ("cal", "znorm"):
    grids[tag] = {}
    for M in MS:
        grids[tag][f"M{M}"] = {}
        for wp in WS:
            for wn in WS:
                rows = [rerank_metrics(c, tag, wp, wn, M) for c in caches]
                mm = float(np.mean([r["R@10"] for r in rows]))
                grids[tag][f"M{M}"][f"wp{wp:g}_wn{wn:g}"] = mm
                per_query[(tag, M, wp, wn)] = rows
    print(f"griglia '{tag}' completata ({time.time() - t0:.0f}s)", flush=True)

# --- risultati --------------------------------------------------------------
print(f"\nbase mirror R@10: {m_base:.3f}   [kill-test: morte se tutta la "
      f"griglia <= base]", flush=True)
flat = [(tag, M, wp, wn, grids[tag][f"M{M}"][f"wp{wp:g}_wn{wn:g}"])
        for tag in grids for M in MS for wp in WS for wn in WS]
n_above = sum(1 for *_, m in flat if m > m_base)
print(f"punti griglia sopra il base: {n_above}/{len(flat)}")

print(f"\nmigliore per (variante, M):")
print(f"{'variante':>8s} {'M':>5s} {'w+':>6s} {'w-':>6s} {'R@10':>7s} {'delta':>8s}")
for tag in ("cal", "znorm"):
    for M in MS:
        cand = [(wp, wn, grids[tag][f"M{M}"][f"wp{wp:g}_wn{wn:g}"])
                for wp in WS for wn in WS]
        wp_b, wn_b, m_b = max(cand, key=lambda x: x[2])
        print(f"{tag:>8s} {M:5d} {wp_b:6g} {wn_b:6g} {m_b:7.3f} "
              f"{m_b - m_base:+8.3f}")

tag_b, M_b, wp_b, wn_b, m_best = max(flat, key=lambda x: x[4])
print(f"\nscelto sul mirror: variante {tag_b}, M={M_b}, w+={wp_b:g}, "
      f"w-={wn_b:g} -> macro R@10 {m_best:.3f} ({m_best - m_base:+.3f})",
      flush=True)

# tabella per-query al punto scelto
rows_best = per_query[(tag_b, M_b, wp_b, wn_b)]
print(f"\n{'query':<46} {'n_src':>6} {'base':>7} {'rerank':>7} {'delta':>8}")
for e, rb, rr in zip(bench_val, base_table, rows_best):
    conds = parse_query(e["query"])
    n_neg = sum(1 for s, _ in conds if s < 0)
    n_pos = len(conds) - n_neg
    tagq = " NEG-DOM" if n_neg > n_pos else (" neg" if n_neg else "")
    print(f"{e['query']:<46} {rb['n_src']:>6} {rb['R@10']:>7.3f} "
          f"{rr['R@10']:>7.3f} {rr['R@10'] - rb['R@10']:>+8.3f}{tagq}")
print(f"{'MACRO (13 query)':<46} {'':>6} {m_base:>7.3f} {m_best:>7.3f} "
      f"{m_best - m_base:>+8.3f}")
for name, cond in (("solo query con negazioni",
                    lambda c: any(s < 0 for s, _ in c)),
                   ("solo negative-dominanti",
                    lambda c: sum(s < 0 for s, _ in c) >
                              sum(s > 0 for s, _ in c))):
    sel = [(rb["R@10"], rr["R@10"]) for e, rb, rr in
           zip(bench_val, base_table, rows_best)
           if cond(parse_query(e["query"]))]
    b = float(np.mean([x[0] for x in sel]))
    r = float(np.mean([x[1] for x in sel]))
    print(f"{name:<46} {'':>6} {b:>7.3f} {r:>7.3f} {r - b:>+8.3f}")
macro_best = {k: float(np.mean([r[k] for r in rows_best]))
              for k in rows_best[0]}
print(f"\nmetriche macro al punto scelto: " +
      "  ".join(f"{k} {v:.3f}" for k, v in macro_best.items()), flush=True)

# --- check meccanicistico: violatori nel top-10, query negative-dominanti ---
def violator_rate(get_top10):
    """Definizione killtest_veto: per ogni (query neg-dom, attr negato),
    frazione degli slot top-10 con label violata; media sulle coppie."""
    fracs = []
    for e, cache in zip(bench_val, caches):
        conds = cache["conds"]
        negs = [c for s, c in conds if s < 0]
        n_pos = len(conds) - len(negs)
        if not negs or len(negs) <= n_pos:
            continue
        gal10 = get_top10(cache)                       # (n_src, 10) idx galleria
        for c in negs:
            fracs.append(float((A_VAL[gal10.reshape(-1), c] > 0)
                               .float().mean()))
    return float(np.mean(fracs))


def top10_base(cache):
    return cache["top"][:, :10]


def top10_rerank(cache):
    TP, TN = cache[tag_b]
    S = cache["vals"][:, :M_b] + wp_b * TP[:, :M_b] + wn_b * TN[:, :M_b]
    return torch.gather(cache["top"][:, :M_b], 1, S.topk(10, dim=1).indices)


v_base = violator_rate(top10_base)
v_rr = violator_rate(top10_rerank)
print(f"\ncheck meccanicistico — %violatori top-10 (negative-dominanti): "
      f"base {100 * v_base:.1f}% -> rerank {100 * v_rr:.1f}%   "
      f"[deve scendere]", flush=True)

verdict = ("MORTA (kill-test: nessun punto sopra il base)" if n_above == 0
           else f"VIVA: delta {m_best - m_base:+.3f}"
                + (" (>= +0.010, adottabile)" if m_best - m_base >= 0.010
                   else " (< +0.010, rumore: non adottare)"))
print(f"\nVERDETTO: {verdict}", flush=True)

OUT.mkdir(exist_ok=True)
json.dump({"base_table": base_table, "m_base": m_base,
           "probe_auc": {ATTR_NAMES[c]: probe_auc[c] for c in bench_cols},
           "grids": grids,
           "best": {"variant": tag_b, "M": M_b, "wp": wp_b, "wn": wn_b,
                    "macro": macro_best},
           "per_query_best": [{"query": e["query"], **r}
                              for e, r in zip(bench_val, rows_best)],
           "violator_rate": {"base": v_base, "rerank": v_rr},
           "n_above_base": n_above, "verdict": verdict},
          open(OUT / "rerank_conjunctive_mirror.json", "w"), indent=1)
print(f"salvato rerank_conjunctive_mirror.json ({time.time() - t0:.0f}s)",
      flush=True)
