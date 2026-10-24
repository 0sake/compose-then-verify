"""Idea 1, terza tappa: sonde migliori + forme discrete (chiusura del gap oracolo).

L'autopsia ha stabilito: il tetto della verifica congiuntiva e' +0.061 (oracolo
label, 0.357) ed e' interamente raggiungibile dentro i top-200; il veto oracolo
+0.027 idem. Le sonde medie-di-classe col termine continuo logsig catturano
+0.000: ogni peso sostanziale degrada. Il collo di bottiglia e' la conversione
label -> sonda, non la forma della proposta ne' la restrizione top-M.

Due mosse dichiarate (una alla volta, stessa filosofia retrieve-then-verify):
  1. sonde LOGISTICHE (regressione logistica lineare sul train, stesse label
     e stesso protocollo delle medie-di-classe, ma ottimizzate sui casi
     difficili) accanto alle medie-di-classe;
  2. forme DISCRETE identiche agli oracoli, con decisioni sogliate:
       veto duro:        S = cos,  S[x] = -inf se qualche attr negato e'
                         deciso presente (sc_c(x) > tau)
       bonus congiuntivo: S = cos + BIG * [tutte le condizioni decise
                         soddisfatte]
     Un solo tau globale condiviso tra gli attributi, sweep dichiarato sul
     mirror: meandiff tau in {-0.5,-0.25,0,0.25,0.5} (unita' calibrate, medie
     di classe a +-1); logistic tau in {-2,-1,0,1,2} (logit).

Diagnosi meccanicistica (prima delle ablazioni): AUC delle sonde ristretta al
POOL dei top-200 del base (i candidati che il re-ranking deve discriminare)
vs AUC globale. Predizione P_C: l'AUC di pool crolla rispetto alla globale
(i top-200 concentrano i casi al confine) tranne che per Male (0.999), il che
spiega perche' solo -Male,-Mustache converte.

Regole invariate: kill se tutta la griglia <= base; adozione solo se delta
macro >= +0.010; solo mirror; zero training del modulo (le sonde logistiche
usano solo le label di train, come le medie di classe e le direzioni B5).
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


def auc_t(scores, labels):
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64,
                                device=scores.device)
    pos = labels > 0
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


BENCH_COLS = sorted({c for e in bench_val for _, c in parse_query(e["query"])})

# --- sonde: medie di classe (killtest_veto) e logistiche ---------------------
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
SC_MD = V_VAL @ P.T - mid                     # (n_val, 40) unita' calibrate

t0 = time.time()


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


SC_LG = torch.zeros(N_GAL, 40, device=DEV)
print("training sonde logistiche (12 attributi benchmark):", flush=True)
auc_rows = []
for c in BENCH_COLS:
    w, b = train_logreg(c)
    SC_LG[:, c] = V_VAL @ w + b
    a_md = auc_t(SC_MD[:, c], A_VAL[:, c])
    a_lg = auc_t(SC_LG[:, c], A_VAL[:, c])
    auc_rows.append({"attr": ATTR_NAMES[c], "auc_meandiff": a_md,
                     "auc_logistic": a_lg})
    print(f"  {ATTR_NAMES[c]:20s} AUC val: meandiff {a_md:.3f}  "
          f"logistic {a_lg:.3f}  ({time.time() - t0:.0f}s)", flush=True)
print(f"  media: meandiff "
      f"{np.mean([r['auc_meandiff'] for r in auc_rows]):.3f}  logistic "
      f"{np.mean([r['auc_logistic'] for r in auc_rows]):.3f}", flush=True)

SCORES = {"meandiff": SC_MD, "logistic": SC_LG}
TAUS = {"meandiff": (-0.5, -0.25, 0.0, 0.25, 0.5),
        "logistic": (-2.0, -1.0, 0.0, 1.0, 2.0)}
BIG = 1e4
BATCH = 2048


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


# --- diagnosi: AUC globale vs AUC sul pool dei top-200 del base --------------
print(f"\ndiagnosi P_C — AUC sull'attributo negato: globale vs pool top-200 "
      f"del base (query con negazioni):", flush=True)
print(f"{'query':<46} {'attr':<16} {'glob md/lg':>12} {'pool md/lg':>12} "
      f"{'%viol pool':>10}")
pool_rows = []
for e in bench_val:
    conds = parse_query(e["query"])
    negs = [c for s, c in conds if s < 0]
    if not negs:
        continue
    src_idx = np.array([int(k) for k in e["ground_truth"]])
    seen_pool = torch.zeros(N_GAL, dtype=torch.bool, device=DEV)
    for lo in range(0, len(src_idx), BATCH):
        hi = min(lo + BATCH, len(src_idx))
        top = base_scores(src_idx[lo:hi], conds).topk(200, dim=1).indices
        seen_pool[top.reshape(-1)] = True
    pool = torch.nonzero(seen_pool, as_tuple=True)[0]
    for c in negs:
        g_md = auc_t(SC_MD[:, c], A_VAL[:, c])
        g_lg = auc_t(SC_LG[:, c], A_VAL[:, c])
        p_md = auc_t(SC_MD[pool, c], A_VAL[pool, c])
        p_lg = auc_t(SC_LG[pool, c], A_VAL[pool, c])
        pv = float((A_VAL[pool, c] > 0).float().mean())
        pool_rows.append({"query": e["query"], "attr": ATTR_NAMES[c],
                          "pool_size": int(len(pool)),
                          "auc_global_md": g_md, "auc_global_lg": g_lg,
                          "auc_pool_md": p_md, "auc_pool_lg": p_lg,
                          "viol_frac_pool": pv})
        print(f"{e['query']:<46} {ATTR_NAMES[c]:<16} "
              f"{g_md:.3f}/{g_lg:.3f} {p_md:>6.3f}/{p_lg:.3f} "
              f"{100 * pv:>9.1f}%", flush=True)

# --- ablazioni: veto duro e bonus congiuntivo, decisioni sogliate ------------
FORMS = ("veto", "bonus")
CFGS = [(pt, tau, form) for pt in SCORES for tau in TAUS[pt] for form in FORMS]
R10 = {cfg: np.zeros(len(bench_val)) for cfg in CFGS}

for qi, e in enumerate(bench_val):
    conds = parse_query(e["query"])
    src_idx = np.array([int(k) for k in e["ground_truth"]])
    gt_arrays = [np.array(v) for v in e["ground_truth"].values()]
    n = len(src_idx)
    # maschere di decisione per (ptype, tau): viol_any e sat_all sulla gallery
    masks = {}
    for pt in SCORES:
        sc = SCORES[pt]
        for tau in TAUS[pt]:
            viol_any = torch.zeros(N_GAL, dtype=torch.bool, device=DEV)
            sat_all = torch.ones(N_GAL, dtype=torch.bool, device=DEV)
            for s, c in conds:
                has = sc[:, c] > tau
                if s < 0:
                    viol_any |= has
                    sat_all &= ~has
                else:
                    sat_all &= has
            masks[(pt, tau)] = (viol_any, sat_all)
    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        scores = base_scores(src_idx[lo:hi], conds)
        gtm = gt_mask_batch(gt_arrays, lo, hi)
        w = (hi - lo) / n
        for pt in SCORES:
            for tau in TAUS[pt]:
                viol_any, sat_all = masks[(pt, tau)]
                for form in FORMS:
                    if form == "veto":
                        if not any(s < 0 for s, _ in conds):
                            S = scores
                        else:
                            S = scores.masked_fill(viol_any, -torch.inf)
                    else:
                        S = scores + BIG * sat_all.float()
                    ord10 = S.topk(10, dim=1).indices
                    hit = torch.gather(gtm, 1, ord10)
                    R10[(pt, tau, form)][qi] += \
                        float(hit.any(1).float().mean()) * w
    print(f"  abl {e['query']:44s} ({time.time() - t0:.0f}s)", flush=True)

# base per-query dal riferimento veto (stessa pipeline, riprodotta due volte
# nelle tappe precedenti con assert), mappato per query
base_by_q = {row["query"]: row["R@10"]
             for row in json.load(open(VETO_JSON))["base"]}
base_pq = np.array([base_by_q[e["query"]] for e in bench_val])
m_base = float(base_pq.mean())
print(f"\nbase mirror R@10 (riferimento): {m_base:.3f}", flush=True)

print(f"\nmacro R@10 per configurazione (base {m_base:.3f}):")
print(f"{'sonda':>9s} {'tau':>6s} {'veto':>7s} {'bonus':>7s}")
for pt in SCORES:
    for tau in TAUS[pt]:
        mv = float(R10[(pt, tau, 'veto')].mean())
        mb = float(R10[(pt, tau, 'bonus')].mean())
        print(f"{pt:>9s} {tau:>6g} {mv:>7.3f} {mb:>7.3f}")

flat = [(pt, tau, form, float(R10[(pt, tau, form)].mean()))
        for (pt, tau, form) in CFGS]
n_above = sum(1 for *_, m in flat if m > m_base)
pt_b, tau_b, form_b, m_best = max(flat, key=lambda x: x[3])
print(f"\npunti sopra il base: {n_above}/{len(flat)}")
print(f"scelto sul mirror: {form_b} con sonda {pt_b}, tau={tau_b:g} -> "
      f"macro R@10 {m_best:.3f} ({m_best - m_base:+.3f})", flush=True)

print(f"\n{'query':<46} {'base':>7} {'best':>7} {'delta':>8}   "
      f"[oracolo: veto 0.323, congiuntivo 0.357]")
best_pq = R10[(pt_b, tau_b, form_b)]
for qi, e in enumerate(bench_val):
    conds = parse_query(e["query"])
    n_neg = sum(1 for s, _ in conds if s < 0)
    tagq = (" NEG-DOM" if n_neg > len(conds) - n_neg
            else (" neg" if n_neg else ""))
    print(f"{e['query']:<46} {base_pq[qi]:>7.3f} {best_pq[qi]:>7.3f} "
          f"{best_pq[qi] - base_pq[qi]:>+8.3f}{tagq}")
print(f"{'MACRO (13 query)':<46} {m_base:>7.3f} {m_best:>7.3f} "
      f"{m_best - m_base:>+8.3f}")

verdict = ("MORTA (nessuna configurazione sopra il base)" if n_above == 0
           else f"delta {m_best - m_base:+.3f}"
                + (" (>= +0.010, ADOTTABILE)" if m_best - m_base >= 0.010
                   else " (< +0.010, rumore: non adottare)"))
print(f"\nVERDETTO: {verdict}", flush=True)

json.dump({"auc": auc_rows, "pool_diagnostic": pool_rows,
           "grid": {f"{pt}_tau{tau:g}_{form}":
                    {"macro": float(R10[(pt, tau, form)].mean()),
                     "per_query": R10[(pt, tau, form)].tolist()}
                    for (pt, tau, form) in CFGS},
           "base_per_query": base_pq.tolist(), "m_base": m_base,
           "best": {"probe": pt_b, "tau": tau_b, "form": form_b,
                    "macro": m_best},
           "n_above_base": n_above, "verdict": verdict},
          open(OUT / "probe_hard_forms_mirror.json", "w"), indent=1)
print(f"salvato probe_hard_forms_mirror.json ({time.time() - t0:.0f}s)",
      flush=True)
