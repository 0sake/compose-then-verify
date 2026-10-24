"""Diagnostica STEP 0: re-rank dei top-200 con la palla di Hamming residua.

Meccanismo sotto test. L'autopsia di Idea 1 mostra (mirror): R@200cov 0.840,
oracolo congiuntivo sui T± 0.357, base 0.296. Quindi il collo di bottiglia NON
e' la verifica dei vincoli (tetto +0.061) ma il ranking del PROFILO RESIDUO:
tra i candidati che soddisfano i T±, il coseno CLIP non ordina per
"Hamming <= 2 dalla sorgente sugli altri ~37 attributi", che e' l'altra meta'
del predicato di rilevanza. Nessun metodo provato ha mai scorato quella meta':
B8/B9 (falsificati) erano ancora UN vettore query (somma lineare full-gallery,
sonde mean-diff); qui invece scoriamo OGNI candidato con
    S(x) = sum_c log P(x soddisfa vincolo c)
         + log P( #mismatch sul resto <= 2 )        [coda Poisson-binomiale]
         + lam * cos(q_DF, x)
con posteriori logistici per-attributo (train only). La coda a "budget di
errori" e' una semantica d'ordine che nessun prodotto scalare puo' esprimere,
e sconta da sola le sonde deboli (agreement ~0.5 -> poco peso).

Varianti (tutte re-rank dei top-200 del base DirectionFusion, solo mirror):
  base        : riproduzione esatta (ancora: deve dare 0.296)
  oracle_rest : label vere su vincoli+resto (ancora: deve dare R@200cov 0.840)
  dp_gtref    : profilo sorgente da label vere, candidati da sonde
                (tetto della meta' deployabile lato-candidato)
  dp_probe    : tutto da sonde — INTERAMENTE deployabile
Griglia dichiarata: lam in {0, 2, 5, 10}. Kill: se dp_probe con ogni lam
resta <= base + 0.010. Adozione (per il metodo finale): delta macro >= +0.010.
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
print(f"mirror rigenerato: {len(bench_val)} query", flush=True)

KS = (1, 5, 10)
MTOP = 200
LAMS = (0.0, 2.0, 5.0, 10.0)

# --- DirectionFusion congelato (base adottato, mirror 0.296) ----------------
ck = torch.load(P2 / "fusion_text_s3000_lr0.0003.pt", map_location=DEV,
                weights_only=False)
E_pos, E_neg = ck["state"]["E_pos"], ck["state"]["E_neg"]


def qfn(v_ref, conds):
    delta = torch.stack([E_pos[c] if s > 0 else E_neg[c]
                         for s, c in conds]).sum(0)
    return F.normalize(v_ref + delta, dim=-1)


# --- sonde logistiche su TUTTI i 40 attributi (train only, budget fisso) ----
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


def auc_t(scores, labels):
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64,
                                device=scores.device)
    pos = labels > 0
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


t0 = time.time()
W = torch.zeros(40, 512, device=DEV)
B = torch.zeros(40, device=DEV)
aucs = {}
for c in range(40):
    w, b = train_logreg(c)
    W[c], B[c] = w, b
    aucs[ATTR_NAMES[c]] = auc_t(V_VAL @ w + b, A_VAL[:, c])
print(f"sonde logistiche 40/40 in {time.time() - t0:.0f}s | "
      f"AUC val: min {min(aucs.values()):.3f} "
      f"media {np.mean(list(aucs.values())):.3f}", flush=True)
Z_VAL = V_VAL @ W.T + B                      # logit (n_val, 40)
P_VAL = torch.sigmoid(Z_VAL).clamp(1e-5, 1 - 1e-5)   # P(attr = +1 | x)


def pb_tail_le2(agree):
    """log P(#disaccordi <= 2) per agreement probs (B, M, R) -> (B, M)."""
    B_, M_, R_ = agree.shape
    st = torch.zeros(B_, M_, 3, device=agree.device)
    st[:, :, 0] = 1.0
    for i in range(R_):
        a = agree[:, :, i]
        m = 1.0 - a
        s0 = st[:, :, 0] * a
        s1 = st[:, :, 1] * a + st[:, :, 0] * m
        s2 = st[:, :, 2] * a + st[:, :, 1] * m
        st = torch.stack([s0, s1, s2], dim=-1)
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


VARIANTS = ["base", "oracle_rest", "dp_gtref", "dp_probe"]
tables = {v: [] for v in VARIANTS}
for v in VARIANTS:
    if v.startswith("dp"):
        tables[v] = {lam: [] for lam in LAMS}

for entry in bench_val:
    conds = parse_query(entry["query"])
    ccols = [c for _, c in conds]
    csigns = torch.tensor([s for s, _ in conds], device=DEV, dtype=torch.float32)
    rest = [c for c in range(40) if c not in ccols]
    src_idx = np.array([int(k) for k in entry["ground_truth"]])
    gt_lists = [set(v_) for v_ in entry["ground_truth"].values()]
    aggs = {v: ({lam: {f"{m}@{K}": 0.0 for m in "RP" for K in KS} for lam in LAMS}
                if v.startswith("dp") else {f"{m}@{K}": 0.0 for m in "RP" for K in KS})
            for v in VARIANTS}
    for lo in range(0, len(src_idx), 1024):
        sl = slice(lo, min(lo + 1024, len(src_idx)))
        idx = torch.from_numpy(src_idx[sl]).to(DEV)
        nb = len(idx)
        q = qfn(V_VAL[idx], conds)
        base = q @ V_VAL.T
        base[torch.arange(nb, device=DEV), idx] = -torch.inf
        w = (sl.stop - sl.start) / len(src_idx)

        top = base.topk(MTOP, dim=1).indices                  # (B, M)
        cosM = base.gather(1, top)                            # (B, M)

        # lato candidato: posteriori sonde e label vere sui top-200
        p_cand = P_VAL[top]                                   # (B, M, 40)
        a_cand = A_VAL[top]                                   # (B, M, 40)

        # vincoli: log P(soddisfatti) via sonde / via label
        pc = p_cand[:, :, ccols]                              # (B, M, C)
        target_pos = (csigns > 0).view(1, 1, -1)
        p_sat = torch.where(target_pos, pc, 1 - pc)
        lg_con_probe = torch.log(p_sat).sum(-1)
        sat_true = torch.stack([(a_cand[:, :, c] == s) for s, c in conds],
                               dim=-1).all(-1)                # (B, M)

        # resto: agreement prob (sonde) e Hamming vero (label)
        p_r = p_cand[:, :, rest]                              # (B, M, R)
        pref_probe = P_VAL[idx][:, None, rest]                # (B, 1, R)
        aref_true = (A_VAL[idx][:, None, rest] > 0).float()   # (B, 1, R)
        agree_pp = p_r * pref_probe + (1 - p_r) * (1 - pref_probe)
        agree_gp = p_r * aref_true + (1 - p_r) * (1 - aref_true)
        lg_rest_pp = pb_tail_le2(agree_pp)
        lg_rest_gp = pb_tail_le2(agree_gp)
        ham_true = (a_cand[:, :, rest] !=
                    A_VAL[idx][:, None, rest]).sum(-1)        # (B, M)

        def scatter_eval(newscore, table_agg):
            full = torch.full_like(base, -torch.inf)
            full.scatter_(1, top, newscore)
            part = eval_entry(full, gt_lists[sl])
            for k in table_agg:
                table_agg[k] += part[k] * w

        scatter_eval(cosM, aggs["base"])
        oracle = sat_true.float() * (ham_true <= 2).float()
        scatter_eval(oracle + 1e-4 * cosM, aggs["oracle_rest"])
        for lam in LAMS:
            scatter_eval(lg_con_probe + lg_rest_gp + lam * cosM,
                         aggs["dp_gtref"][lam])
            scatter_eval(lg_con_probe + lg_rest_pp + lam * cosM,
                         aggs["dp_probe"][lam])

    row = {"query": entry["query"], "n_src": len(src_idx)}
    tables["base"].append({**row, **aggs["base"]})
    tables["oracle_rest"].append({**row, **aggs["oracle_rest"]})
    for v in ("dp_gtref", "dp_probe"):
        for lam in LAMS:
            tables[v][lam].append({**row, **aggs[v][lam]})
    print(f"{entry['query']:44s} base {aggs['base']['R@10']:.3f} | "
          f"oracle_rest {aggs['oracle_rest']['R@10']:.3f} | "
          f"gtref(l0) {aggs['dp_gtref'][0.0]['R@10']:.3f} | "
          f"probe(l0) {aggs['dp_probe'][0.0]['R@10']:.3f} | "
          f"probe(l5) {aggs['dp_probe'][5.0]['R@10']:.3f}", flush=True)


def macro(table, key="R@10"):
    return float(np.mean([r[key] for r in table]))


out = {"auc_val_40": aucs,
       "base_macro": macro(tables["base"]),
       "oracle_rest_macro": macro(tables["oracle_rest"]),
       "dp_gtref_macro": {str(l): macro(tables["dp_gtref"][l]) for l in LAMS},
       "dp_probe_macro": {str(l): macro(tables["dp_probe"][l]) for l in LAMS},
       "tables": {"base": tables["base"], "oracle_rest": tables["oracle_rest"],
                  "dp_gtref": {str(l): tables["dp_gtref"][l] for l in LAMS},
                  "dp_probe": {str(l): tables["dp_probe"][l] for l in LAMS}}}
OUT.mkdir(exist_ok=True)
with open(OUT / "restball_rerank_mirror.json", "w") as f:
    json.dump(out, f, indent=1)
print("\n=== MACRO R@10 (mirror, re-rank top-200) ===")
print(f"base            {out['base_macro']:.3f}   (ancora: 0.296)")
print(f"oracle_rest     {out['oracle_rest_macro']:.3f}   (ancora: 0.840)")
for l in LAMS:
    print(f"dp_gtref lam={l:<4} {out['dp_gtref_macro'][str(l)]:.3f}")
for l in LAMS:
    print(f"dp_probe lam={l:<4} {out['dp_probe_macro'][str(l)]:.3f}")
print(f"salvato in {OUT / 'restball_rerank_mirror.json'} "
      f"({time.time() - t0:.0f}s totali)")
