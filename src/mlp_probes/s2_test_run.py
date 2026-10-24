"""RUN DI TEST (unico) — stack congelato sul mirror, 19 lug.

Stack: DirectionFusion (fusion_text_s3000_lr0.0003.pt) -> re-rank top-200 con
S = sum_c log P(vincolo c) + log P(#mismatch resto <= 2) + 1e-4 * cos,
posteriori da 40 sonde MLP 512->64->1 (train-90, temperatura su held-10,
seed 0, 1500 passi AdamW batch 8192, lr 1e-3, wd 1e-4) — gradino `mlp_global`
di S2, adottato sul mirror a 0.523. M=200, lambda=0: tutto congelato.

Protocollo del run:
  1. ancora A (stack): mirror macro R@10 deve riprodurre 0.523;
  2. ancora B (harness test): base DirectionFusion sul test deve riprodurre
     il numero gia' speso 0.300;
  3. UN solo numero nuovo: test del metodo, con CI bootstrap appaiato (1000
     rep, seed 0) sul delta macro R@10 vs base.
Nessun'altra variante viene valutata sul test.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.paths import CELEBA, EVAL_JSON, CHECKPOINTS, load_features, results_dir
CKPT = CHECKPOINTS / "fusion_text_s3000_lr0.0003.pt"
OUT = results_dir("mlp_probes")
DEV = "cuda"
torch.manual_seed(0)
T0 = time.time()

# --- dati -------------------------------------------------------------------
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
N_TRAIN = E_TRAIN.shape[0]
A_TRAIN = torch.from_numpy(A_SPLIT["train"].astype(np.float32)).to(DEV)
Y_ALL = (A_TRAIN > 0).float()
V = {"val": E[SPLIT_IDX["val"]].to(DEV), "test": E[SPLIT_IDX["test"]].to(DEV)}
A_V = {s: torch.from_numpy(A_SPLIT[s].astype(np.float32)).to(DEV)
       for s in ("val", "test")}
print(f"dati pronti ({time.time() - T0:.0f}s)", flush=True)


def parse_query(q):
    return [(1 if t.strip()[0] == "+" else -1, NAME2COL[t.strip()[1:].strip()])
            for t in q.split(",")]


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
    BENCH = json.load(f)                      # 14 entry ufficiali (test)
bench_val, seen = [], set()
for entry in BENCH:
    if entry["query"] in seen:
        continue
    seen.add(entry["query"])
    targets, viol = rebuild_ground_truth(entry["query"], A_SPLIT["val"])
    gt = {str(j): t for j, t in targets.items() if viol[j]}
    bench_val.append({"query": entry["query"], "ground_truth": gt})
print(f"benchmark: test {len(BENCH)} entry (ufficiale), "
      f"mirror {len(bench_val)} ({time.time() - T0:.0f}s)", flush=True)

KS = (1, 5, 10)


def eval_entry(scores, gt_lists):
    topk = scores.topk(max(KS), dim=1).indices.cpu().numpy()
    out = {}
    hit10 = []
    for i, gt in enumerate(gt_lists):
        hit10.append(int(len(set(topk[i, :10].tolist()) & gt) > 0))
    for K in KS:
        r = p = 0.0
        for i, gt in enumerate(gt_lists):
            hits = len(set(topk[i, :K].tolist()) & gt)
            r += hits > 0
            p += hits / K
        out[f"R@{K}"] = r / len(gt_lists)
        out[f"P@{K}"] = p / len(gt_lists)
    return out, np.array(hit10, dtype=np.uint8)


def macro(table, key="R@10"):
    return float(np.mean([row[key] for row in table]))


# --- DirectionFusion congelato ----------------------------------------------
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
E_pos, E_neg = ck["state"]["E_pos"], ck["state"]["E_neg"]


def qfn(v_ref, conds):
    delta = torch.stack([E_pos[c] if s > 0 else E_neg[c]
                         for s, c in conds]).sum(0)
    return F.normalize(v_ref + delta, dim=-1)


# --- sonde MLP: ricetta S2 identica (seed 0) --------------------------------
g_cpu = torch.Generator().manual_seed(0)
perm = torch.randperm(N_TRAIN, generator=g_cpu)
TR90 = perm[:int(0.9 * N_TRAIN)].to(DEV)
HOLD = perm[int(0.9 * N_TRAIN):].to(DEV)


def mlp_init(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    W1 = (torch.randn(40, 512, 64, generator=g, device=DEV) * (2 / 512) ** 0.5
          ).requires_grad_()
    b1 = torch.zeros(40, 64, device=DEV, requires_grad=True)
    W2 = (torch.randn(40, 64, generator=g, device=DEV) * (1 / 64) ** 0.5
          ).requires_grad_()
    b2 = torch.zeros(40, device=DEV, requires_grad=True)
    return [W1, b1, W2, b2]


def mlp_logits(X, prm, chunk=16384):
    W1, b1, W2, b2 = prm
    outs = []
    for lo in range(0, len(X), chunk):
        h = torch.relu(torch.einsum("bd,adh->bah", X[lo:lo + chunk], W1) + b1)
        outs.append((h * W2).sum(-1) + b2)
    return torch.cat(outs)


def train_mlp(steps=1500, bs=8192, lr=1e-3, wd=1e-4, seed=0):
    prm = mlp_init(seed)
    opt = torch.optim.AdamW(prm, lr=lr, weight_decay=wd)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(steps):
        idx = TR90[torch.randint(len(TR90), (bs,), generator=g, device=DEV)]
        z = mlp_logits(E_TRAIN[idx], prm, chunk=bs)
        loss = F.binary_cross_entropy_with_logits(z, Y_ALL[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [p_.detach() for p_ in prm]


@torch.no_grad()
def fit_temperature(z_hold, y_hold):
    Ts = torch.exp(torch.linspace(np.log(0.25), np.log(4.0), 25, device=DEV))
    T_out = torch.ones(40, device=DEV)
    for c in range(40):
        bces = torch.stack([
            F.binary_cross_entropy_with_logits(z_hold[:, c] / T_, y_hold[:, c])
            for T_ in Ts])
        T_out[c] = Ts[bces.argmin()]
    return T_out


t = time.time()
prm = train_mlp()
with torch.no_grad():
    T_ = fit_temperature(mlp_logits(E_TRAIN[HOLD], prm), Y_ALL[HOLD])
    P_POST = {s: torch.sigmoid(mlp_logits(V[s], prm) / T_).clamp(1e-5, 1 - 1e-5)
              for s in ("val", "test")}
print(f"sonde MLP addestrate ({time.time() - t:.0f}s)", flush=True)

M = 200


def pb_tail_le2(agree):
    B_, M_, _ = agree.shape
    st = torch.zeros(B_, M_, 3, device=agree.device)
    st[:, :, 0] = 1.0
    for i in range(agree.shape[-1]):
        a = agree[:, :, i]
        m = 1.0 - a
        st = torch.stack([st[:, :, 0] * a,
                          st[:, :, 1] * a + st[:, :, 0] * m,
                          st[:, :, 2] * a + st[:, :, 1] * m], dim=-1)
    return torch.log(st.sum(-1).clamp_min(1e-30))


def run_eval(bench, split, collect_hits=False):
    '''Tabelle per-entry di base e stack (stesso topk, stesso pool).'''
    Vs, P_post = V[split], P_POST[split]
    tab_base, tab_new, hits_base, hits_new = [], [], [], []
    for entry in bench:
        conds = parse_query(entry["query"])
        ccols = [c for _, c in conds]
        rest = [c for c in range(40) if c not in ccols]
        csigns = torch.tensor([s for s, _ in conds], device=DEV) > 0
        src_idx = np.array([int(k) for k in entry["ground_truth"]])
        gt_lists = [set(v) for v in entry["ground_truth"].values()]
        agg_b = {f"{m_}@{K}": 0.0 for m_ in "RP" for K in KS}
        agg_n = dict(agg_b)
        hb, hn = [], []
        for lo in range(0, len(src_idx), 2048):
            sl = slice(lo, min(lo + 2048, len(src_idx)))
            idx = torch.from_numpy(src_idx[sl]).to(DEV)
            q = qfn(Vs[idx], conds)
            sc = q @ Vs.T
            sc[torch.arange(len(idx), device=DEV), idx] = -torch.inf
            part_b, h_b = eval_entry(sc, gt_lists[sl])
            top = sc.topk(M, dim=1).indices
            cos = sc.gather(1, top)
            p_cand = P_post[top]
            pc = p_cand[:, :, ccols]
            lg_con = torch.log(torch.where(csigns.view(1, 1, -1),
                                           pc, 1 - pc)).sum(-1)
            p_r = p_cand[:, :, rest]
            pref = P_post[idx][:, None, rest]
            lg_rest = pb_tail_le2(p_r * pref + (1 - p_r) * (1 - pref))
            full = torch.full_like(sc, -torch.inf)
            full.scatter_(1, top, lg_con + lg_rest + 1e-4 * cos)
            part_n, h_n = eval_entry(full, gt_lists[sl])
            w = (sl.stop - sl.start) / len(src_idx)
            for k in agg_b:
                agg_b[k] += part_b[k] * w
                agg_n[k] += part_n[k] * w
            hb.append(h_b)
            hn.append(h_n)
        row = {"query": entry["query"], "n_src": len(src_idx)}
        tab_base.append({**row, **agg_b})
        tab_new.append({**row, **agg_n})
        if collect_hits:
            hits_base.append(np.concatenate(hb))
            hits_new.append(np.concatenate(hn))
    return tab_base, tab_new, hits_base, hits_new


# --- ancora A: mirror deve riprodurre 0.523 ---------------------------------
t = time.time()
_, tab_val_new, _, _ = run_eval(bench_val, "val")
MIRROR_STACK = macro(tab_val_new)
print(f"ancora A — stack sul mirror: {MIRROR_STACK:.3f} (atteso 0.523)  "
      f"({time.time() - t:.0f}s)", flush=True)

# --- TEST: ancora B (base 0.300 gia' speso) + numero nuovo ------------------
t = time.time()
tab_tb, tab_tn, hits_b, hits_n = run_eval(BENCH, "test", collect_hits=True)
print(f"ancora B — base DirectionFusion test: {macro(tab_tb):.3f} "
      f"(gia' speso: 0.300)", flush=True)
print(f"\n=== TEST (run unico) — stack mlp_global ===", flush=True)
print(f"{'query':44s} {'n_src':>5s}  {'base':>6s} {'stack':>6s}")
for rb, rn in zip(tab_tb, tab_tn):
    print(f"{rb['query']:44s} {rb['n_src']:5d}  {rb['R@10']:6.3f} "
          f"{rn['R@10']:6.3f}")
for k in ("R@1", "R@5", "R@10", "P@1", "P@5", "P@10"):
    print(f"macro {k:5s}: base {macro(tab_tb, k):.3f} -> stack "
          f"{macro(tab_tn, k):.3f}")

# bootstrap appaiato sul delta macro R@10 (1000 rep, seed 0)
rng = np.random.default_rng(0)
deltas = np.empty(1000)
for r in range(1000):
    db = dn = 0.0
    for hb, hn in zip(hits_b, hits_n):
        ridx = rng.integers(0, len(hb), len(hb))
        db += hb[ridx].mean()
        dn += hn[ridx].mean()
    deltas[r] = (dn - db) / len(hits_b)
lo_ci, hi_ci = np.percentile(deltas, [2.5, 97.5])
DELTA = macro(tab_tn) - macro(tab_tb)
print(f"\ndelta R@10 macro: {DELTA:+.3f}, CI bootstrap 95% appaiato "
      f"[{lo_ci:+.3f}, {hi_ci:+.3f}]")

out = {"mirror_stack_anchor": MIRROR_STACK,
       "test_base_anchor": macro(tab_tb),
       "test_base_table": tab_tb, "test_stack_table": tab_tn,
       "test_macros": {k: macro(tab_tn, k)
                       for k in ("R@1", "R@5", "R@10", "P@1", "P@5", "P@10")},
       "delta_R10": DELTA, "ci95": [float(lo_ci), float(hi_ci)]}
with open(OUT / "s2_test_run.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\nsalvato in {OUT / 's2_test_run.json'} ({time.time() - T0:.0f}s totali)")
