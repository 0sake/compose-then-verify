"""Proposta 2, passo finale: DirectionFusion con init testuale vs init appaiate.

Protocollo identico alla pipeline (Sez. 8b del report): InfoNCE su query
sintetiche campionate con la regola esatta del benchmark, AdamW, budget fisso
3000 passi senza early stopping, lr scelta sul mirror tra {1e-4, 3e-4, 1e-3},
seed 0. Init = B3 ottimizzato del rispettivo set di direzioni:
  text   : D_text,  (alpha*, beta*) dalla griglia  (1.75, 1.00)
  paired : D_pair,  (alpha*, beta*) dalla griglia  (0.75, 0.50)
Il run text e' il controllo di riproducibilita' (sessione precedente: 0.296).
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.paths import CELEBA, EVAL_JSON, VETO_JSON, CHECKPOINTS, load_features, results_dir
OUT = results_dir("p2_paired_directions")
DEV = "cuda"
SEED = 0

# --- dati (identico a ablation_b3_directions.py) ----------------------------
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
V = {s: E[SPLIT_IDX[s]].to(DEV) for s in SPLIT_IDX}
E_TRAIN = V["train"]
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


def evaluate(query_fn, bench, split, batch=2048):
    Vs = V[split]
    table = []
    for entry in bench:
        conds = parse_query(entry["query"])
        src_idx = np.array([int(k) for k in entry["ground_truth"]])
        gt_lists = [set(v) for v in entry["ground_truth"].values()]
        agg = {f"{m}@{K}": 0.0 for m in "RP" for K in KS}
        for lo in range(0, len(src_idx), batch):
            sl = slice(lo, min(lo + batch, len(src_idx)))
            idx = torch.from_numpy(src_idx[sl]).to(DEV)
            q = query_fn(Vs[idx], conds)
            scores = q @ Vs.T
            scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
            part = eval_entry(scores, gt_lists[sl])
            w = (sl.stop - sl.start) / len(src_idx)
            for k in agg:
                agg[k] += part[k] * w
        table.append({"query": entry["query"], "n_src": len(src_idx), **agg})
    return table


def macro(table, key):
    return float(np.mean([row[key] for row in table]))


# --- direzioni e coefficienti dall'ablazione B3 ------------------------------
DIRS = torch.load(CHECKPOINTS / "directions_all_sets.pt", map_location=DEV,
                  weights_only=False)
B3RES = json.load(open(OUT / "b3_direction_ablation_mirror.json"))
INITS = {
    "text": (DIRS["D_text"].to(DEV), B3RES["text"]["alpha"], B3RES["text"]["beta"]),
    "paired": (DIRS["D_pair"].to(DEV), B3RES["paired"]["alpha"], B3RES["paired"]["beta"]),
}
for name, (D, a, b) in INITS.items():
    print(f"init {name}: (alpha, beta) = ({a}, {b})", flush=True)

# --- campionamento e training (Sez. 8b, verbatim) ----------------------------
GEN = torch.Generator().manual_seed(SEED)


def sample_group(n_src, max_tries=20):
    for _ in range(max_tries):
        k = int(torch.randint(1, 4, (1,), generator=GEN))
        cols = torch.randperm(40, generator=GEN)[:k]
        signs = torch.randint(0, 2, (k,), generator=GEN) * 2.0 - 1.0
        cols_d, signs_d = cols.to(DEV), signs.to(DEV)

        sat = (A_TRAIN[:, cols_d] == signs_d).all(1)
        viol = (A_TRAIN[:, cols_d] == -signs_d).all(1)
        if int(sat.sum()) < 5 or int(viol.sum()) < n_src:
            continue

        cand_idx = torch.nonzero(sat, as_tuple=True)[0]
        viol_idx = torch.nonzero(viol, as_tuple=True)[0]
        pick = viol_idx[torch.randperm(len(viol_idx), device=DEV)[:4 * n_src]]

        rest = torch.tensor([c for c in range(40) if c not in cols.tolist()],
                            device=DEV)
        dots = A_TRAIN[cand_idx][:, rest] @ A_TRAIN[pick][:, rest].T
        valid = (len(rest) - dots) / 2 <= 2
        ok = torch.nonzero(valid.sum(0) >= 5, as_tuple=True)[0][:n_src]
        if len(ok) < n_src:
            continue

        srcs = pick[ok]
        tgt = []
        for j in ok.tolist():
            v = torch.nonzero(valid[:, j], as_tuple=True)[0]
            tgt.append(cand_idx[v[torch.randint(0, len(v), (1,), device=DEV)]])
        return cols_d, signs_d, srcs, torch.cat(tgt)
    return None


class DirectionFusion(nn.Module):
    def __init__(self, D, a_init, b_init):
        super().__init__()
        self.E_pos = nn.Parameter(a_init * D.clone())
        self.E_neg = nn.Parameter(-b_init * D.clone())

    def forward(self, v_ref, cols, signs):
        q = v_ref.clone()
        for c, s in zip(cols.tolist(), signs.tolist()):
            q = q + (self.E_pos[c] if s > 0 else self.E_neg[c])
        return F.normalize(q, dim=-1)


def query_fn_from(model):
    def fn(v_ref, conds):
        cols = torch.tensor([c for _, c in conds], device=DEV)
        signs = torch.tensor([float(s) for s, _ in conds], device=DEV)
        with torch.no_grad():
            return model(v_ref, cols, signs)
    return fn


def train(model, steps, lr, groups=8, n_src=32, log_every=100):
    global GEN
    GEN = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    log_tau = nn.Parameter(torch.tensor(math.log(0.07), device=DEV))
    opt.add_param_group({"params": [log_tau], "lr": lr})
    curve, t0 = [], time.time()
    for step in range(1, steps + 1):
        qs, ts, ss = [], [], []
        while len(qs) < groups:
            g = sample_group(n_src)
            if g is None:
                continue
            cols, signs, srcs, tgts = g
            qs.append(model(E_TRAIN[srcs], cols, signs))
            ts.append(E_TRAIN[tgts])
            ss.append(E_TRAIN[srcs])
        Q = torch.cat(qs)
        bank = torch.cat(ts + ss)
        logits = Q @ bank.T / log_tau.exp()
        loss = F.cross_entropy(logits, torch.arange(len(Q), device=DEV))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            curve.append({"step": step, "loss": loss.item(),
                          "tau": log_tau.exp().item()})
            if step % (log_every * 5) == 0 or step == 1:
                print(f"  step {step:5d}  loss {loss.item():.4f}  "
                      f"tau {log_tau.exp().item():.3f}  "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return curve


LRS = (1e-4, 3e-4, 1e-3)
runs = {}
for name, (D, a0, b0) in INITS.items():
    for lr in LRS:
        tag = f"{name}_s3000_lr{lr:g}"
        path = CHECKPOINTS / f"fusion_{tag}.pt"
        torch.manual_seed(SEED)
        model = DirectionFusion(D, a0, b0).to(DEV)
        if path.exists():
            ck = torch.load(path, map_location=DEV, weights_only=False)
            model.load_state_dict(ck["state"])
            m = ck["mirror_R10"]
            print(f"[{tag}] da cache (mirror R@10 {m:.3f})", flush=True)
        else:
            print(f"--- {tag} ---", flush=True)
            curve = train(model, 3000, lr)
            m = macro(evaluate(query_fn_from(model), bench_val, "val"), "R@10")
            torch.save({"state": model.state_dict(), "curve": curve,
                        "mirror_R10": m}, path)
            print(f"[{tag}] mirror macro R@10 {m:.3f}", flush=True)
        runs[(name, lr)] = (model, m)

print(f"\n{'init':>8s} {'lr':>8s} {'mirror R@10':>12s}")
for (name, lr), (_, m) in runs.items():
    print(f"{name:>8s} {lr:8g} {m:12.3f}")

best = {}
for name in INITS:
    lr_sel = max(LRS, key=lambda lr: runs[(name, lr)][1])
    best[name] = (lr_sel, runs[(name, lr_sel)][0])
    print(f"selezionato sul mirror per {name}: lr = {lr_sel:g} "
          f"(R@10 {runs[(name, lr_sel)][1]:.3f})")

# --- tabella per-query: text-init vs paired-init (+ riferimento veto base) ---
tables = {n: evaluate(query_fn_from(m), bench_val, "val")
          for n, (_, m) in best.items()}
base_prev = {row["query"]: row["R@10"] for row in json.load(open(VETO_JSON))["base"]}

print(f"\n{'query':<46} {'n_src':>6} {'prev':>7} {'text':>7} {'paired':>7} {'delta':>7}")
for i, e in enumerate(bench_val):
    q = e["query"]
    rt = tables["text"][i]["R@10"]
    rp = tables["paired"][i]["R@10"]
    mark = "  <- negativa" if "-" in q else ""
    print(f"{q:<46} {len(e['ground_truth']):>6} {base_prev[q]:>7.3f} "
          f"{rt:>7.3f} {rp:>7.3f} {rp - rt:>+7.3f}{mark}")
mt, mp = macro(tables["text"], "R@10"), macro(tables["paired"], "R@10")
print(f"{'MACRO':<46} {'':>6} {np.mean(list(base_prev.values())):>7.3f} "
      f"{mt:>7.3f} {mp:>7.3f} {mp - mt:>+7.3f}")

neg_q = [i for i, e in enumerate(bench_val) if "-" in e["query"]]
negdom = [i for i, e in enumerate(bench_val)
          if all(s < 0 for s, _ in parse_query(e["query"]))]
for label, idxs in (("solo query con negazioni", neg_q),
                    ("solo negative-dominanti", negdom)):
    t = np.mean([tables["text"][i]["R@10"] for i in idxs])
    p = np.mean([tables["paired"][i]["R@10"] for i in idxs])
    print(f"{label:<53} {t:>7.3f} {p:>7.3f} {p - t:>+7.3f}")

json.dump({"best_lr": {n: best[n][0] for n in best},
           "mirror_tables": tables,
           "sweep": {f"{n}_lr{lr:g}": runs[(n, lr)][1]
                     for (n, lr) in runs}},
          open(OUT / "fusion_init_ablation_mirror.json", "w"), indent=1)
print("\nsalvato fusion_init_ablation_mirror.json", flush=True)
