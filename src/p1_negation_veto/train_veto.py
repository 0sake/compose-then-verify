# ARCHIVE, not runnable: imports `baselines` and `train_fusion`, development
# modules that were not preserved. Kept as the record of the P1 experiment;
# its results are in output/results/p1_negation_veto/.
"""Ablazione minima proposta 1: DirectionFusion CONGELATO +- veto di semispazio.

Score con veto (solo sulle condizioni negate della query):
  s(x) = cos(q, x) - sum_{a in neg} lambda_a * softplus(kappa_a * (p_a(x) - tau_a))
con p_a(x) = P_a . x - mid_a punteggio calibrato della sonda (medie di classe
del train, scala +-1). Parametri per attributo: lambda_a (init 0: a init il
modello coincide ESATTAMENTE con DirectionFusion, come il gate 3c),
kappa_a = exp(k_a) (init 2), tau_a (init 0). 120 parametri.

Regola dichiarata in anticipo: veto disattivato (lambda_a = 0 fisso) per gli
attributi con AUC della sonda su validation < 0.75.

Protocollo identico a 3b: budget fisso 3000 passi, niente early stopping,
stessa pipeline InfoNCE (negativi misti), sweep lr a 3 punti selezionato sul
mirror di validation; batch identici tra i run (generatori riseedati).
Il regime dei parametri e' scalare (come ScalarFusion, lr 0.02), quindi la
griglia e' {3e-3, 1e-2, 3e-2}.
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_fusion
from baselines import (DEV, V, attr_names, bench_val, embs, eval_entry,
                       evaluate, macro, parse_query, part_arr, KS)
from train_fusion import DirectionFusion, E, SCRATCH, query_fn_from, sample_group

# --- sonde calibrate e maschera AUC --------------------------------------------
pk = torch.load(SCRATCH / "probes_calibrated.pt", map_location=DEV,
                weights_only=False)
P, mid = pk["P"].to(DEV), pk["mid"].to(DEV)

lines = (SCRATCH / "celeba" / "list_attr_celeba.txt").read_text().splitlines()
attrs_all = np.array([[int(x) for x in ln.split()[1:]] for ln in lines[2:] if ln.strip()],
                     dtype=np.int8)
A_val = torch.from_numpy(attrs_all[part_arr == 1].astype(np.float32)).to(DEV)
Vs = V["val"]
SC_val = Vs @ P.T - mid                                   # (n_val, 40) calibrati


def auc(scores, labels):
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64,
                                device=scores.device)
    pos = labels > 0
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


probe_auc = torch.tensor([auc(SC_val[:, c], A_val[:, c]) for c in range(40)],
                         device=DEV)
mask = (probe_auc >= 0.75)
print(f"sonde attive nel veto: {int(mask.sum())}/40 "
      f"(escluse: {[attr_names[c] for c in range(40) if not mask[c]]})")

# --- modello -------------------------------------------------------------------
class Veto(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.lam = nn.Parameter(torch.zeros(40))
        self.logk = nn.Parameter(torch.full((40,), float(np.log(2.0))))
        self.tau = nn.Parameter(torch.zeros(40))
        self.register_buffer("mask", mask.float())

    def penalty(self, sc, neg_cols):
        """sc: (n, 40) punteggi calibrati -> penalita' (n,)."""
        p = torch.zeros(sc.shape[0], device=sc.device)
        for c in neg_cols:
            p = p + self.mask[c] * self.lam[c] * F.softplus(
                self.logk[c].exp() * (sc[:, c] - self.tau[c]))
        return p


fusion = DirectionFusion(a_init=1.75, b_init=1.0).to(DEV)
ck = torch.load(SCRATCH / "fusion_directions_s3000_lr0.0003.pt",
                map_location=DEV, weights_only=False)
fusion.load_state_dict(ck["state"])
fusion.eval()
for p_ in fusion.parameters():
    p_.requires_grad_(False)
qfn = query_fn_from(fusion)


# --- valutazione con veto sul mirror --------------------------------------------
def evaluate_veto(veto, batch=2048):
    table = []
    with torch.no_grad():
        for entry in bench_val:
            conds = parse_query(entry["query"])
            negs = [c for s, c in conds if s < 0]
            pen = veto.penalty(SC_val, negs) if negs else None
            src_idx = np.array([int(k) for k in entry["ground_truth"].keys()])
            gt_lists = [set(v) for v in entry["ground_truth"].values()]
            agg = {f"{m}@{K}": 0.0 for m in "RP" for K in KS}
            for lo in range(0, len(src_idx), batch):
                sl = slice(lo, min(lo + batch, len(src_idx)))
                idx = torch.from_numpy(src_idx[sl]).to(DEV)
                q = qfn(Vs[idx], conds)
                scores = q @ Vs.T
                if pen is not None:
                    scores = scores - pen
                scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
                part = eval_entry(scores, src_idx[sl], gt_lists[sl])
                w = (sl.stop - sl.start) / len(src_idx)
                for k in agg:
                    agg[k] += part[k] * w
            table.append({"query": entry["query"], "n_src": len(src_idx), **agg})
    return table


# --- training ------------------------------------------------------------------
def train_veto(steps, lr, groups=8, n_src=32, log_every=500):
    train_fusion.GEN.manual_seed(0)                       # batch identici tra run
    torch.manual_seed(0)
    veto = Veto(mask).to(DEV)
    opt = torch.optim.AdamW(veto.parameters(), lr=lr, weight_decay=0.0)
    log_tau = nn.Parameter(torch.tensor(float(np.log(0.07)), device=DEV))
    opt.add_param_group({"params": [log_tau], "lr": lr})
    for step in range(1, steps + 1):
        qs, ts, ss, negs_g = [], [], [], []
        while len(qs) < groups:
            g = sample_group(n_src)
            if g is None:
                continue
            cols, signs, srcs, tgts = g
            with torch.no_grad():
                qs.append(fusion(E[srcs], cols, signs))
            ts.append(E[tgts])
            ss.append(E[srcs])
            negs_g.append([c for c, s in zip(cols.tolist(), signs.tolist())
                           if s < 0])
        Q = torch.cat(qs)                                  # (G*S, 512), staccato
        bank = torch.cat(ts + ss)                          # (2*G*S, 512)
        sc_bank = bank @ P.T - mid                         # (2*G*S, 40)
        base = Q @ bank.T                                  # costante
        pen = torch.stack([veto.penalty(sc_bank, ng) for ng in negs_g])  # (G, 2GS)
        pen = torch.repeat_interleave(pen, n_src, dim=0)   # (G*S, 2GS)
        logits = (base - pen) / log_tau.exp()
        labels = torch.arange(len(Q), device=DEV)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}  "
                  f"tau {log_tau.exp().item():.3f}", flush=True)
    return veto


if __name__ == "__main__":
    base_table = evaluate(qfn, bench_val, "val")
    m_base = macro(base_table, "R@10")
    print(f"riferimento congelato — mirror R@10 DirectionFusion: {m_base:.3f}\n")

    results = {}
    for lr in (3e-3, 1e-2, 3e-2):
        print(f"--- veto, 3000 passi, lr {lr} ---")
        veto = train_veto(3000, lr)
        t = evaluate_veto(veto)
        m = macro(t, "R@10")
        results[lr] = (m, t, veto.state_dict())
        print(f"  mirror R@10 con veto: {m:.3f}  (base {m_base:.3f})\n")

    best_lr = max(results, key=lambda k: results[k][0])
    m_best, t_best, sd_best = results[best_lr]
    print(f"=== sweep: {{{', '.join(f'{k}: {v[0]:.3f}' for k, v in results.items())}}} "
          f"-> lr scelto sul mirror: {best_lr}")

    # tabella per-query: base vs veto (query con negazioni evidenziate)
    print(f"\n{'query':44s} {'n_src':>6s} {'base':>7s} {'veto':>7s} {'delta':>7s}")
    for rb, rv, entry in zip(base_table, t_best, bench_val):
        has_neg = any(s < 0 for s, _ in parse_query(entry["query"]))
        negdom = (sum(s < 0 for s, _ in parse_query(entry["query"])) >
                  sum(s > 0 for s, _ in parse_query(entry["query"])))
        tag = " NEG-DOM" if negdom else (" neg" if has_neg else "")
        print(f"{rb['query']:44s} {rb['n_src']:6d} {rb['R@10']:7.3f} "
              f"{rv['R@10']:7.3f} {rv['R@10']-rb['R@10']:+7.3f}{tag}")
    print(f"{'MACRO (13 query)':44s} {'':6s} {m_base:7.3f} {m_best:7.3f} "
          f"{m_best-m_base:+7.3f}")
    for name, cond in (("solo query con negazioni",
                        lambda c: any(s < 0 for s, _ in c)),
                       ("solo negative-dominanti",
                        lambda c: sum(s < 0 for s, _ in c) > sum(s > 0 for s, _ in c))):
        sel = [(rb["R@10"], rv["R@10"]) for rb, rv, e in
               zip(base_table, t_best, bench_val) if cond(parse_query(e["query"]))]
        b, v = (np.mean([x[0] for x in sel]), np.mean([x[1] for x in sel]))
        print(f"{name:44s} {'':6s} {b:7.3f} {v:7.3f} {v-b:+7.3f}")

    # parametri appresi sugli attributi negati del benchmark
    veto_best = Veto(mask).to(DEV)
    veto_best.load_state_dict(sd_best)
    neg_cols = sorted({c for e in bench_val for s, c in parse_query(e["query"])
                       if s < 0})
    print(f"\nparametri appresi (attributi negati del benchmark):")
    print(f"{'attributo':20s} {'lambda':>8s} {'kappa':>8s} {'tau':>8s}")
    for c in neg_cols:
        print(f"{attr_names[c]:20s} {veto_best.lam[c].item():8.3f} "
              f"{veto_best.logk[c].exp().item():8.3f} {veto_best.tau[c].item():8.3f}")

    torch.save({"state": sd_best, "lr": best_lr, "mirror_R10": m_best,
                "mask": mask.cpu()}, SCRATCH / f"veto_s3000_lr{best_lr}.pt")
    json.dump({"base": base_table, "veto": t_best, "lr": best_lr},
              open(SCRATCH / "veto_ablation_mirror.json", "w"), indent=1)
    print(f"\nsalvati veto_s3000_lr{best_lr}.pt e veto_ablation_mirror.json")
