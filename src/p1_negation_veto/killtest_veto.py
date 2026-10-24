# ARCHIVE, not runnable: imports `baselines` and `train_fusion`, development
# modules that were not preserved. Kept as the record of the P1 experiment;
# its results are in output/results/p1_negation_veto/.
"""Kill-test della proposta 1 (veto di semispazio sulla negazione).

Diagnosi prescritte PRIMA di addestrare:
  (0) riproduzione: DirectionFusion (checkpoint lr 3e-4) sul mirror deve dare
      R@10 ~ 0.296, altrimenti lo stack ricostruito non e' affidabile;
  (1) frazione dei top-10 di DirectionFusion che VIOLANO l'attributo negato
      (label di validation) — se < 20% sulle query negative-dominanti, il veto
      non ha nulla da rimuovere e l'idea muore qui;
  (2) AUC su validation delle sonde calibrate (medie di classe del train) per
      gli attributi negati — regola dichiarata in anticipo: lambda_a = 0 se
      AUC < 0.75;
  (3) veto-ORACOLO (labels vere al posto della sonda): tetto superiore del
      guadagno ottenibile da qualunque veto, a costo di training zero.
"""
import numpy as np
import torch

from baselines import (DEV, V, attr_names, bench_val, embs, evaluate, macro,
                       parse_query, part_arr, KS)
from train_fusion import DirectionFusion, query_fn_from, SCRATCH

# --- label di validation (stesso ordine dello split V["val"]) -----------------
lines = (SCRATCH / "celeba" / "list_attr_celeba.txt").read_text().splitlines()
attrs_all = np.array([[int(x) for x in ln.split()[1:]] for ln in lines[2:] if ln.strip()],
                     dtype=np.int8)
A_val = torch.from_numpy(attrs_all[part_arr == 1].astype(np.float32)).to(DEV)

# --- (0) riproduzione del modello finale sul mirror ----------------------------
ck = torch.load(SCRATCH / "fusion_directions_s3000_lr0.0003.pt",
                map_location=DEV, weights_only=False)
model = DirectionFusion(a_init=1.75, b_init=1.0).to(DEV)
model.load_state_dict(ck["state"])
model.eval()
qfn = query_fn_from(model)

t = evaluate(qfn, bench_val, "val")
m = macro(t, "R@10")
print(f"(0) riproduzione — mirror R@10 DirectionFusion: {m:.3f} "
      f"(checkpoint dichiarava {ck['mirror_R10']:.3f})")
assert abs(m - ck["mirror_R10"]) < 1e-6, "stack non riproducibile, fermarsi"

# --- (1) frazione di violatori nei top-10 --------------------------------------
Vs = V["val"]


def topk_indices(entry, k=10, batch=2048):
    conds = parse_query(entry["query"])
    src_idx = np.array([int(s) for s in entry["ground_truth"].keys()])
    outs = []
    for lo in range(0, len(src_idx), batch):
        idx = torch.from_numpy(src_idx[lo:lo + batch]).to(DEV)
        q = qfn(Vs[idx], conds)
        scores = q @ Vs.T
        scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
        outs.append(scores.topk(k, dim=1).indices)
    return src_idx, torch.cat(outs), conds


print(f"\n(1) violatori dell'attributo negato nei top-10 (mirror, validation):")
print(f"{'query':44s} {'n_src':>6s} {'attr negato':>16s} {'%viol top-10':>12s}")
frac_by_query = {}
for entry in bench_val:
    conds = parse_query(entry["query"])
    negs = [c for s, c in conds if s < 0]
    if not negs:
        continue
    src_idx, top, _ = topk_indices(entry)
    for c in negs:
        # violatore: possiede l'attributo vietato (label +1)
        viol = (A_val[top.reshape(-1), c] > 0).float().mean().item()
        n_pos = sum(1 for s, _ in conds if s > 0)
        frac_by_query.setdefault(entry["query"], []).append(viol)
        dom = "NEG-DOM" if len(negs) > n_pos else ""
        print(f"{entry['query']:44s} {len(src_idx):6d} {attr_names[c]:>16s} "
              f"{100*viol:11.1f}% {dom}")

negdom = []
for entry in bench_val:
    conds = parse_query(entry["query"])
    negs = [c for s, c in conds if s < 0]
    n_pos = len(conds) - len(negs)
    if negs and len(negs) > n_pos:
        negdom += frac_by_query[entry["query"]]
print(f"\nmedia %violatori sulle query negative-dominanti: {100*np.mean(negdom):.1f}%"
      f"   [soglia kill-test: < 20% = idea morta]")

# --- (2) AUC delle sonde calibrate su validation --------------------------------
E_tr = embs[torch.from_numpy(part_arr == 0)].to(DEV)
E_tr = E_tr / E_tr.norm(dim=1, keepdim=True)
A_tr = torch.from_numpy(attrs_all[part_arr == 0].astype(np.float32)).to(DEV)


def auc(scores, labels):
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64,
                                device=scores.device)
    pos = labels > 0
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


probes, mids = [], []
for i in range(40):
    pos = A_tr[:, i] > 0
    d = E_tr[pos].mean(0) - E_tr[~pos].mean(0)
    d = d / d.norm()
    m_pos, m_neg = (E_tr[pos] @ d).mean(), (E_tr[~pos] @ d).mean()
    scale = 2.0 / (m_pos - m_neg)
    probes.append(d * scale)
    mids.append(((m_pos + m_neg) / 2 * scale).item())
P = torch.stack(probes)
mid = torch.tensor(mids, device=DEV)

neg_cols = sorted({c for e in bench_val for s, c in parse_query(e["query"]) if s < 0})
print(f"\n(2) AUC sonde calibrate (train->validation) sugli attributi negati "
      f"[regola: lambda=0 se < 0.75]:")
for c in neg_cols:
    a = auc(Vs @ P[c] - mid[c], A_val[:, c])
    flag = "  <-- ESCLUSO dal veto" if a < 0.75 else ""
    print(f"  {attr_names[c]:20s} AUC {a:.3f}{flag}")

torch.save({"P": P.cpu(), "mid": mid.cpu()}, SCRATCH / "probes_calibrated.pt")

# --- (3) veto oracolo: tetto del guadagno ---------------------------------------
def eval_with_oracle_veto(entry, k=10, batch=2048):
    """Scarta dal ranking le immagini con label violata (veto perfetto)."""
    conds = parse_query(entry["query"])
    negs = [c for s, c in conds if s < 0]
    src_idx = np.array([int(s) for s in entry["ground_truth"].keys()])
    gt_lists = [set(v) for v in entry["ground_truth"].values()]
    viol_mask = torch.zeros(len(Vs), dtype=torch.bool, device=DEV)
    for c in negs:
        viol_mask |= A_val[:, c] > 0
    hits_at = {K: 0.0 for K in KS}
    for lo in range(0, len(src_idx), batch):
        sl = slice(lo, min(lo + batch, len(src_idx)))
        idx = torch.from_numpy(src_idx[sl]).to(DEV)
        q = qfn(Vs[idx], conds)
        scores = q @ Vs.T
        scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
        scores[:, viol_mask] = -torch.inf
        top = scores.topk(max(KS), dim=1).indices.cpu().numpy()
        for i, gt in enumerate(gt_lists[sl]):
            for K in KS:
                hits_at[K] += len(set(top[i, :K].tolist()) & gt) > 0
    return {f"R@{K}": hits_at[K] / len(src_idx) for K in KS}


print(f"\n(3) veto ORACOLO (labels vere) vs DirectionFusion, query con negazioni:")
print(f"{'query':44s} {'R@10 base':>10s} {'R@10 oracolo':>13s} {'delta':>7s}")
base_all, oracle_all = [], []
rows_neg = [r for r in t]
for entry, row in zip(bench_val, t):
    conds = parse_query(entry["query"])
    if not any(s < 0 for s, _ in conds):
        continue
    o = eval_with_oracle_veto(entry)
    print(f"{entry['query']:44s} {row['R@10']:10.3f} {o['R@10']:13.3f} "
          f"{o['R@10']-row['R@10']:+7.3f}")
    base_all.append(row["R@10"])
    oracle_all.append(o["R@10"])
print(f"{'MEDIA (query con negazioni)':44s} {np.mean(base_all):10.3f} "
      f"{np.mean(oracle_all):13.3f} {np.mean(oracle_all)-np.mean(base_all):+7.3f}")

# macro complessivo mirror se il veto oracolo si applica solo dove ci sono negazioni
mix = []
for entry, row in zip(bench_val, t):
    conds = parse_query(entry["query"])
    if any(s < 0 for s, _ in conds):
        mix.append(eval_with_oracle_veto(entry)["R@10"])
    else:
        mix.append(row["R@10"])
print(f"\nmirror R@10 macro: base {m:.3f} -> con veto oracolo {np.mean(mix):.3f} "
      f"(tetto teorico dell'idea)")
