"""Proposta 4, ablazione minima (zero training): additiva vs rotazionale sul mirror.

Ogni (attributo, segno) e' una rotazione di piano invece di una traslazione:
    q = exp( sum_c theta_c A_c ) v_ref,   A_c = d'_c u_c^T - u_c d'_c^T
con (u_c, d'_c) base ortonormale del piano dell'attributo. theta_c per
"pari lunghezza d'arco": per ogni sorgente, theta risolto perche' la
rotazione produca lo stesso arco dell'edit additivo B3 applicato da solo
(delta = alpha*d_text o -beta*d_text, (alpha,beta)=(1.75,1.0) dalla griglia).
Se la massa in piano di v e' troppo piccola, theta e' cappato a pi.

Varianti di piano (entrambe training-free, dichiarate in anticipo):
  text-plane : u = prompt negativo, d = prompt positivo dell'attributo
  img-plane  : u = media train classe-0, d = media train classe-1
Unico iperparametro: scala globale s su theta, sweep {0.5,0.75,1,1.25,1.5}
sul mirror (l'additiva ha avuto la griglia 10x10 su alpha,beta).
Riferimento additivo B3 text ricalcolato nello stesso path di valutazione.
Il test non viene toccato.
"""
import json
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
SPLIT_IDX = {s: np.nonzero(PART == c)[0] for s, c in
             (("train", 0), ("val", 1), ("test", 2))}
A_SPLIT = {s: ATTRS[SPLIT_IDX[s]] for s in SPLIT_IDX}

d = load_features()
fn2idx = {f: i for i, f in enumerate(d["filenames"])}
order = [fn2idx[f] for f in filenames]
E = F.normalize(d["embeddings"][order], dim=-1)
V_VAL = E[SPLIT_IDX["val"]].to(DEV)
X_TR = E[SPLIT_IDX["train"]].to(DEV)
Y_TR = torch.from_numpy(A_SPLIT["train"] == 1).to(DEV)


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

# --- direzioni testuali B3 (riferimento additivo) e ancore dei piani -------
DIRS = torch.load(P2 / "directions_all_sets.pt", map_location=DEV,
                  weights_only=False)
B3RES = json.load(open(results_dir("p2_paired_directions") / "b3_direction_ablation_mirror.json"))
ALPHA, BETA = B3RES["text"]["alpha"], B3RES["text"]["beta"]
D_text = DIRS["D_text"].to(DEV)
print(f"additiva B3 text: (alpha, beta) = ({ALPHA}, {BETA})", flush=True)

# ancore testuali: prompt positivo/negativo per attributo (stessi della pipeline)
from transformers import CLIPModel, CLIPProcessor

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
for p in clip.parameters():
    p.requires_grad_(False)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

BASE = "a photo of a person"
PROMPTS = {
    "5_o_Clock_Shadow": ("a photo of a person with stubble on their face", "a photo of a person with a clean-shaven face"),
    "Arched_Eyebrows": ("a photo of a person with arched eyebrows", "a photo of a person with straight eyebrows"),
    "Attractive": ("a photo of an attractive person", "a photo of an unattractive person"),
    "Bags_Under_Eyes": ("a photo of a person with bags under the eyes", "a photo of a person with smooth skin under the eyes"),
    "Bald": ("a photo of a bald person", "a photo of a person with a full head of hair"),
    "Bangs": ("a photo of a person with bangs covering the forehead", "a photo of a person with the forehead visible"),
    "Big_Lips": ("a photo of a person with big full lips", "a photo of a person with thin lips"),
    "Big_Nose": ("a photo of a person with a big nose", "a photo of a person with a small nose"),
    "Black_Hair": ("a photo of a person with black hair", BASE),
    "Blond_Hair": ("a photo of a person with blond hair", BASE),
    "Blurry": ("a blurry photo of a person", "a sharp photo of a person"),
    "Brown_Hair": ("a photo of a person with brown hair", BASE),
    "Bushy_Eyebrows": ("a photo of a person with bushy eyebrows", "a photo of a person with thin eyebrows"),
    "Chubby": ("a photo of a chubby person with a round face", "a photo of a slim person with a thin face"),
    "Double_Chin": ("a photo of a person with a double chin", "a photo of a person with a defined jawline"),
    "Eyeglasses": ("a photo of a person wearing eyeglasses", BASE),
    "Goatee": ("a photo of a person with a goatee beard", "a photo of a clean-shaven person"),
    "Gray_Hair": ("a photo of a person with gray hair", BASE),
    "Heavy_Makeup": ("a photo of a person wearing heavy makeup", "a photo of a person with a natural face without makeup"),
    "High_Cheekbones": ("a photo of a person with high cheekbones", "a photo of a person with flat cheekbones"),
    "Male": ("a photo of a man", "a photo of a woman"),
    "Mouth_Slightly_Open": ("a photo of a person with the mouth slightly open", "a photo of a person with the mouth closed"),
    "Mustache": ("a photo of a person with a mustache", "a photo of a clean-shaven person"),
    "Narrow_Eyes": ("a photo of a person with narrow eyes", "a photo of a person with wide open eyes"),
    "No_Beard": ("a photo of a clean-shaven person with no beard", "a photo of a person with a beard"),
    "Oval_Face": ("a photo of a person with an oval face", BASE),
    "Pale_Skin": ("a photo of a person with pale skin", "a photo of a person with tanned skin"),
    "Pointy_Nose": ("a photo of a person with a pointy nose", "a photo of a person with a round nose"),
    "Receding_Hairline": ("a photo of a person with a receding hairline", "a photo of a person with a full hairline"),
    "Rosy_Cheeks": ("a photo of a person with rosy cheeks", BASE),
    "Sideburns": ("a photo of a person with sideburns", "a photo of a clean-shaven person"),
    "Smiling": ("a photo of a smiling person", "a photo of a person with a neutral serious expression"),
    "Straight_Hair": ("a photo of a person with straight hair", "a photo of a person with curly hair"),
    "Wavy_Hair": ("a photo of a person with wavy hair", "a photo of a person with straight hair"),
    "Wearing_Earrings": ("a photo of a person wearing earrings", BASE),
    "Wearing_Hat": ("a photo of a person wearing a hat", "a photo of a bareheaded person"),
    "Wearing_Lipstick": ("a photo of a person wearing lipstick", "a photo of a person with bare lips"),
    "Wearing_Necklace": ("a photo of a person wearing a necklace", BASE),
    "Wearing_Necktie": ("a photo of a person wearing a necktie", BASE),
    "Young": ("a photo of a young person", "a photo of an old person"),
}


def text_features(texts):
    tok = processor(text=texts, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = clip.get_text_features(input_ids=tok["input_ids"],
                                     attention_mask=tok["attention_mask"])
    return out if torch.is_tensor(out) else out.pooler_output


Tp = F.normalize(text_features([PROMPTS[a][0] for a in ATTR_NAMES]), dim=1)
Tn = F.normalize(text_features([PROMPTS[a][1] for a in ATTR_NAMES]), dim=1)

Mu1 = F.normalize(torch.stack([X_TR[Y_TR[:, a]].mean(0) for a in range(40)]), dim=1)
Mu0 = F.normalize(torch.stack([X_TR[~Y_TR[:, a]].mean(0) for a in range(40)]), dim=1)


def make_planes(U_anchor, D_anchor):
    """Base ortonormale (u, d') per attributo. Ritorna (40,512) e (40,512)."""
    U = F.normalize(U_anchor, dim=1)
    Dp = D_anchor - (D_anchor * U).sum(1, keepdim=True) * U
    return U, F.normalize(Dp, dim=1)


mu_hat = F.normalize(X_TR.mean(0), dim=0)
PLANES = {"text-plane": make_planes(Tn, Tp),
          "img-plane": make_planes(Mu0, Mu1),
          "paired-plane": make_planes(mu_hat[None].expand(40, -1).clone(),
                                      DIRS["D_pair"].to(DEV))}
for name, (U, Dp) in PLANES.items():
    inmass = ((V_VAL @ U.T) ** 2 + (V_VAL @ Dp.T) ** 2).mean(0).sqrt()
    print(f"{name}: massa in piano media della galleria "
          f"min {inmass.min():.3f} media {inmass.mean():.3f} "
          f"max {inmass.max():.3f}", flush=True)

# --- valutazione -----------------------------------------------------------
KS = (1, 5, 10)


def eval_entry(scores, gt_lists):
    topk = scores.topk(max(KS), dim=1).indices.cpu().numpy()
    out = {}
    for K in KS:
        r = 0.0
        for i, gt in enumerate(gt_lists):
            r += len(set(topk[i, :K].tolist()) & gt) > 0
        out[f"R@{K}"] = r / len(gt_lists)
    return out


def evaluate(query_fn, batch=2048):
    table = []
    for entry in bench_val:
        conds = parse_query(entry["query"])
        src_idx = np.array([int(kk) for kk in entry["ground_truth"]])
        gt_lists = [set(v) for v in entry["ground_truth"].values()]
        agg = {f"R@{K}": 0.0 for K in KS}
        for lo in range(0, len(src_idx), batch):
            sl = slice(lo, min(lo + batch, len(src_idx)))
            idx = torch.from_numpy(src_idx[sl]).to(DEV)
            q = query_fn(V_VAL[idx], conds)
            scores = q @ V_VAL.T
            scores[torch.arange(len(idx), device=DEV), idx] = -torch.inf
            part = eval_entry(scores, gt_lists[sl])
            w = (sl.stop - sl.start) / len(src_idx)
            for kk in agg:
                agg[kk] += part[kk] * w
        table.append({"query": entry["query"], "n_src": len(src_idx), **agg})
    return table


def macro(table, key):
    return float(np.mean([row[key] for row in table]))


def additive_b3(v, conds):
    q = v.clone()
    for s, c in conds:
        q += (ALPHA if s > 0 else -BETA) * D_text[c]
    return q


def make_rotational(U, Dp, scale):
    def rot(v, conds):
        n = v.shape[0]
        # arco dell'edit additivo alone, per sorgente e condizione
        thetas = []
        for s, c in conds:
            delta = (ALPHA if s > 0 else -BETA) * D_text[c]
            q1 = F.normalize(v + delta, dim=-1)
            cosphi = (v * q1).sum(-1).clamp(-1, 1)
            p2 = (v @ U[c]) ** 2 + (v @ Dp[c]) ** 2
            costheta = (1 - (1 - cosphi) / p2.clamp_min(1e-8)).clamp(-1, 1)
            th = scale * torch.arccos(costheta) * (1 if s > 0 else -1)
            thetas.append(th)
        # base congiunta dei piani della query
        P = torch.cat([torch.stack([U[c], Dp[c]]) for _, c in conds])
        Usvd, S, _ = torch.linalg.svd(P.T, full_matrices=False)
        B = Usvd[:, S > 1e-6]                       # 512 x r
        r = B.shape[1]
        G = torch.zeros(n, r, r, device=DEV)
        for th, (s, c) in zip(thetas, conds):
            u_s, d_s = B.T @ U[c], B.T @ Dp[c]
            K = torch.outer(d_s, u_s) - torch.outer(u_s, d_s)
            G += th[:, None, None] * K[None]
        R = torch.matrix_exp(G)
        vB = v @ B                                  # n x r
        q = v + torch.einsum("nij,nj->ni", R - torch.eye(r, device=DEV), vB) @ B.T
        return q
    return rot


# riferimento additivo (deve riprodurre B3 text 0.203)
add_table = evaluate(additive_b3)
print(f"\nadditiva B3 text, mirror macro R@10 = {macro(add_table, 'R@10'):.3f} "
      f"(atteso 0.203)", flush=True)

SCALES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
all_results = {}
for name, (U, Dp) in PLANES.items():
    print(f"\n[{name}] sweep scala s sul mirror:")
    tabs = {}
    for s in SCALES:
        t = evaluate(make_rotational(U, Dp, s))
        tabs[s] = t
        print(f"  s = {s:<5} macro R@10 = {macro(t, 'R@10'):.3f}", flush=True)
    s_best = max(SCALES, key=lambda s: macro(tabs[s], "R@10"))
    all_results[name] = {"s_best": s_best, "table": tabs[s_best],
                         "sweep": {f"{s:g}": macro(tabs[s], "R@10")
                                   for s in SCALES}}
    print(f"  s* = {s_best}")

# --- confronto per query e per k --------------------------------------------
print(f"\n{'query':<46} {'k':>2} {'addit':>7} "
      + " ".join(f"{n:>10}" for n in PLANES))
for i, e in enumerate(bench_val):
    k = len(parse_query(e["query"]))
    vals = " ".join(f"{all_results[n]['table'][i]['R@10']:>10.3f}"
                    for n in PLANES)
    print(f"{e['query']:<46} {k:>2} {add_table[i]['R@10']:>7.3f} {vals}")
print(f"{'MACRO':<46} {'':>2} {macro(add_table, 'R@10'):>7.3f} "
      + " ".join(f"{macro(all_results[n]['table'], 'R@10'):>10.3f}"
                 for n in PLANES))
per_k = {}
for i, e in enumerate(bench_val):
    per_k.setdefault(len(parse_query(e["query"])), []).append(i)
for k, idxs in sorted(per_k.items()):
    a = np.mean([add_table[i]["R@10"] for i in idxs])
    vals = " ".join(
        f"{np.mean([all_results[n]['table'][i]['R@10'] for i in idxs]):>10.3f}"
        for n in PLANES)
    print(f"{f'k={k} (n={len(idxs)})':<46} {'':>2} {a:>7.3f} {vals}")

json.dump({"additive": add_table,
           **{n: {"s_best": r["s_best"], "sweep": r["sweep"],
                  "table": r["table"]} for n, r in all_results.items()}},
          open(OUT / "rotation_ablation_mirror.json", "w"), indent=1)
print("\nsalvato rotation_ablation_mirror.json", flush=True)
