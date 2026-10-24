"""Ablazione minima Proposta 2: B3 con tre set di direzioni sul mirror di validation.

Set confrontati, stessa griglia (alpha, beta) della pipeline (0.25..2.50, passo 0.25):
  - text    : direzioni testuali con selezione per-attributo su AUC val (pipeline attuale)
  - B5      : medie di classe marginali  mean(x|a=1) - mean(x|a=0)  sul train
  - paired  : coppie appaiate h=0 (differiscono SOLO su a), esatte via somme di gruppo

Protocollo: tutte le scelte sul mirror (validation). Il test non viene toccato.
Il mirror rigenerato viene verificato contro i n_src di veto_ablation_mirror.json.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.paths import CELEBA, EVAL_JSON, VETO_JSON, CHECKPOINTS, load_features, results_dir
OUT = results_dir("p2_paired_directions")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# --- attributi e split (identico alla pipeline) ----------------------------
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
sizes = {p: int((PART == p).sum()) for p in (0, 1, 2)}
assert sizes == {0: 162770, 1: 19867, 2: 19962}, "split non canonico"
SPLIT_IDX = {s: np.nonzero(PART == c)[0] for s, c in
             (("train", 0), ("val", 1), ("test", 2))}
A_SPLIT = {s: ATTRS[SPLIT_IDX[s]] for s in SPLIT_IDX}

# --- embeddings (deeplearning: stessa estrazione CLIP, gia' normalizzati) --
d = load_features()
fn2idx = {f: i for i, f in enumerate(d["filenames"])}
order = [fn2idx[f] for f in filenames]
E = F.normalize(d["embeddings"][order], dim=-1)
V = {s: E[SPLIT_IDX[s]].to(DEV) for s in SPLIT_IDX}
print(f"embeddings: {E.shape}, split su {DEV}")


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

# --- mirror di validation, con verifica contro l'esperimento veto ----------
bench_val, seen = [], set()
for entry in BENCH:
    if entry["query"] in seen:
        continue
    seen.add(entry["query"])
    targets, viol = rebuild_ground_truth(entry["query"], A_SPLIT["val"])
    gt = {str(j): t for j, t in targets.items() if viol[j]}
    bench_val.append({"query": entry["query"], "ground_truth": gt})
print(f"mirror: {len(bench_val)} query uniche")

veto_ref = {row["query"]: row["n_src"] for row in json.load(open(VETO_JSON))["base"]}
for e in bench_val:
    assert len(e["ground_truth"]) == veto_ref[e["query"]], \
        f"mirror diverso dal precedente su {e['query']}"
print("mirror identico a quello dell'esperimento veto (n_src per query)")

# --- valutazione batched (identica alla pipeline) ---------------------------
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


# --- set 1: direzioni testuali con selezione per-attributo (pipeline) ------
import os
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
assert set(PROMPTS) == set(ATTR_NAMES)


def text_features(texts):
    tok = processor(text=texts, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = clip.get_text_features(input_ids=tok["input_ids"],
                                     attention_mask=tok["attention_mask"])
    return out if torch.is_tensor(out) else out.pooler_output


def auc(scores, labels):
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    ranks = scores.argsort().argsort().astype(np.float64) + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


Tp = F.normalize(text_features([PROMPTS[a][0] for a in ATTR_NAMES]), dim=1)
Tn = F.normalize(text_features([PROMPTS[a][1] for a in ATTR_NAMES]), dim=1)
tb = F.normalize(text_features([BASE])[0], dim=0)
CAND = {"raw": Tp,
        "delta-ant": F.normalize(Tp - Tn, dim=1),
        "delta-base": F.normalize(Tp - tb[None, :], dim=1)}
y_all = A_SPLIT["val"]
scores_all = {k: (V["val"] @ M.T).cpu().numpy() for k, M in CAND.items()}
choice = {}
for i, a in enumerate(ATTR_NAMES):
    aucs = {k: auc(scores_all[k][:, i], y_all[:, i]) for k in CAND}
    choice[a] = max(aucs, key=aucs.get)
D_text = torch.stack([CAND[choice[a]][i] for i, a in enumerate(ATTR_NAMES)]).to(DEV)
print("direzioni testuali selezionate:",
      Counter(choice.values()).most_common())

# --- set 2 e 3: direzioni immagine (B5 marginali e appaiate h=0) -----------
Xt = E[SPLIT_IDX["train"]]
At = torch.tensor(A_SPLIT["train"] == 1)

mu1 = torch.stack([Xt[At[:, a]].mean(0) for a in range(40)])
mu0 = torch.stack([Xt[~At[:, a]].mean(0) for a in range(40)])
D_b5 = F.normalize(mu1 - mu0, dim=-1).to(DEV)

weights = (1 << np.arange(40)).astype(object)
keys = (At.numpy().astype(object) @ weights)
gsum = defaultdict(lambda: torch.zeros(512))
gcnt = defaultdict(int)
for i, k in enumerate(keys):
    gsum[k] += Xt[i]
    gcnt[k] += 1
D_pair = torch.zeros(40, 512)
n_pairs = np.zeros(40, dtype=np.int64)
for a in range(40):
    flip = 1 << a
    num = torch.zeros(512)
    tot = 0
    for k, c0 in gcnt.items():
        if k & flip:
            continue
        c1 = gcnt.get(k ^ flip, 0)
        if c1 == 0:
            continue
        num += c0 * gsum[k ^ flip] - c1 * gsum[k]
        tot += c0 * c1
    n_pairs[a] = tot
    if tot > 0:
        D_pair[a] = num / tot
D_pair = F.normalize(D_pair, dim=-1).to(DEV)
print(f"direzioni appaiate h=0: coppie min {n_pairs.min()} "
      f"({ATTR_NAMES[n_pairs.argmin()]}), max {n_pairs.max()}")

# --- griglia B3 per ogni set -------------------------------------------------
GRID = [round(x, 2) for x in np.arange(0.25, 2.51, 0.25)]


def make_b3(D, a, b):
    def b3(v_ref, conds):
        q = v_ref.clone()
        for s, c in conds:
            q += (a if s > 0 else -b) * D[c]
        return q
    return b3


results = {}
for name, D in (("text", D_text), ("b5", D_b5), ("paired", D_pair)):
    grid = {}
    for a in GRID:
        for b in GRID:
            t = evaluate(make_b3(D, a, b), bench_val, "val")
            grid[(a, b)] = [row["R@10"] for row in t]
    (a_s, b_s) = max(grid, key=lambda ab: np.mean(grid[ab]))
    best_table = evaluate(make_b3(D, a_s, b_s), bench_val, "val")
    results[name] = {"alpha": a_s, "beta": b_s,
                     "macro_R10": macro(best_table, "R@10"),
                     "macro_R5": macro(best_table, "R@5"),
                     "macro_R1": macro(best_table, "R@1"),
                     "table": best_table}
    print(f"{name:7s} (alpha*, beta*) = ({a_s}, {b_s})  "
          f"macro R@10 mirror = {results[name]['macro_R10']:.3f}")

# --- tabella per-query -------------------------------------------------------
print(f"\n{'query':<46} {'n_src':>6} {'text':>7} {'B5':>7} {'paired':>7}")
for i, e in enumerate(bench_val):
    r = {n: results[n]["table"][i]["R@10"] for n in results}
    neg = "-" in e["query"]
    mark = "  <- negativa" if neg else ""
    print(f"{e['query']:<46} {len(e['ground_truth']):>6} "
          f"{r['text']:>7.3f} {r['b5']:>7.3f} {r['paired']:>7.3f}{mark}")

json.dump({n: {k: v for k, v in r.items()} for n, r in results.items()},
          open(OUT / "b3_direction_ablation_mirror.json", "w"), indent=1)
torch.save({"D_text": D_text.cpu(), "D_b5": D_b5.cpu(), "D_pair": D_pair.cpu(),
            "choice": choice, "n_pairs": n_pairs, "ATTR_NAMES": ATTR_NAMES},
           CHECKPOINTS / "directions_all_sets.pt")
print("\nsalvati b3_direction_ablation_mirror.json e directions_all_sets.pt")
