"""Genera notebooks/exploration/probe_ladder.ipynb (S2: sonde non lineari / pool-conditional).

Il notebook e' costruito da questo script e poi eseguito con nbconvert
(cwd = notebooks/exploration).
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "exploration" / "probe_ladder.ipynb"

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": src})


md(r"""# S2 — Sonde non lineari / pool-conditional per il re-rank a predicato completo

**Contesto** (documents/lab_log.md, sezione "Re-rank a predicato completo", 19 lug): il re-rank
dei top-200 di DirectionFusion con `S = Σ log P(vincolo) + log P(#mismatch resto ≤ 2)`
su 40 sonde logistiche porta il mirror a **0.425** (base 0.296). Gap residuo decomposto:
lato sorgente +0.138 (dp_gtref 0.563), **lato candidati +0.277** (oracle_rest 0.840).
La diagnosi I1 dice dove sta il lato candidati: l'AUC delle sonde lineari **crolla sul
pool** dei top-200 (Heavy_Makeup 0.967→0.761) — perdono discriminazione esattamente
dove il re-rank le usa.

**Meccanismo S2**: teste MLP per-attributo (512→64→1) al posto delle lineari, con
training pesato sui pool difficili (top-200 di query DirectionFusion costruite sul
train), per recuperare AUC nella regione di deployment. Temperatura per-attributo su
un held-out di train (10%) perché la coda di Poisson-binomiale consuma probabilità,
non ranking: un MLP sovraconfidente può migliorare l'AUC e peggiorare la DP.

**Scala (un cambiamento per gradino), tutto seed 0, budget fissi, solo mirror:**

| gradino | cosa cambia |
|---|---|
| `lin_full` | ancora: sonde lineari full-train, ricetta S1 (riproduzione 0.425) |
| `lin90_temp` | stesse lineari su train-90 + temperatura su held-10 (confondente calibrazione) |
| `mlp_global` | MLP 512→64→1, training globale su train-90 + temperatura |
| `mlp_pool` | come sopra, meta' di ogni batch campionata ∝ √(conteggio pool) |

**Predizioni e regole dichiarate prima del run:**
- P1 (meccanismo): pool-AUC media sui 6 attributi globalmente piu' deboli del resto:
  `mlp_global` ≥ `lin_full` + 0.02. Se fallisce E il macro non si muove → capacita' morta
  (tetto delle feature ViT-B/32), strategia chiusa.
- P2 (macro): miglior gradino nel range **0.46–0.52** mirror. Adozione: ≥ **0.435**
  (ancora 0.425 + 0.010, regola invariata del protocollo).
- P3 (ordine): `mlp_pool` ≥ `mlp_global` ≥ `lin_full`.
- Se `lin90_temp` sale da sola vicino ai gradini MLP → il guadagno e' calibrazione,
  non capacita' (interpretazione, non kill).
- Righe `dp_gtref` (profilo sorgente da label vere) per i gradini chiave: misurano il
  progresso del SOLO lato candidati verso il tetto 0.840.

Test NON toccato. Path e dati: variabili d'ambiente di `src/paths.py`.""")

code(r"""# --- config, seed, dati -----------------------------------------------------
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from src.paths import CELEBA, EVAL_JSON, CHECKPOINTS, load_features, results_dir

CKPT = CHECKPOINTS / "fusion_text_s3000_lr0.0003.pt"
OUT  = results_dir("mlp_probes")

DEV = "cuda"
torch.manual_seed(0)
T0 = time.time()
print(f"GPU: {torch.cuda.get_device_name(0)}")""")

code(r"""# --- dati (identico alla pipeline S1) ---------------------------------------
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
N_TRAIN = E_TRAIN.shape[0]
A_TRAIN = torch.from_numpy(A_SPLIT["train"].astype(np.float32)).to(DEV)
A_VAL = torch.from_numpy(A_SPLIT["val"].astype(np.float32)).to(DEV)
Y_ALL = (A_TRAIN > 0).float()
print(f"train {N_TRAIN}, val {N_GAL}  ({time.time() - T0:.0f}s)")""")

code(r"""# --- benchmark mirror + harness di valutazione ------------------------------
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
    BENCH = json.load(f)
bench_val, seen = [], set()
for entry in BENCH:
    if entry["query"] in seen:
        continue
    seen.add(entry["query"])
    targets, viol = rebuild_ground_truth(entry["query"], A_SPLIT["val"])
    gt = {str(j): t for j, t in targets.items() if viol[j]}
    bench_val.append({"query": entry["query"], "ground_truth": gt})

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


def macro(table, key="R@10"):
    return float(np.mean([row[key] for row in table]))


print(f"mirror: {len(bench_val)} query  ({time.time() - T0:.0f}s)")""")

code(r"""# --- DirectionFusion congelato + cache dei pool top-200 (ancora: base 0.296) -
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
E_pos, E_neg = ck["state"]["E_pos"], ck["state"]["E_neg"]


def qfn(v_ref, conds):
    delta = torch.stack([E_pos[c] if s > 0 else E_neg[c]
                         for s, c in conds]).sum(0)
    return F.normalize(v_ref + delta, dim=-1)


M = 200
POOLS = []
base_table = []
for entry in bench_val:
    conds = parse_query(entry["query"])
    ccols = [c for _, c in conds]
    src_idx = np.array([int(k) for k in entry["ground_truth"]])
    gt_lists = [set(v) for v in entry["ground_truth"].values()]
    tops, coss = [], []
    agg = {f"{m_}@{K}": 0.0 for m_ in "RP" for K in KS}
    for lo in range(0, len(src_idx), 2048):
        sl = slice(lo, min(lo + 2048, len(src_idx)))
        idx = torch.from_numpy(src_idx[sl]).to(DEV)
        q = qfn(V_VAL[idx], conds)
        sc = q @ V_VAL.T
        sc[torch.arange(len(idx), device=DEV), idx] = -torch.inf
        top = sc.topk(M, dim=1).indices
        tops.append(top)
        coss.append(sc.gather(1, top))
        part = eval_entry(sc, gt_lists[sl])
        w = (sl.stop - sl.start) / len(src_idx)
        for k in agg:
            agg[k] += part[k] * w
    POOLS.append({"query": entry["query"], "conds": conds, "ccols": ccols,
                  "rest": [c for c in range(40) if c not in ccols],
                  "src": torch.from_numpy(src_idx).to(DEV),
                  "gt": gt_lists,
                  "top": torch.cat(tops), "cos": torch.cat(coss)})
    base_table.append({"query": entry["query"], **agg})
BASE_MACRO = macro(base_table)
print(f"base DirectionFusion mirror: {BASE_MACRO:.3f} (ancora S1: 0.296)  "
      f"({time.time() - T0:.0f}s)")""")

code(r"""# --- scorer: coda Poisson-binomiale + re-rank generico sui pool cached ------
def pb_tail_le2(agree):
    '''log P(#disaccordi <= 2) per agreement probs (B, M, R) -> (B, M).'''
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


def rerank_eval(P_post, source_mode="probe"):
    '''Re-rank dei pool cached con posteriori P_post (n_val, 40).

    source_mode: "probe" (profilo sorgente dalle stesse sonde) oppure
    "gt" (profilo sorgente dalle label vere — solo diagnostica).
    '''
    table = []
    for e in POOLS:
        ccols, rest = e["ccols"], e["rest"]
        csigns = torch.tensor([s for s, _ in e["conds"]], device=DEV) > 0
        n = len(e["src"])
        agg = {f"{m_}@{K}": 0.0 for m_ in "RP" for K in KS}
        for lo in range(0, n, 2048):
            sl = slice(lo, min(lo + 2048, n))
            top, cos = e["top"][sl], e["cos"][sl]
            idx = e["src"][sl]
            p_cand = P_post[top]
            pc = p_cand[:, :, ccols]
            lg_con = torch.log(torch.where(csigns.view(1, 1, -1),
                                           pc, 1 - pc)).sum(-1)
            p_r = p_cand[:, :, rest]
            if source_mode == "probe":
                pref = P_post[idx][:, None, rest]
            else:
                pref = (A_VAL[idx][:, None, rest] > 0).float()
            lg_rest = pb_tail_le2(p_r * pref + (1 - p_r) * (1 - pref))
            score = lg_con + lg_rest + 1e-4 * cos
            full = torch.full((top.shape[0], N_GAL), -torch.inf, device=DEV)
            full.scatter_(1, top, score)
            part = eval_entry(full, e["gt"][sl])
            w = (sl.stop - sl.start) / n
            for k in agg:
                agg[k] += part[k] * w
        table.append({"query": e["query"], **agg})
    return table


print("scorer pronto")""")

code(r"""# --- gradino 0: sonde lineari full-train (ancora S1: dp_probe 0.425) --------
def train_logreg(c, rows=None, steps=400, lr=0.05, wd=1e-4):
    X = E_TRAIN if rows is None else E_TRAIN[rows]
    y = (A_TRAIN[:, c] > 0).float() if rows is None else Y_ALL[rows][:, c]
    w = torch.zeros(512, device=DEV, requires_grad=True)
    b = torch.zeros((), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        z = X @ w + b
        loss = F.binary_cross_entropy_with_logits(z, y) + wd * (w * w).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


t = time.time()
W_LIN = torch.zeros(40, 512, device=DEV)
B_LIN = torch.zeros(40, device=DEV)
for c in range(40):
    W_LIN[c], B_LIN[c] = train_logreg(c)
P_LIN = torch.sigmoid(V_VAL @ W_LIN.T + B_LIN).clamp(1e-5, 1 - 1e-5)
tab_lin = rerank_eval(P_LIN)
MACRO_LIN = macro(tab_lin)
print(f"lin_full dp_probe mirror: {MACRO_LIN:.3f} (ancora S1: 0.425)  "
      f"[sonde {time.time() - t:.0f}s]")""")

code(r"""# --- pool difficili sul TRAIN (per il gradino mlp_pool) ----------------------
# Sorgenti = immagini di train che violano TUTTI i vincoli (regola benchmark),
# <= 1500 per tipo di query (seed 0); candidati = top-200 nella gallery train.
t = time.time()
A_tr_np = A_SPLIT["train"]
pool_count = torch.zeros(N_TRAIN, device=DEV)
rng = np.random.default_rng(0)
for e in POOLS:
    viol = np.ones(N_TRAIN, dtype=bool)
    for s, c in e["conds"]:
        viol &= A_tr_np[:, c] != s
    src = np.nonzero(viol)[0]
    if len(src) > 1500:
        src = rng.choice(src, 1500, replace=False)
    for lo in range(0, len(src), 512):
        idx = torch.from_numpy(src[lo:lo + 512]).to(DEV)
        q = qfn(E_TRAIN[idx], e["conds"])
        sc = q @ E_TRAIN.T
        sc[torch.arange(len(idx), device=DEV), idx] = -torch.inf
        top = sc.topk(M, dim=1).indices
        pool_count.scatter_add_(0, top.flatten(),
                                torch.ones(top.numel(), device=DEV))
in_pool = int((pool_count > 0).sum())
srt = pool_count.sort(descending=True).values
share1 = float(srt[:N_TRAIN // 100].sum() / pool_count.sum())
print(f"pool train: {in_pool}/{N_TRAIN} immagini toccate; "
      f"top-1% degli slot = {share1:.0%} (eco hubness)  ({time.time() - t:.0f}s)")""")

code(r"""# --- MLP a 40 teste (512->64->1), training batched, temperatura su held-10 ---
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


def train_mlp(pool_weights=None, steps=1500, bs=8192, lr=1e-3, wd=1e-4, seed=0):
    prm = mlp_init(seed)
    opt = torch.optim.AdamW(prm, lr=lr, weight_decay=wd)
    g = torch.Generator(device=DEV).manual_seed(seed)
    if pool_weights is not None:
        pw = pool_weights[TR90].sqrt()      # radice per temperare gli hub
    for _ in range(steps):
        if pool_weights is None:
            idx = TR90[torch.randint(len(TR90), (bs,), generator=g, device=DEV)]
        else:
            u = TR90[torch.randint(len(TR90), (bs // 2,), generator=g, device=DEV)]
            p = TR90[torch.multinomial(pw, bs // 2, replacement=True, generator=g)]
            idx = torch.cat([u, p])
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
prm1 = train_mlp(None)
with torch.no_grad():
    T1 = fit_temperature(mlp_logits(E_TRAIN[HOLD], prm1), Y_ALL[HOLD])
    P_MLP1 = torch.sigmoid(mlp_logits(V_VAL, prm1) / T1).clamp(1e-5, 1 - 1e-5)
print(f"mlp_global: {time.time() - t:.0f}s | T: min {T1.min():.2f} "
      f"med {T1.median():.2f} max {T1.max():.2f}")""")

code(r"""# --- gradini mlp_pool e lin90_temp ------------------------------------------
t = time.time()
prm2 = train_mlp(pool_weights=pool_count)
with torch.no_grad():
    T2 = fit_temperature(mlp_logits(E_TRAIN[HOLD], prm2), Y_ALL[HOLD])
    P_MLP2 = torch.sigmoid(mlp_logits(V_VAL, prm2) / T2).clamp(1e-5, 1 - 1e-5)
print(f"mlp_pool: {time.time() - t:.0f}s | T: min {T2.min():.2f} "
      f"med {T2.median():.2f} max {T2.max():.2f}")

t = time.time()
W_L90 = torch.zeros(40, 512, device=DEV)
B_L90 = torch.zeros(40, device=DEV)
for c in range(40):
    W_L90[c], B_L90[c] = train_logreg(c, rows=TR90)
with torch.no_grad():
    TL = fit_temperature(E_TRAIN[HOLD] @ W_L90.T + B_L90, Y_ALL[HOLD])
    P_L90T = torch.sigmoid((V_VAL @ W_L90.T + B_L90) / TL).clamp(1e-5, 1 - 1e-5)
print(f"lin90_temp: {time.time() - t:.0f}s | T: min {TL.min():.2f} "
      f"med {TL.median():.2f} max {TL.max():.2f}")""")

code(r"""# --- diagnosi AUC: globale e di pool (validation), lineare vs MLP ------------
def auc_t(scores, labels):
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64,
                                device=scores.device)
    pos = labels > 0
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    if n_p == 0 or n_n == 0:
        return None
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


VARP = {"lin_full": P_LIN, "mlp_global": P_MLP1, "mlp_pool": P_MLP2}
auc_glob = {v: [auc_t(P[:, c], A_VAL[:, c]) for c in range(40)]
            for v, P in VARP.items()}
WEAK6 = list(np.argsort(auc_glob["lin_full"])[:6])
print("6 attributi globalmente piu' deboli (lineare):",
      [ATTR_NAMES[c] for c in WEAK6])

rng = np.random.default_rng(0)
auc_pool = {v: {c: [] for c in range(40)} for v in VARP}
for e in POOLS:
    n = len(e["src"])
    rows = rng.choice(n, min(300, n), replace=False)
    cand = torch.unique(e["top"][torch.from_numpy(rows).to(DEV)])
    for c in e["rest"]:
        lab = A_VAL[cand, c]
        for v, P in VARP.items():
            a = auc_t(P[cand, c], lab)
            if a is not None:
                auc_pool[v][c].append(a)


def pool_mean(v, cols):
    vals = [a for c in cols for a in auc_pool[v][c]]
    return float(np.mean(vals))


print(f"\n{'attr':18s}  {'glob lin':>8s} {'glob mlp':>8s} | "
      f"{'pool lin':>8s} {'pool mlpG':>9s} {'pool mlpP':>9s}")
for c in WEAK6:
    print(f"{ATTR_NAMES[c]:18s}  {auc_glob['lin_full'][c]:8.3f} "
          f"{auc_glob['mlp_global'][c]:8.3f} | "
          f"{np.mean(auc_pool['lin_full'][c]):8.3f} "
          f"{np.mean(auc_pool['mlp_global'][c]):9.3f} "
          f"{np.mean(auc_pool['mlp_pool'][c]):9.3f}")
POOL_W6 = {v: pool_mean(v, WEAK6) for v in VARP}
POOL_ALL = {v: pool_mean(v, range(40)) for v in VARP}
print(f"\npool-AUC weak-6 : lin {POOL_W6['lin_full']:.3f}  "
      f"mlpG {POOL_W6['mlp_global']:.3f}  mlpP {POOL_W6['mlp_pool']:.3f}")
print(f"pool-AUC resto  : lin {POOL_ALL['lin_full']:.3f}  "
      f"mlpG {POOL_ALL['mlp_global']:.3f}  mlpP {POOL_ALL['mlp_pool']:.3f}")""")

code(r"""# --- valutazione mirror: tutti i gradini + righe gtref -----------------------
t = time.time()
tabs = {"lin_full": tab_lin,
        "lin90_temp": rerank_eval(P_L90T),
        "mlp_global": rerank_eval(P_MLP1),
        "mlp_pool": rerank_eval(P_MLP2)}
tabs_gt = {v: rerank_eval(VARP[v], source_mode="gt") for v in VARP}

print(f"{'gradino':12s} {'dp_probe':>9s} {'dp_gtref':>9s}")
print(f"{'base':12s} {BASE_MACRO:9.3f} {'—':>9s}")
for v in tabs:
    gt = f"{macro(tabs_gt[v]):9.3f}" if v in tabs_gt else f"{'—':>9s}"
    print(f"{v:12s} {macro(tabs[v]):9.3f} {gt}")

BEST = max(tabs, key=lambda v: macro(tabs[v]))
print(f"\nmiglior gradino: {BEST} ({macro(tabs[BEST]):.3f}) — per-query vs lin_full:")
for r_lin, r_b in zip(tab_lin, tabs[BEST]):
    print(f"  {r_lin['query']:44s} {r_lin['R@10']:.3f} -> {r_b['R@10']:.3f}")
for k in ("R@1", "R@5", "R@10", "P@10"):
    print(f"  macro {k:5s}: lin {macro(tab_lin, k):.3f} -> {BEST} "
          f"{macro(tabs[BEST], k):.3f}")
print(f"({time.time() - t:.0f}s)")""")

code(r"""# --- verdetto automatico contro le soglie dichiarate + salvataggio ----------
ANCHOR = 0.425
best_macro = macro(tabs[BEST])
d_auc = POOL_W6["mlp_global"] - POOL_W6["lin_full"]
mech = d_auc >= 0.02
adopt = best_macro >= ANCHOR + 0.010
order_ok = (macro(tabs["mlp_pool"]) >= macro(tabs["mlp_global"]) - 0.005
            >= macro(tabs["lin_full"]) - 0.010)
in_band = 0.46 <= best_macro <= 0.52
calib_only = macro(tabs["lin90_temp"]) >= best_macro - 0.010

print(f"P1 meccanismo : pool-AUC weak-6 {POOL_W6['lin_full']:.3f} -> "
      f"{POOL_W6['mlp_global']:.3f} (delta {d_auc:+.3f}, soglia +0.02) -> "
      f"{'VIVO' if mech else 'FALLITO'}")
print(f"P2 macro      : best {BEST} {best_macro:.3f} "
      f"(banda dichiarata 0.46-0.52: {'dentro' if in_band else 'FUORI'}; "
      f"adozione >= {ANCHOR + 0.010:.3f}: {'SI' if adopt else 'NO'})")
print(f"P3 ordine     : mlp_pool {macro(tabs['mlp_pool']):.3f} vs "
      f"mlp_global {macro(tabs['mlp_global']):.3f} vs "
      f"lin_full {macro(tabs['lin_full']):.3f} -> "
      f"{'come previsto' if order_ok else 'SMENTITO'}")
print(f"confondente   : lin90_temp {macro(tabs['lin90_temp']):.3f} -> "
      f"{'guadagno = CALIBRAZIONE, non capacita' if calib_only else 'capacita reale oltre la calibrazione'}")
if not mech and not adopt:
    print("\nVERDETTO: S2 MORTA (tetto delle feature) — chiudere la strategia.")
elif adopt:
    print(f"\nVERDETTO: ADOTTARE {BEST} sul mirror "
          f"({ANCHOR:.3f} -> {best_macro:.3f}, {best_macro - ANCHOR:+.3f}).")
else:
    print("\nVERDETTO: segnale sul meccanismo ma macro sotto soglia — non adottare.")

out = {"base_macro": BASE_MACRO,
       "auc_global": {v: dict(zip(ATTR_NAMES, auc_glob[v])) for v in auc_glob},
       "weak6": [ATTR_NAMES[c] for c in WEAK6],
       "pool_auc_weak6": POOL_W6, "pool_auc_rest": POOL_ALL,
       "temps": {"mlp_global": T1.tolist(), "mlp_pool": T2.tolist(),
                 "lin90": TL.tolist()},
       "macros_probe": {v: macro(tabs[v]) for v in tabs},
       "macros_gtref": {v: macro(tabs_gt[v]) for v in tabs_gt},
       "tables_probe": tabs, "tables_gtref": tabs_gt,
       "best": BEST, "adopt": bool(adopt)}
with open(OUT / "s2_results_mirror.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\nsalvato in {(OUT / 's2_results_mirror.json').relative_to(OUT.parents[1])} "
      f"({time.time() - T0:.0f}s totali)")""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"scritto {OUT} ({len(CELLS)} celle)")
