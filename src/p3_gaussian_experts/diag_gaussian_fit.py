"""Proposta 3, diagnosi anticipata: la gaussiana fitta meglio della sonda lineare?

Per ogni attributo, momenti di classe dal train (mu, Sigma con shrinkage
Ledoit-Wolf calcolato su GPU) e tre AUC su validation:
  LDA  : LLR gaussiano con covarianza condivisa (lineare, caso annidato)
  QDA  : LLR gaussiano con covarianza per classe (quadratico = l'esperto PoE)
  probe: regressione logistica (300 passi Adam full-batch, tutte e 40 insieme)
Regola dichiarata nella proposta: se la gaussiana (QDA) fitta peggio del
probe, fermarsi. Zero training del modello finale, solo validation.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.paths import CELEBA, load_features, results_dir
OUT = results_dir("p3_gaussian_experts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

BENCH_ATTRS = ["Smiling", "Eyeglasses", "Heavy_Makeup", "Male", "Young",
               "Blond_Hair", "Mustache", "Black_Hair", "Wavy_Hair",
               "Chubby", "Wearing_Hat", "Wearing_Lipstick"]

lines = (CELEBA / "list_attr_celeba.txt").read_text().splitlines()
ATTR_NAMES = lines[1].split()
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
X_TR = E[SPLIT_IDX["train"]].to(DEV)
X_VA = E[SPLIT_IDX["val"]].to(DEV)
Y_TR = torch.from_numpy((A_SPLIT["train"] == 1)).to(DEV)
Y_VA = (A_SPLIT["val"] == 1)
D = X_TR.shape[1]


def ledoit_wolf(X):
    """Shrinkage LW verso (tr(S)/d) I, tutto su GPU. Ritorna Sigma, lambda."""
    n = X.shape[0]
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / n
    m = torch.trace(S) / D
    d2 = ((S - m * torch.eye(D, device=DEV)) ** 2).sum()
    q = (Xc @ S * Xc).sum(1)                      # x_c^T S x_c
    norms4 = (Xc ** 2).sum(1) ** 2
    b2_raw = (norms4.sum() - 2 * q.sum() + n * (S ** 2).sum()) / n ** 2
    b2 = torch.minimum(b2_raw, d2)
    lam = (b2 / d2).item()
    return (1 - lam) * S + lam * m * torch.eye(D, device=DEV), lam, mu


def gauss_logpdf(X, mu, Sigma):
    L = torch.linalg.cholesky(Sigma)
    z = torch.cholesky_solve((X - mu).T, L)       # Sigma^-1 (x-mu)
    quad = ((X - mu).T * z).sum(0)
    logdet = 2 * torch.log(torch.diagonal(L)).sum()
    return -0.5 * (quad + logdet)


def auc(scores, labels):
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    ranks = scores.argsort().argsort().astype(np.float64) + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --- probe logistiche, tutte e 40 in parallelo ------------------------------
W = torch.zeros(40, D, device=DEV, requires_grad=True)
b = torch.zeros(40, device=DEV, requires_grad=True)
opt = torch.optim.Adam([W, b], lr=0.05)
Yf = Y_TR.float()
for step in range(300):
    logits = X_TR @ W.T + b
    loss = F.binary_cross_entropy_with_logits(logits, Yf)
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    probe_scores = (X_VA @ W.T + b).cpu().numpy()
print(f"probe logistiche: loss finale {loss.item():.4f}", flush=True)

# --- LDA / QDA per attributo ------------------------------------------------
report = []
print(f"\n{'attributo':<18} {'n_pos':>7} {'AUC_LDA':>8} {'AUC_QDA':>8} "
      f"{'AUC_probe':>9} {'QDA-probe':>9} {'lam1':>5} {'lam0':>5}")
for a, name in enumerate(ATTR_NAMES):
    X1, X0 = X_TR[Y_TR[:, a]], X_TR[~Y_TR[:, a]]
    S1, lam1, mu1 = ledoit_wolf(X1)
    S0, lam0, mu0 = ledoit_wolf(X0)
    n1, n0 = len(X1), len(X0)
    Sp = (n1 * S1 + n0 * S0) / (n1 + n0)          # pooled per LDA
    with torch.no_grad():
        llr_qda = (gauss_logpdf(X_VA, mu1, S1)
                   - gauss_logpdf(X_VA, mu0, S0)).cpu().numpy()
        llr_lda = (gauss_logpdf(X_VA, mu1, Sp)
                   - gauss_logpdf(X_VA, mu0, Sp)).cpu().numpy()
    y = Y_VA[:, a].astype(int)
    r = {"attr": name, "n_pos_train": int(n1),
         "auc_lda": auc(llr_lda, y), "auc_qda": auc(llr_qda, y),
         "auc_probe": auc(probe_scores[:, a], y),
         "lw_lambda_pos": lam1, "lw_lambda_neg": lam0,
         "bench": name in BENCH_ATTRS}
    report.append(r)
    mark = "  <- benchmark" if r["bench"] else ""
    print(f"{name:<18} {n1:>7} {r['auc_lda']:>8.3f} {r['auc_qda']:>8.3f} "
          f"{r['auc_probe']:>9.3f} {r['auc_qda'] - r['auc_probe']:>+9.3f} "
          f"{lam1:>5.2f} {lam0:>5.2f}{mark}", flush=True)

for label, sel in (("tutti i 40", report),
                   ("benchmark", [r for r in report if r["bench"]])):
    dl = np.mean([r["auc_lda"] for r in sel])
    dq = np.mean([r["auc_qda"] for r in sel])
    dp = np.mean([r["auc_probe"] for r in sel])
    print(f"\n[{label}] AUC medie: LDA {dl:.3f}  QDA {dq:.3f}  probe {dp:.3f}"
          f"  (QDA-probe {dq - dp:+.3f}, QDA-LDA {dq - dl:+.3f})")
n_worse = sum(r["auc_qda"] < r["auc_probe"] for r in report if r["bench"])
print(f"attributi benchmark con QDA < probe: {n_worse}/"
      f"{sum(r['bench'] for r in report)}")

OUT.mkdir(exist_ok=True)
json.dump(report, open(OUT / "gaussian_fit_diagnostic.json", "w"), indent=1)
print("\nsalvato gaussian_fit_diagnostic.json", flush=True)
