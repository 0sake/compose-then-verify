"""Shared paths for the experiment scripts in src/.

Every location can be overridden with an environment variable, so the scripts
run unchanged on any machine. Run the scripts from the repository root as
modules, e.g. `python -m src.predicate_rerank.restball_rerank`.

    CELEBA_DIR       folder with list_attr_celeba.txt, list_eval_partition.txt
                     (and img_align_celeba/ for the report notebook)
    CELEBA_EVAL      benchmark JSON (default: data/celeba_evaluation.json)
    CLIP_FEATURES    cached CLIP ViT-B/32 features, .npz written by the report
                     notebook (Section 4) or a .pt dict with the same keys
    CTV_CHECKPOINTS  DirectionFusion checkpoints and direction sets
    CTV_OUTPUT       root of results/ (JSON) written by the scripts
"""
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

CELEBA = Path(os.environ.get("CELEBA_DIR", DATA / "celeba"))
EVAL_JSON = Path(os.environ.get("CELEBA_EVAL", DATA / "celeba_evaluation.json"))
FEATURES = Path(os.environ.get("CLIP_FEATURES", DATA / "clip_vitb32_celeba.npz"))
CHECKPOINTS = Path(os.environ.get("CTV_CHECKPOINTS", REPO / "checkpoints"))
OUTPUT = Path(os.environ.get("CTV_OUTPUT", REPO / "output"))


def results_dir(name):
    """output/results/<name>/, created on first use."""
    d = OUTPUT / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# P1 mirror table, used by most scripts to assert the rebuilt mirror (n_src)
VETO_JSON = OUTPUT / "results" / "p1_negation_veto" / "veto_ablation_mirror.json"


def load_features(path=FEATURES):
    """{'filenames': [str], 'embeddings': float32 tensor (N, 512), unnormalized}."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        return {"filenames": [str(f) for f in z["filenames"]],
                "embeddings": torch.from_numpy(z["embeddings"].astype(np.float32))}
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {"filenames": list(d["filenames"]),
            "embeddings": torch.as_tensor(d["embeddings"]).float()}
