"""Extract the numbers printed by notebooks/report.ipynb into CSV files.

The report notebook is the source of truth for the final results (Kaggle run,
2026-08-10). This script parses the text outputs of its executed cells and
writes them unchanged to output/results/report/, so the README and the figures
can cite a results file. Cells are located by the text they print, not by
index. Run from the repository root:

    python -m src.extract_report_results
"""
import csv
import json
import re

from src.paths import REPO, results_dir

NB = REPO / "notebooks" / "report.ipynb"
OUT = results_dir("report")
NUM = r"[-+]?\d+\.\d+"


def cell_texts():
    with open(NB) as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join("".join(o.get("text", "")) for o in cell.get("outputs", [])
                       if o.get("output_type") == "stream" and o.get("name") == "stdout")
        if text:
            yield text


TEXTS = list(cell_texts())


def find(marker):
    hits = [t for t in TEXTS if marker in t]
    assert len(hits) == 1, f"{marker!r}: {len(hits)} cells"
    return hits[0]


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name:32s} {len(rows):3d} rows")


def table_rows(block, n_num, lead_int=False):
    """Rows '<label> [int] f1 ... fn' -> (label, int or None, [number strings])."""
    n_col = r"(?P<n>\d+)\s+" if lead_int else ""
    pat = re.compile(rf"^(?P<label>\S.*?)\s+{n_col}"
                     rf"(?P<nums>(?:{NUM}\s+){{{n_num - 1}}}{NUM})\s*$")
    out = []
    for line in block.splitlines():
        m = pat.match(line)
        if m:
            nums = m.group("nums").split()      # kept as printed
            out.append((m.group("label").strip(), m.group("n") if lead_int else None, nums))
    return out


# --- test: one line per method (B0-B9, stage 1, full) ------------------------
t = find("Ours  full system")
rows = table_rows(t.split("P@10", 1)[1], 4)
assert len(rows) == 12
write("test_methods.csv", ["method", "R@1", "R@5", "R@10", "P@10"],
      [[label, *nums] for label, _, nums in rows])

# --- test: per-query R@K and P@K for B3, stage 1, full ------------------------
t = find("Full system (+ predicate re-rank)")
per_query = []
for block in t.strip().split("\n\n"):
    system = block.splitlines()[0].split("  ")[0].strip()
    rows = table_rows(block, 6)
    assert len(rows) == 15, (system, len(rows))       # 14 entries + macro
    for i, (label, _, nums) in enumerate(rows):
        per_query.append([system, i if label != "Macro Avg." else "macro", label, *nums])
write("test_per_query.csv",
      ["system", "entry", "query", "R@1", "R@5", "R@10", "P@1", "P@5", "P@10"], per_query)

# --- test: the two declared runs (n_src per entry, deltas, bootstrap CIs) ----
t1 = find("declared run 1 - stage 1 vs B3")
n_src = [[i, label, int(n)] for i, (label, n, _) in enumerate(table_rows(t1, 2, lead_int=True))]
assert len(n_src) == 14
write("test_n_sources.csv", ["entry", "query", "n_src"], n_src)
runs = []
for t in (t1, find("declared run 2 - full system vs stage 1")):
    m = re.search(rf"delta macro R@10 \((.+?)\): ({NUM}), bootstrap 95% CI \[({NUM}), ({NUM})\]", t)
    runs.append([m.group(1), *m.groups()[1:]])
write("test_declared_runs.csv", ["comparison", "delta_R@10", "ci95_low", "ci95_high"], runs)

# --- mirror: training-free ladder ---------------------------------------------
tf = []
t = find("B3 grid search on the mirror")
m = re.search(rf"\(alpha\*, beta\*\) = \(({NUM}), ({NUM})\), macro R@10 = ({NUM})", t)
tf.append(["B3", m.group(3), f"alpha={m.group(1)} beta={m.group(2)}"])
for b in ("B0", "B1", "B2"):
    tf.append([b, re.search(rf"{b} mirror macro R@10: ({NUM})", t).group(1), ""])
t = find("H1  per-query ORACLE")
tf.append(["H1 per-query oracle", re.search(rf"ORACLE\s+macro R@10 ({NUM})", t).group(1), ""])
for marker in ("B4 erasure+injection", "B6 dual fixed steps"):
    for line in find(marker).splitlines():
        m = re.match(rf"(B\d) .*?: mirror ({NUM}) \((.*?)\)", line) or \
            re.match(rf"(B\d) .*?: mirror ({NUM}) at \((.*?)\)", line)
        if m:
            tf.append([m.group(1), m.group(2), m.group(3)])
write("mirror_training_free.csv", ["method", "mirror_R@10", "selected"], tf)

# --- mirror: stage-1 training (pipeline check, lr sweep, gating) --------------
s1 = []
t = find("learned (a, b)")
m = re.search(rf"learned \(a, b\) = \(({NUM}), ({NUM})\)", t)
s1.append(["ScalarFusion (8a pipeline check)", "",
           re.search(rf"mirror macro R@10: ({NUM})", t).group(1),
           f"learned a={m.group(1)} b={m.group(2)}, 1000 steps"])
t = find("selected on the mirror: lr")
for lr, r in re.findall(rf"^\s+(\d\.\d+)\s+({NUM})\s*$", t, re.M):
    s1.append(["DirectionFusion", lr, r, ""])
t = find("gated mirror R@10")
m = re.search(rf"\[gated_s3000_lr(\S+)\] trained: mirror macro R@10 ({NUM})", t)
s1.append(["GatedFusion", m.group(1), m.group(2), ""])
write("mirror_stage1.csv", ["model", "lr", "mirror_R@10", "note"], s1)

# --- mirror: oracle decomposition of the stage-1 ranking ----------------------
t = find("o-sat  o-rel")
write("mirror_oracles.csv", ["query", "base", "cov50", "cov200", "cov500", "o-sat", "o-rel"],
      [[label, *nums] for label, _, nums in table_rows(t, 6)])

# --- mirror: stage-2 probe ladder and structure ablations ---------------------
s2 = [["stage 1 (DirectionFusion)", re.search(rf"\[stage 1: ({NUM})\]", find("(0) linear probes")).group(1)]]
t = find("(0) linear probes")
s2.append(["(0) linear probes", re.search(rf"mirror macro R@10 ({NUM})", t).group(1)])
s2.append(["(0) linear probes, GT source profile", re.search(rf"GT source profile: ({NUM})", t).group(1)])
t = find("(b) MLP + temperature")
s2.append(["(a) linear-90 + temperature", re.search(rf"\(a\) linear-90 \+ temperature: mirror ({NUM})", t).group(1)])
s2.append(["(b) MLP + temperature", re.search(rf"\(b\) MLP \+ temperature:\s+mirror ({NUM})", t).group(1)])
s2.append(["(b) MLP + temperature, GT source profile", re.search(rf"GT source profile: ({NUM})", t).group(1)])
t = find("constraint half only")
for label, pat in (("constraint half only", "constraint half only"),
                   ("preservation half only", "preservation half only"),
                   ("joint predicate, M=100", r"joint predicate, M= 100"),
                   ("joint predicate, M=200", r"joint predicate, M= 200"),
                   ("joint predicate, M=500", r"joint predicate, M= 500")):
    s2.append([label, re.search(rf"{pat}\s*: ({NUM})", t).group(1)])
t = find("o-sat  o-rel")
s2.append(["coverage ceiling cov@200", re.search(rf"^macro\s+(?:{NUM}\s+){{2}}({NUM})", t, re.M).group(1)])
write("mirror_stage2.csv", ["variant", "mirror_R@10"], s2)
t = find("stage1  linear     MLP")
write("mirror_stage2_per_query.csv", ["query", "stage1", "linear", "MLP"],
      [[label, *nums] for label, _, nums in table_rows(t, 3)])

# --- stage-1 geometry and zero-shot direction AUC -----------------------------
t = find("cos(e-, -e+)")
write("direction_geometry.csv", ["attribute", "norm_e+", "norm_e-", "cos(e+,d)", "cos(e-,-d)", "cos(e-,-e+)"],
      [[label, *nums] for label, _, nums in table_rows(t, 5)])
t = find("mean val AUC over the 12 benchmark attributes")
auc = []
for line in t.splitlines():
    m = re.match(rf"^(\w+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+(\S+)\s*(\*?)\s*$", line)
    if m:
        auc.append([m.group(1), *m.groups()[1:5], m.group(6), bool(m.group(7))])
assert len(auc) == 40
write("text_direction_auc.csv", ["attribute", "freq+", "raw", "delta-ant", "delta-base", "choice", "in_benchmark"], auc)

# --- SAE appendix -------------------------------------------------------------
sae = []
t = find("Gate 1 reconstruction cos")
sae.append(["gate1_reconstruction_cos", re.search(rf"reconstruction cos\s+({NUM})", t).group(1)])
sae.append(["gate3_dead_fraction", re.search(rf"dead neuron fraction\s+({NUM})", t).group(1)])
t = find("gate 2  permutation coverage")
sae.append(["gate2_permutation_coverage", re.search(r"permutation coverage\s+(\d+/\d+)", t).group(1)])
sae.append(["null_p95", re.search(rf"p95 ({NUM})", t).group(1)])
t = find("concept-space overlap")
sae.append(["overlap_image_directions", re.search(rf"image directions \(B5\)\s+({NUM})", t).group(1)])
sae.append(["overlap_text_directions", re.search(rf"text directions \(B3\)\s+({NUM})", t).group(1)])
write("sae_summary.csv", ["metric", "value"], sae)
