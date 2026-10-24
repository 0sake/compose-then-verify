"""README figures from the extracted results (output/results/report/*.csv).

Run `python -m src.extract_report_results` first, then

    python -m src.make_figures

Also copies two figures rendered by the report notebook (learning curve and
the lipstick qualitative example) to output/figures/.
"""
import base64
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.paths import OUTPUT, REPO

RES = OUTPUT / "results" / "report"
FIG = OUTPUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# one hue, light -> dark: training-free -> stage 1 -> full system
LIGHT, MID, DARK = "#86b6ef", "#3987e5", "#184f95"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def read(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


def style_x(ax, xmax):
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


# --- 1. test macro R@10, every method --------------------------------------
rows = read("test_methods.csv")
labels = [" ".join(r["method"].split()) for r in rows]
vals = [float(r["R@10"]) for r in rows]
colors = [MID if "stage 1" in l else DARK if "full system" in l else LIGHT for l in labels]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
y = range(len(rows))[::-1]
ax.barh(list(y), vals, color=colors, height=0.62)
ax.set_yticks(list(y), labels)
for yi, l, v, r in zip(y, labels, vals, rows):
    if l.startswith(("B2", "B3", "Ours")):          # selective labels
        ax.text(v + 0.008, yi, r["R@10"], va="center", fontsize=9, color=INK2)
style_x(ax, 0.62)
ax.set_xlabel("macro Recall@10 on the test benchmark (14 entries)")
ax.set_title("From latent arithmetic to verify-then-rank", loc="left", fontsize=12)
save(fig, "test_methods_r10.png")

# --- 2. per-query R@10: B3 -> stage 1 -> full ------------------------------
pq = [r for r in read("test_per_query.csv") if r["entry"] not in ("7", "macro")]   # entry 7 duplicates -Young
systems = [("B3 (tuned delta arithmetic)", "B3 (tuned arithmetic)", LIGHT),
           ("Stage 1 (learned directions)", "Stage 1 (learned directions)", MID),
           ("Full system (+ predicate re-rank)", "Full system (+ predicate re-rank)", DARK)]
queries = [r["query"] for r in pq if r["system"] == systems[0][0]]
fig, ax = plt.subplots(figsize=(7.5, 7.2))
h = 0.26
for k, (sysname, legend, color) in enumerate(systems):
    v = {r["query"]: float(r["R@10"]) for r in pq if r["system"] == sysname}
    ys = [len(queries) - 1 - i + (1 - k) * h for i in range(len(queries))]
    ax.barh(ys, [v[q] for q in queries], height=h * 0.92, color=color, label=legend)
ax.set_yticks([len(queries) - 1 - i for i in range(len(queries))], queries)
style_x(ax, 1.0)
ax.set_xlabel("Recall@10 on the test benchmark")
ax.set_title("Per-query Recall@10", loc="left", fontsize=12)
ax.legend(loc="lower right", frameon=False, fontsize=9)
save(fig, "test_per_query_r10.png")

# --- 3. stage-2 probe ladder on the validation mirror ------------------------
s2 = {r["variant"]: r["mirror_R@10"] for r in read("mirror_stage2.csv")}
steps = [("stage 1 alone", "stage 1 (DirectionFusion)", MID),
         ("+ re-rank, linear probes", "(0) linear probes", DARK),
         ("+ re-rank, linear-90 + temperature", "(a) linear-90 + temperature", DARK),
         ("+ re-rank, MLP + temperature", "(b) MLP + temperature", DARK)]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
y = list(range(len(steps)))[::-1]
ax.barh(y, [float(s2[k]) for _, k, _ in steps], color=[c for *_, c in steps], height=0.58)
ax.set_yticks(y, [l for l, *_ in steps])
for yi, (_, k, _) in zip(y, steps):
    ax.text(float(s2[k]) + 0.008, yi, s2[k], va="center", fontsize=9, color=INK2)
for key, text, ty in (("(b) MLP + temperature, GT source profile", "source profile\nfrom labels", 0),
                      ("coverage ceiling cov@200", "coverage ceiling\n(cov@200)", 1)):
    x = float(s2[key])
    ax.axvline(x, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(x + 0.008, ty, f"{text}: {s2[key]}", fontsize=8, color=INK2, va="center")
style_x(ax, 1.0)
ax.set_xlabel("macro Recall@10 on the validation mirror (13 queries)")
ax.set_title("Stage 2: what the predicate re-rank adds, and what is left", loc="left", fontsize=12)
save(fig, "mirror_stage2_ladder.png")

# --- 4. figures rendered by the report notebook -------------------------------
with open(REPO / "notebooks" / "report.ipynb") as f:
    cells = json.load(f)["cells"]
for marker, name in (("learning curve of the selected run", "stage1_learning_curve.png"),
                     ('find_entry((1, "Wearing_Lipstick")', "qualitative_lipstick_triple.png")):
    hits = [c for c in cells if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(hits) == 1, marker
    png = [o["data"]["image/png"] for o in hits[0]["outputs"] if "image/png" in o.get("data", {})]
    assert len(png) == 1, marker
    (FIG / name).write_bytes(base64.b64decode("".join(png[0])))
    print("wrote", FIG / name)
