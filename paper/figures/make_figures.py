#!/usr/bin/env python
"""Generate the architecture-dependence figure for pic_calculus.tex §8.

Two independent probes of "how much of the model the decode engages" both split Pythia vs Qwen:
  (A) attribution — r_eff, the effective number of decode blocks (tau* sweep, 14m→72B, recon 1.00).
  (B) causal     — k*/L, the convergence depth (fieldrun --converge-depth).

Data embedded for reproducibility (sources: pil/results/tau_star_entropy_72b.txt and
pil/results/converge_depth_ladder.txt). Run with any python that has matplotlib:
    pil/.venv/bin/python make_figures.py   ->  arch_dependence.pdf (+ .png preview)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- (A) tau* sweep: (params_M, r_eff) by architecture (14 models, to 72B) ---
PYTHIA_REFF = [(14, 8.1), (70, 6.9), (160, 11.1), (410, 23.2), (1410, 21.0),
               (2780, 25.6), (6900, 26.1), (11600, 30.0)]
QWEN_REFF = [(494, 13.2), (1540, 15.3), (3090, 13.9), (7620, 10.7), (14770, 15.7), (72710, 16.8)]

# --- (B) causal convergence depth: (params_M, mean k*/L) by architecture (current ladder) ---
PYTHIA_CD = [(70, 0.967), (160, 0.962), (410, 0.853), (1010, 0.815), (1410, 0.804)]
QWEN_CD = [(494, 0.908), (1540, 0.874), (3090, 0.927)]

PY = dict(color="#1f77b4", marker="o", label="Pythia (GPT-NeoX)")
QW = dict(color="#d62728", marker="s", label="Qwen2.5 (rope)")


def series(ax, pts, style):
    xs, ys = zip(*sorted(pts))
    ax.plot(xs, ys, linestyle="-", linewidth=1.5, markersize=6, **style)


fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))

series(a, PYTHIA_REFF, PY)
series(a, QWEN_REFF, QW)
a.set_xscale("log")
a.set_xlabel("parameters (millions)")
a.set_ylabel(r"$r_{\mathrm{eff}}$  (effective decode blocks)")
a.set_title("(A) attribution: decode blocks engaged")
a.legend(frameon=False, fontsize=8, loc="upper left")
a.text(0.5, 0.06, "Pythia grows ($\\approx$0.5$\\,n_b$);  Qwen flat",
       transform=a.transAxes, fontsize=8, color="#555")

series(b, PYTHIA_CD, PY)
series(b, QWEN_CD, QW)
b.set_xscale("log")
b.set_xlabel("parameters (millions)")
b.set_ylabel(r"mean $k^\star/L$  (causal convergence depth)")
b.set_title("(B) causal: prediction depth")
b.set_ylim(0.70, 1.01)
b.legend(frameon=False, fontsize=8, loc="lower left")
b.text(0.5, 0.9, "deep is the norm ($\\geq$0.8);  Pythia declines, Qwen breaks",
       transform=b.transAxes, fontsize=8, color="#555", ha="center")

for ax in (a, b):
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)

fig.tight_layout()
fig.savefig("arch_dependence.pdf", bbox_inches="tight")
fig.savefig("arch_dependence.png", dpi=140, bbox_inches="tight")
print("wrote arch_dependence.pdf + .png")
