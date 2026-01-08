#!/usr/bin/env python3
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

plt.style.use("tableau-colorblind10")

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Bulk phase plotting tool.")

parser.add_argument("--plaq_avg", nargs="+", help="log_hmc_extract.txt files")
parser.add_argument("--plaq_history", nargs="+", help="plaq_history.txt files")

parser.add_argument("--metadata", required=True)
parser.add_argument("--label", default="no")
parser.add_argument("--plot_styles", default=None)

parser.add_argument("--bulk_merged", required=True)
parser.add_argument("--bulk_single", required=True)

args = parser.parse_args()

show_legend = args.label.lower() == "yes"
if args.plot_styles:
    plt.style.use(args.plot_styles)

# ------------------------------------------------------------
# Pattern for log_hmc_extract paths
# ------------------------------------------------------------
pattern = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
    r"hmc/log_hmc_extract\.txt"
)

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------
with open(args.metadata, "r") as f:
    metadata = yaml.safe_load(f)

# ------------------------------------------------------------
# Load average plaquette
# ------------------------------------------------------------
def load_plaq_avg(filepath):
    with open(filepath) as f:
        header = f.readline().split()
        values = f.readline().split()
    d = dict(zip(header, values))
    return float(d["plaq"]), float(d["plaq_err"])

# ------------------------------------------------------------
# Load plaquette history
# ------------------------------------------------------------
def load_plaq_history(filepath):
    t, p = np.loadtxt(filepath, unpack=True)
    return t, p

# ------------------------------------------------------------
# Parse entries from plaq_avg
# ------------------------------------------------------------
entries = []
for fp in args.plaq_avg:
    m = pattern.search(fp)
    if m is None:
        raise ValueError(f"Cannot parse input path: {fp}")

    p = m.groupdict()
    e = {k: float(v) if k not in ["Nt", "Ns", "Ls"] else int(v)
         for k, v in p.items()}
    e["avg_path"] = fp
    entries.append(e)

# Attach matching plaq_history files
if len(args.plaq_history) != len(entries):
    raise ValueError("Mismatch: plaq_history count != plaq_avg count")

for e, hist_fp in zip(entries, args.plaq_history):
    e["history_path"] = hist_fp

# ------------------------------------------------------------
# NO FILTERING FOR MERGED PLOT
# merged plot uses ALL masses
# ------------------------------------------------------------
entries_full = list(entries)

# ------------------------------------------------------------
# FILTER ONLY FOR HISTORY PLOT
# ------------------------------------------------------------
allowed_masses = {0.01, 0.10}
entries_hist = [e for e in entries if np.isclose(e["mass"], list(allowed_masses)).any()]

if len(entries_hist) == 0:
    raise ValueError("No entries left after filtering masses for history plot")

# ------------------------------------------------------------
# Group by beta (for merged plot)
# ------------------------------------------------------------
beta_groups_full = defaultdict(list)
for e in entries_full:
    beta_groups_full[e["beta"]].append(e)

betas = sorted(beta_groups_full.keys())

# ------------------------------------------------------------
# Group by beta (for history plot)
# ------------------------------------------------------------
beta_groups_hist = defaultdict(list)
for e in entries_hist:
    beta_groups_hist[e["beta"]].append(e)

# ------------------------------------------------------------
# ──────────────────────
# PART 1: MERGED PLOT (all masses)
# ──────────────────────
# ------------------------------------------------------------

plt.figure(figsize=(3.5, 3), layout="constrained")

colors = [f"C{i}" for i in range(10)]
markers = ["o", "s", "D", "^", "v", "<", ">"]

for i, beta in enumerate(betas):
    group = beta_groups_full[beta]

    masses, plaquettes, errors = [], [], []
    for e in group:
        p, err = load_plaq_avg(e["avg_path"])
        masses.append(e["mass"])
        plaquettes.append(p)
        errors.append(err)

    masses, plaquettes, errors = zip(*sorted(zip(masses, plaquettes, errors)))

    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]

    plt.errorbar(
        masses, plaquettes, yerr=errors,
        fmt=marker, color=color, mec=color, label=rf"$\beta={beta}$"
    )
    plt.plot(masses, plaquettes, "--", color=color)

plt.xlabel(r"$am_0$")
plt.ylabel(r"$\langle P \rangle$")
plt.ylim(0.5870, 0.6125)
if show_legend:
    plt.legend()

plt.savefig(args.bulk_merged, dpi=300)
plt.close()

# ------------------------------------------------------------
# ──────────────────────
# PART 2: HISTORY MULTIPLOT (filtered masses only)
# ──────────────────────
# ------------------------------------------------------------

betas_hist = sorted(beta_groups_hist.keys())
n_beta = len(betas_hist)

fig, axes = plt.subplots(
    n_beta, 1,
    figsize=(3.5, 3),
    sharex=True,
    layout="constrained"
)

if n_beta == 1:
    axes = [axes]

# Global minimal MC time
t_min_global = np.inf
for beta in betas_hist:
    for e in beta_groups_hist[beta]:
        t, _ = load_plaq_history(e["history_path"])
        t_min_global = min(t_min_global, t.min())

if not np.isfinite(t_min_global):
    t_min_global = 0.0

# Plot histories
for i, (ax, beta) in enumerate(zip(axes, betas_hist)):
    group = beta_groups_hist[beta]

    for e in group:
        t, p = load_plaq_history(e["history_path"])
        ax.plot(t, p, alpha=0.5, label=rf"$am_0={e['mass']}$")

    ax.set_ylabel(rf"$\mathcal{{P}}\;[\beta={beta}]$")

    # Legend only in the top subplot
    if show_legend and i == 0:
        ax.legend(loc='upper right')

axes[-1].set_xlabel("Monte Carlo time")
axes[-1].set_xlim(left=t_min_global, right=7700)

plt.savefig(args.bulk_single, dpi=300)
plt.close()
