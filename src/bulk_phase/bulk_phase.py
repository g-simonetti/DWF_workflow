#!/usr/bin/env python3
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import matplotlib as mpl

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
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
# Path patterns (NF2-like and NF0-like)
# ------------------------------------------------------------
pattern_nf = re.compile(
    r"NF(?P<NF>\d+)/"
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
    r"hmc/log_hmc_extract\.txt"
)

# ------------------------------------------------------------
# Load metadata (kept; not used directly below, but preserved)
# ------------------------------------------------------------
with open(args.metadata, "r") as f:
    metadata = yaml.safe_load(f)

# ------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------
def load_plaq_avg(filepath):
    with open(filepath) as f:
        header = f.readline().split()
        values = f.readline().split()
    d = dict(zip(header, values))
    return float(d["plaq"]), float(d["plaq_err"])

def load_plaq_history(filepath):
    t, p = np.loadtxt(filepath, unpack=True)
    return t, p

def derive_beta_plot_path(path: str) -> str:
    """
    Create a second output path without requiring a new CLI arg.
    Example: bulk_merged.png -> bulk_merged_vs_beta.png
    """
    m = re.search(r"(\.[^.]+)$", path)
    if m:
        ext = m.group(1)
        return path[: -len(ext)] + "_vs_beta" + ext
    return path + "_vs_beta.png"

# ------------------------------------------------------------
# Parse input entries (now also reads NF from path)
# ------------------------------------------------------------
entries = []
for fp in args.plaq_avg:
    m = pattern_nf.search(fp)
    if m is None:
        raise ValueError(f"Cannot parse input path: {fp}")

    p = m.groupdict()
    e = {
        k: float(v) if k not in {"NF", "Nt", "Ns", "Ls"} else int(float(v))
        for k, v in p.items()
    }
    e["avg_path"] = fp
    entries.append(e)

# Attach histories
if len(args.plaq_history) != len(entries):
    raise ValueError("Mismatch: plaq_history count != plaq_avg count")

for e, hist_fp in zip(entries, args.plaq_history):
    e["history_path"] = hist_fp

# ------------------------------------------------------------
# Split datasets:
#   - NF!=0: normal NF2-style points/curves
#   - NF==0: draw as horizontal lines (one per beta)
# ------------------------------------------------------------
entries_nf0 = [e for e in entries if e.get("NF", None) == 0]
entries_nfX = [e for e in entries if e.get("NF", None) != 0]  # "normal" ones

if len(entries_nfX) == 0:
    raise ValueError("No non-NF0 entries found (nothing to plot as points)")

# For merged plot we use only NF!=0 series as before
entries_full = list(entries_nfX)

# For history plot we keep the same mass filter, but only for NF!=0
allowed_masses = {0.01, 0.10}
entries_hist = [
    e for e in entries_nfX
    if np.isclose(e["mass"], list(allowed_masses)).any()
]
if not entries_hist:
    raise ValueError("No non-NF0 entries left after mass filtering for history plot")

# ------------------------------------------------------------
# Group by beta
# ------------------------------------------------------------
beta_groups_full = defaultdict(list)
beta_groups_hist = defaultdict(list)
beta_groups_nf0 = defaultdict(list)

for e in entries_full:
    beta_groups_full[e["beta"]].append(e)

for e in entries_hist:
    beta_groups_hist[e["beta"]].append(e)

for e in entries_nf0:
    beta_groups_nf0[e["beta"]].append(e)

betas = sorted(beta_groups_full)
betas_hist = sorted(beta_groups_hist)

# ------------------------------------------------------------
# (1) Colormap: viridis
# ------------------------------------------------------------
cmap = mpl.cm.get_cmap("viridis")

n = len(betas)
u = np.linspace(0, 1, n)

# keep your gentle nonlinear stretch
vals = (np.exp(1.5 * u) - 1) / (np.exp(1.5) - 1)
vals[-1] = 1.0

beta_to_color = {b: cmap(v) for b, v in zip(betas, vals)}

markers = ["o", "s", "D", "^", "v", "<", ">"]

# ============================================================
# PART 1 — MERGED PLOT (all masses for NF!=0)
#   plus NF0 horizontal lines
# ============================================================
fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")

# Track x-range (still useful for setting limits if you want)
x_min, x_max = np.inf, -np.inf

for i, beta in enumerate(betas):
    group = beta_groups_full[beta]

    masses, plaquettes, errors = [], [], []
    for e in group:
        p, err = load_plaq_avg(e["avg_path"])
        masses.append(e["mass"])
        plaquettes.append(p)
        errors.append(err)

    masses, plaquettes, errors = zip(*sorted(zip(masses, plaquettes, errors)))

    x_min = min(x_min, min(masses))
    x_max = max(x_max, max(masses))

    color = beta_to_color[beta]
    marker = markers[i % len(markers)]

    ax.errorbar(
        masses,
        plaquettes,
        yerr=errors,
        fmt=marker,
        linestyle="none",
        color=color,
        mec=color,
        mfc=color,
        label=rf"$\beta={beta}$",
        zorder=2,
    )

    ax.plot(masses, plaquettes, ":", color=color, zorder=1)

# (2) NF0: horizontal lines using axhline; place them over points with zorder
for beta, group0 in beta_groups_nf0.items():
    if beta not in beta_to_color:
        continue
    color = beta_to_color[beta]
    for e0 in group0:
        p0, _ = load_plaq_avg(e0["avg_path"])
        ax.axhline(
            p0,
            color=color,
            linestyle="--",   # dashed
            alpha=0.95,
            linewidth=1.2,
            zorder=5,         # ensure it sits over points
        )

ax.set_xlabel(r"$am_0$")
ax.set_ylabel(r"$\langle P \rangle$")
ax.set_ylim(0.36, 0.64)

if show_legend:
    ax.legend(loc="upper right", ncol=4)

fig.savefig(args.bulk_merged, dpi=300)
plt.close(fig)

# ============================================================
# PART 3 — NEW PLOT: <P> vs beta, labels are bare masses
#   (auto output name derived from bulk_merged; no new CLI arg)
# ============================================================
beta_plot_path = derive_beta_plot_path(args.bulk_merged)

# Organize NF!=0 averages by mass
mass_groups = defaultdict(list)  # mass -> list of (beta, plaq, err)
for beta in betas:
    for e in beta_groups_full[beta]:
        p, err = load_plaq_avg(e["avg_path"])
        mass_groups[e["mass"]].append((beta, p, err))

unique_masses = sorted(mass_groups.keys())
cm_mass = mpl.cm.get_cmap("viridis", max(2, len(unique_masses)))

fig, ax = plt.subplots(figsize=(6.5, 3.0), layout="constrained")

for j, mass in enumerate(unique_masses):
    triplets = sorted(mass_groups[mass], key=lambda t: t[0])
    xb = [t[0] for t in triplets]
    yp = [t[1] for t in triplets]
    ye = [t[2] for t in triplets]

    ax.errorbar(
        xb,
        yp,
        yerr=ye,
        fmt=markers[j % len(markers)],
        linestyle=":",
        color=cm_mass(j),
        label=rf"$am_0={mass}$",
        zorder=2,
    )

ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\langle P \rangle$")
ax.set_ylim(0.36, 0.64)

if show_legend:
    ax.legend(loc="best", ncol=3)

fig.savefig(beta_plot_path, dpi=300)
plt.close(fig)

# ============================================================
# PART 2 — HISTORY MULTIPLOT (two masses for NF!=0)
#   plus NF0 horizontal lines (axhline)
# ============================================================
n_beta = len(betas_hist)

fig, axes = plt.subplots(
    n_beta, 1,
    figsize=(3.5, 3),
    sharex=True,
    layout="constrained",
)

if n_beta == 1:
    axes = [axes]

# Global MC time minimum (based on NF!=0 histories, as before)
t_min_global = np.inf
for beta in betas_hist:
    for e in beta_groups_hist[beta]:
        t, _ = load_plaq_history(e["history_path"])
        t_min_global = min(t_min_global, t.min())

if not np.isfinite(t_min_global):
    t_min_global = 0.0

mass_linestyle = {0.01: ":", 0.10: ":"}

# Plot (reverse so largest β appears on top)
for ax, beta in zip(axes[::-1], betas_hist):
    color = beta_to_color[beta]
    group = sorted(beta_groups_hist[beta], key=lambda e: e["mass"])

    for e in group:
        t, p = load_plaq_history(e["history_path"])
        ax.plot(
            t,
            p,
            color=color,
            alpha=0.6,
            linestyle=mass_linestyle.get(e["mass"], ":"),
            label=rf"$am_0={e['mass']}$",
            zorder=1,
        )

    # NF0: horizontal line at the NF0 plaquette for this beta
    if beta in beta_groups_nf0:
        for e0 in beta_groups_nf0[beta]:
            p0, _ = load_plaq_avg(e0["avg_path"])
            ax.axhline(
                p0,
                color=color,
                linestyle="--",
                alpha=0.95,
                linewidth=1.2,
                zorder=5,  # above traces
            )

    ax.set_ylabel(rf"$\mathcal{{P}}\;[\beta={beta}]$")

    if show_legend and ax is axes[0]:
        ax.legend(loc="upper right")

axes[-1].set_xlabel("Monte Carlo time")
axes[-1].set_xlim(left=t_min_global, right=6900)

fig.savefig(args.bulk_single, dpi=300)
plt.close(fig)
