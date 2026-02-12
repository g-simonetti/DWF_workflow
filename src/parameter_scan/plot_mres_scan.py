#!/usr/bin/env python3
"""
plot_mres_scan.py (SIMPLIFIED)

- NO YAML
- NO CSV
- NO --scan / --metadata / --use flags
- ALWAYS produces the "merged_m" style plot
- Groups points directly from the input file paths
"""

import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("tableau-colorblind10")


# ------------------------------------------------------------
# Args (minimal)
# ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Plot residual mass scans (merged-mass only), grouping from filepath."
)
parser.add_argument("input", nargs="+", help="List of m_res_fit.txt files")
parser.add_argument("--output_filename", required=True, help="Output plot filename")
parser.add_argument("--label", type=str, default="no", help="Set to 'yes' to include legend")
parser.add_argument("--plot_styles", default=None, help="Matplotlib style file to use")
args = parser.parse_args()

show_legend = args.label.lower() == "yes"

if args.plot_styles:
    plt.style.use(args.plot_styles)


# ------------------------------------------------------------
# Regex: parse params from file path
# (matches your directory layout; NF is intentionally ignored)
# ------------------------------------------------------------
pattern = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
    r"residual_mass/m_res_fit\.txt"
)


# ------------------------------------------------------------
# Load m_res_fit robustly (works for 1x2, Nx2, etc.)
# ------------------------------------------------------------
def load_mres_fit(path: str) -> tuple[float, float]:
    data = np.loadtxt(path, comments="#")
    flat = np.ravel(data)
    if flat.size < 2:
        raise ValueError(f"{path} does not contain at least two numbers (y, err).")
    return float(flat[0]), float(flat[1])


# ------------------------------------------------------------
# Parse input file list into entries
# ------------------------------------------------------------
entries = []
for path in args.input:
    m = pattern.search(path)
    if m is None:
        raise ValueError(f"Cannot parse parameters from path:\n{path}")

    d = m.groupdict()
    entry = {
        k: int(v) if k in {"Nt", "Ns", "Ls"} else float(v)
        for k, v in d.items()
    }
    entry["path"] = path
    entries.append(entry)

if not entries:
    raise ValueError("No valid input files provided.")


# ------------------------------------------------------------
# merged_m grouping
#
# For each scan parameter in (alpha, a5, M5, mpv):
#   group points by fixed parameters excluding that scan parameter.
# Then, inside each subplot, we further group by mass (as your old merged_m did).
#
# We will only keep groups with >= 2 distinct scan values.
# ------------------------------------------------------------
SCAN_PARAMS = ["alpha", "a5", "M5", "mpv"]
ALL_KEYS = ["Nt", "Ns", "Ls", "beta", "mass", "mpv", "alpha", "a5", "M5"]


def _k(v):
    # stable hash key for floats
    return round(v, 12) if isinstance(v, float) else v


def build_merged_m_groups(entries_list):
    groups = defaultdict(list)  # groups[param] = [subset1, subset2, ...]

    for param in SCAN_PARAMS:
        fixed_keys = [k for k in ALL_KEYS if k != param]

        buckets = defaultdict(list)
        for e in entries_list:
            fixed = tuple((k, _k(e[k])) for k in fixed_keys)
            buckets[fixed].append(e)

        # Keep only groups that actually scan over 'param' (>=2 distinct values)
        for subset in buckets.values():
            distinct_x = {_k(e[param]) for e in subset}
            if len(distinct_x) >= 2:
                groups[param].append(subset)

    return groups


merged_groups = build_merged_m_groups(entries)


# ------------------------------------------------------------
# Titles + labels (reuse your style)
# ------------------------------------------------------------
xlabels = {
    "mpv": r"$am_{\rm PV}$",
    "M5": r"$am_5$",
    "alpha": r"$\alpha$",
    "a5": r"$a_5/a$",
}


def make_subplot_title(param, ref):
    alpha, a5, M5, mpv_val = ref["alpha"], ref["a5"], ref["M5"], ref["mpv"]

    if param == "a5":
        return rf"$\begin{{array}}{{c}} \alpha={alpha},\; am_5={M5}, \\ am_{{\mathrm{{PV}}}}={mpv_val} \\[6pt] \end{{array}}$"
    elif param == "alpha":
        return rf"$\begin{{array}}{{c}} a_5/a={a5},\; am_5={M5}, \\ am_{{\mathrm{{PV}}}}={mpv_val} \\[6pt] \end{{array}}$"
    elif param == "M5":
        return rf"$\begin{{array}}{{c}} \alpha={alpha},\; a_5/a={a5}, \\ am_{{\mathrm{{PV}}}}={mpv_val} \\[6pt] \end{{array}}$"
    elif param == "mpv":
        return rf"$\begin{{array}}{{c}} \alpha={alpha},\; a_5/a={a5}, \\  am_5={M5} \\[6pt] \end{{array}}$"
    return ""


# ------------------------------------------------------------
# Plot merged_m
# ------------------------------------------------------------
def plot_merged_m(groups, outname):
    # same palette idea as your original merged_m
    color_palettes = {
        "alpha": ["C0", "C1"],
        "a5":    ["C2", "C3"],
        "M5":    ["C4", "C5"],
        "mpv":   ["C6", "C7"],
    }
    line_styles = ["--", ":"]

    fig, axes = plt.subplots(1, 4, figsize=(7, 2), sharey=True, layout="constrained")
    params = ["alpha", "a5", "M5", "mpv"]

    for i, param in enumerate(params):
        ax = axes[i]

        if param not in groups or len(groups[param]) == 0:
            ax.axis("off")
            continue

        # Group subsets by mass
        mass_groups = defaultdict(list)
        for subset in groups[param]:
            mass_groups[_k(subset[0]["mass"])].append(subset)

        masses = sorted(mass_groups.keys())

        for j, mass_value in enumerate(masses):
            color = color_palettes[param][j % len(color_palettes[param])]
            linestyle = line_styles[j % len(line_styles)]
            marker = "o" if j == 0 else "s"

            mass_label = rf"$am_0={mass_value}$"

            # small x-shift to reduce overlap between masses (like your original)
            all_x = []
            for subset in mass_groups[mass_value]:
                for e in subset:
                    all_x.append(e[param])
            span = max(all_x) - min(all_x) if len(all_x) > 1 else 1.0
            dx = 0.01 * span * (-1 if j == 0 else 1)

            for idx_subset, subset in enumerate(mass_groups[mass_value]):
                x_vals, y_vals, y_err = [], [], []

                for e in subset:
                    y, err = load_mres_fit(e["path"])
                    x_vals.append(e[param] + dx)
                    y_vals.append(y)
                    y_err.append(err)

                xs, ys, es = zip(*sorted(zip(x_vals, y_vals, y_err)))

                if j == 0 and idx_subset == 0:
                    ax.set_title(make_subplot_title(param, subset[0]), fontsize=8)

                fmt = marker + linestyle
                ax.errorbar(
                    xs, ys, yerr=es,
                    fmt=fmt, mec=color, mfc=color, color=color,
                    label=mass_label if idx_subset == 0 else None
                )
                ax.plot(xs, ys, linestyle, color=color)

        ax.set_xlabel(xlabels[param])
        ax.set_yscale("log")
        ax.set_ylim(1.2e-3, 1.2e-1)

        if i == 0:
            ax.set_ylabel(r"$a m_{\rm res}$")

        if show_legend:
            ax.legend(loc="upper center", fontsize=7)

    plt.savefig(outname, dpi=300)
    plt.close()


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
plot_merged_m(merged_groups, args.output_filename)
