#!/usr/bin/env python3
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

plt.style.use("tableau-colorblind10")

# ------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot plaquette scans using metadata-driven grouping.")
parser.add_argument("input", nargs="+", help="List of log_hmc_extract.txt files")   # >>> CHANGED
parser.add_argument("--output_filename", required=True, help="Output plot filename")
parser.add_argument("--label", type=str, default="no", help="Set to 'yes' to include legend")
parser.add_argument("--plot_styles", default=None, help="Matplotlib style file to use")
parser.add_argument("--scan", required=True, help="Scan parameter (a5, alpha, M5, mpv, merged, merged_m)")
parser.add_argument("--metadata", required=True, help="YAML metadata file describing scan structure")
args = parser.parse_args()

show_legend = args.label.lower() == "yes"

if args.plot_styles:
    plt.style.use(args.plot_styles)

# ------------------------------------------------------------
# Regex pattern for filepaths (IDENTICAL except filename)
# ------------------------------------------------------------
pattern = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
    r"hmc/log_hmc_extract\.txt"     # >>> CHANGED
)

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------
with open(args.metadata, "r") as f:
    metadata_entries = yaml.safe_load(f)

# ------------------------------------------------------------
# Load plaquette values
# ------------------------------------------------------------
def load_plaq(filepath):         # >>> CHANGED (replaces load_mres_fit)
    with open(filepath) as f:
        header = f.readline().split()
        values = f.readline().split()
    d = dict(zip(header, values))
    return float(d["plaq"]), float(d["plaq_err"])    # >>> CHANGED

# ------------------------------------------------------------
# Collect entries
# ------------------------------------------------------------
data_entries = []
for filepath in args.input:
    m = pattern.search(filepath)
    if m is None:
        raise ValueError(f"Cannot parse parameters from path: {filepath}")

    p = m.groupdict()
    entry = {k: float(v) if k not in ["Nt", "Ns", "Ls"] else int(v) for k, v in p.items()}
    entry["filepath"] = filepath
    data_entries.append(entry)

# ------------------------------------------------------------
# Group entries (UNCHANGED)
# ------------------------------------------------------------
def get_scan_groups(entries, metadata):
    groups = defaultdict(list)

    for block in metadata:
        scan_param = next((k for k, v in block.items() if isinstance(v, list)), None)
        if scan_param is None:
            continue

        fixed = {k: v for k, v in block.items() if not isinstance(v, list)}

        subset = [
            e for e in entries
            if all(np.isclose(e[k], v) for k, v in fixed.items() if k in e)
        ]

        if subset:
            groups[scan_param].append(subset)

    return groups

scan_groups = get_scan_groups(data_entries, metadata_entries)

# ------------------------------------------------------------
# Title formatting (IDENTICAL)
# ------------------------------------------------------------
def make_legend_label(param, ref):
    return rf"$am_0={ref['mass']}$"

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
    else:
        return ""

xlabels = { "mpv": r"$am_{\rm PV}$", "M5": r"$am_5$", "alpha": r"$\alpha$", "a5": r"$a_5/a$" }

# ------------------------------------------------------------
# MERGED (only load_plaq + ylabel changed)
# ------------------------------------------------------------
def plot_merged(groups, outname):
    colors = {"alpha": "C5", "a5": "C4", "M5": "C3", "mpv": "C1"}

    fig, axes = plt.subplots(1, 4, figsize=(7, 2), sharey=True, layout="constrained")
    params = ["alpha", "a5", "M5", "mpv"]

    for i, param in enumerate(params):
        ax = axes[i]

        if param not in groups:
            ax.axis("off")
            continue

        for subset in groups[param]:
            x_vals, y_vals, y_err = [], [], []
            for e in subset:
                y, err = load_plaq(e["filepath"])         # >>> CHANGED
                x_vals.append(e[param])
                y_vals.append(y)
                y_err.append(err)

            xs, ys, es = zip(*sorted(zip(x_vals, y_vals, y_err)))

            ref = subset[0]
            marker = "o" if ref["Ls"] == 8 else "s"
            color = colors[param]

            mass_label = rf"$am_0={ref['mass']}$"

            ax.set_title(make_subplot_title(param, ref), fontsize=8)
            ax.errorbar(xs, ys, yerr=es, fmt=marker, mec=color, color=color, label=mass_label)
            ax.plot(xs, ys, "--", color=color)

            ax.set_xlabel(xlabels[param])
            # keep original scale? For plaquette linear is correct, but if needed:
            # ax.set_yscale("linear")

        if i == 0:
            ax.set_ylabel(r"$\langle P\rangle$")      # >>> CHANGED

        if show_legend:
            ax.legend(loc="upper center")

    plt.savefig(outname, dpi=300)
    plt.close()

# ------------------------------------------------------------
# MERGED-MASS (only load_plaq + ylabel changed, nothing else touched)
# ------------------------------------------------------------
def plot_merged_m(groups, outname):
    colors = {"alpha": "C5", "a5": "C0", "M5": "C3", "mpv": "C1"}
    
    fig, axes = plt.subplots(1, 4, figsize=(7, 2), sharey=True, layout="constrained")
    params = ["alpha", "a5", "M5", "mpv"]

    for i, param in enumerate(params):
        ax = axes[i]

        if param not in groups:
            ax.axis("off")
            continue

        mass_groups = defaultdict(list)
        for subset in groups[param]:
            mass_groups[subset[0]["mass"]].append(subset)

        masses = sorted(mass_groups.keys())

        for j, mass_value in enumerate(masses):
            color = colors[param]
            alpha_val = 0.5 if mass_value == 0.02 else 1.0
            marker = "o" if j == 0 else "s"
            mass_label = rf"$am_0={mass_value}$"

            all_x = [e[param] for s in mass_groups[mass_value] for e in s]
            span = max(all_x) - min(all_x) if len(all_x) > 1 else 1.0
            dx = 0.01 * span * (-1 if j == 0 else +1)

            for subset in mass_groups[mass_value]:

                x_vals, y_vals, y_err = [], [], []
                for e in subset:
                    y, err = load_plaq(e["filepath"])   # >>> CHANGED
                    x_vals.append(e[param] + dx)
                    y_vals.append(y)
                    y_err.append(err)

                xs, ys, es = zip(*sorted(zip(x_vals, y_vals, y_err)))

                if param == "mpv":
                    subset = [e for e in subset if e["mass"] != 0.02]
                if not subset:
                    continue

                ref_title = subset[0]

                if param == "mpv" or j == 0:
                    ax.set_title(make_subplot_title(param, ref_title), fontsize=8)

                ax.errorbar(xs, ys, yerr=es, fmt=marker, mec=color, color=color, alpha=alpha_val, label=mass_label)
                ax.plot(xs, ys, "--", alpha=alpha_val, color=color)

        ax.set_xlabel(xlabels[param])
        if i == 0:
            ax.set_ylabel(r"$\langle P\rangle$")  # >>> CHANGED

        if show_legend:
            ax.legend(loc="best", fontsize=7)

    plt.savefig(outname, dpi=300)
    plt.close()

# ------------------------------------------------------------
# SINGLE scan
# ------------------------------------------------------------
def plot_single(scan_param, entries, outname):

    x_vals, y_vals, y_err = [], [], []

    for e in entries:
        y, err = load_plaq(e["filepath"])    # >>> CHANGED
        x_vals.append(e[scan_param])
        y_vals.append(y)
        y_err.append(err)

    xs, ys, es = zip(*sorted(zip(x_vals, y_vals, y_err)))

    ref = entries[0]
    marker = "o" if ref["Ls"] == 8 else "s"

    plt.figure(figsize=(3.5, 2), layout="constrained")

    plt.errorbar(xs, ys, yerr=es, fmt=marker, mec="C0", color="C0")
    plt.plot(xs, ys, "--", color="C0")

    plt.xlabel(xlabels.get(scan_param, scan_param))
    plt.ylabel(r"$\langle P\rangle$")     # >>> CHANGED

    if show_legend:
        plt.legend()

    plt.savefig(outname)
    plt.close()

# ------------------------------------------------------------
# Dispatch (UNCHANGED)
# ------------------------------------------------------------
if args.scan == "merged":
    plot_merged(scan_groups, args.output_filename)
elif args.scan == "merged_m":
    plot_merged_m(scan_groups, args.output_filename)
else:
    plot_single(args.scan, scan_groups[args.scan][0], args.output_filename)
