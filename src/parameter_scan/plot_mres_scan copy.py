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
parser = argparse.ArgumentParser(description="Plot residual mass scans using metadata-driven grouping.")
parser.add_argument("input", nargs="+", help="List of m_res_fit.txt files")
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
# Regex pattern for filepaths
# ------------------------------------------------------------
pattern = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/residual_mass/m_res_fit\.txt"
)

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------
with open(args.metadata, "r") as f:
    metadata_entries = yaml.safe_load(f)

# ------------------------------------------------------------
# Load m_res_fit values
# ------------------------------------------------------------
def load_mres_fit(filepath):
    data = np.loadtxt(filepath, comments="#")
    return data[0], data[1]

# ------------------------------------------------------------
# Collect all entries from input filepaths
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
# Group entries according to metadata
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
# Labels & titles
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

xlabels = {
    "mpv": r"$am_{\rm PV}$",
    "M5": r"$am_5$",
    "alpha": r"$\alpha$",
    "a5": r"$a_5/a$",
}

# ------------------------------------------------------------
# MERGED 
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
            # collect data
            x_vals, y_vals, y_err = [], [], []
            for e in subset:
                y, err = load_mres_fit(e["filepath"])
                x_vals.append(e[param])
                y_vals.append(y)
                y_err.append(err)

            # sort by x
            xs, ys, es = zip(*sorted(zip(x_vals, y_vals, y_err)))

            # plot style
            ref = subset[0]
            marker = "o" if ref["Ls"] == 8 else "s"
            color = colors[param]

            # label with mass
            mass_label = rf"$am_0={ref['mass']}$"

            # title
            ax.set_title(make_subplot_title(param, ref), fontsize=8)

            # plot
            ax.errorbar(xs, ys, yerr=es, fmt=marker,
                        mec=color, color=color, label=mass_label)
            ax.plot(xs, ys, "--", color=color)

            # axes
            ax.set_xlabel(xlabels[param])
            ax.set_yscale("log")
            ax.set_ylim(6e-4, 6e-2)

        # y-label only on first subplot
        if i == 0:
            ax.set_ylabel(r"$a m_{\rm res}$")

        # legend
        if show_legend:
            ax.legend(loc="upper center")

    plt.savefig(outname, dpi=300)
    plt.close()


# ------------------------------------------------------------
# MERGED-MASS 
# ------------------------------------------------------------
def plot_merged_m(groups, outname):
    # Each scan parameter receives its own PALETTE of two colors
    # → avoids alpha shading, gives full color distinction between masses
    color_palettes = {
        "alpha": ["C0", "C1"],
        "a5":    ["C2", "C3"],
        "M5":    ["C4", "C5"],
        "mpv":   ["C6", "C7"],
    }

    # Different linestyles for the two masses
    line_styles = ["--", ":"]

    fig, axes = plt.subplots(1, 4, figsize=(7, 2), sharey=True, layout="constrained")
    params = ["alpha", "a5", "M5", "mpv"]

    for i, param in enumerate(params):
        ax = axes[i]

        if param not in groups:
            ax.axis("off")
            continue

        # group subsets by mass
        mass_groups = defaultdict(list)
        for subset in groups[param]:
            mass_groups[subset[0]["mass"]].append(subset)

        masses = sorted(mass_groups.keys())

        for j, mass_value in enumerate(masses):

            # MASS COLOR + STYLE
            color = color_palettes[param][j % len(color_palettes[param])]
            linestyle = line_styles[j % len(line_styles)]
            marker = "o" if j == 0 else "s"
            mass_label = rf"$am_0={mass_value}$"

            # compute small horizontal offset
            all_x = []
            for subset in mass_groups[mass_value]:
                for e in subset:
                    all_x.append(e[param])
            span = max(all_x) - min(all_x) if len(all_x) > 1 else 1.0
            dx = 0.01 * span * (-1 if j == 0 else +1)

            for subset in mass_groups[mass_value]:
                x_vals, y_vals, y_err = [], [], []
                for e in subset:
                    y, err = load_mres_fit(e["filepath"])
                    x_vals.append(e[param] + dx)
                    y_vals.append(y)
                    y_err.append(err)

                xs, ys, es = zip(*sorted(zip(x_vals, y_vals, y_err)))

                # Reference entry for title (always plotted)
                ref_for_title = subset[0]

                # Title shown for first appearance of each param
                if j == 0:
                    ax.set_title(make_subplot_title(param, ref_for_title), fontsize=8)

                # Plot
                fmt = marker + linestyle   # e.g. "o--" or "s:"
                ax.errorbar(xs, ys, yerr=es, fmt=fmt, mec=color, mfc=color, color=color,
                label=mass_label if subset is mass_groups[mass_value][0] else None)
                
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
# SINGLE scan
# ------------------------------------------------------------
def plot_single(scan_param, entries, outname):

    x_vals, y_vals, y_err = [], [], []

    for e in entries:
        y, err = load_mres_fit(e["filepath"])
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
    plt.ylabel(r"$a m_{\rm res}$")
    plt.yscale("log")

    if show_legend:
        plt.legend()

    plt.savefig(outname)
    plt.close()

# ------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------
if args.scan == "merged":
    plot_merged(scan_groups, args.output_filename)
elif args.scan == "merged_m":
    plot_merged_m(scan_groups, args.output_filename)
else:
    plot_single(args.scan, scan_groups[args.scan][0], args.output_filename)
