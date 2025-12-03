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
parser.add_argument("--scan", required=True, help="Scan parameter (a5, alpha, M5, mpv, Ls, mass, beta, t, merged)")
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
# Read m_res_fit.txt values
# ------------------------------------------------------------
def load_mres_fit(filepath):
    data = np.loadtxt(filepath, comments="#")
    return data[0], data[1]

# ------------------------------------------------------------
# Collect data entries
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
# Group entries by scan parameter according to metadata
# ------------------------------------------------------------
def get_scan_groups(entries, metadata):
    groups = defaultdict(list)
    for block in metadata:
        scan_param = None
        for key, value in block.items():
            if isinstance(value, list):
                scan_param = key
                break
        if not scan_param:
            continue

        fixed_params = {k: v for k, v in block.items() if not isinstance(v, list)}

        subset = [
            e for e in entries
            if all(np.isclose(e[k], v) for k, v in fixed_params.items() if k in e)
        ]
        if subset:
            groups[scan_param].append(subset)
    return groups

scan_groups = get_scan_groups(data_entries, metadata_entries)

# ------------------------------------------------------------
# Legend label 
# ------------------------------------------------------------
def make_legend_label(param, ref):
    return rf"$am_0={ref['mass']}$"

# ------------------------------------------------------------
# Subplot titles 
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Axis labels
# ------------------------------------------------------------
xlabels = {
    "mpv": r"$am_{\mathrm{PV}}$",
    "M5": r"$am_5$",
    "alpha": r"$\alpha$",
    "a5": r"$a_5/a$",
    "mass": r"$am_0$",
    "beta": r"$\beta$",
    "Ls": r"$L_s$",
}

# ------------------------------------------------------------
# MERGED PLOT (4 subplots)
# ------------------------------------------------------------
def plot_merged(groups, outname):
    colors = {"mpv": "C1", "M5": "C3", "alpha": "C5", "a5": "C4"}

    fig, axes = plt.subplots(1, 4, figsize=(7, 2), sharey=True, layout="constrained")

    params = ["alpha", "a5", "M5", "mpv"]

    for i, param in enumerate(params):
        ax = axes[i]

        # Blank out mpv panel completely
        if param == "mpv" or param == "a5" or param == "M5" or param not in groups:
            ax.axis("off")
            continue

        for subset in groups[param]:
            x_vals, y_vals, y_err = [], [], []

            for e in subset:
                y, err = load_mres_fit(e["filepath"])
                x_vals.append(e[param])
                y_vals.append(y)
                y_err.append(err)

            x_sorted, y_sorted, err_sorted = zip(*sorted(zip(x_vals, y_vals, y_err)))

            ref = subset[0]
            Ls = ref["Ls"]
            marker = "o" if Ls == 8 else "s"
            color = colors[param]

            legend_label = make_legend_label(param, ref)
            ax.set_title(make_subplot_title(param, ref), fontsize=8)

            ax.errorbar(
                x_sorted, y_sorted, yerr=err_sorted,
                fmt=marker, mec=color, color=color,
                label=legend_label
            )

            ax.plot(x_sorted, y_sorted, "--", color=color)
            ax.set_xlabel(xlabels[param])
            ax.set_yscale("log")
            ax.set_ylim(6e-4, 6e-2)

            if i == 0:
                ax.set_ylabel(r"$a m_{\mathrm{res}}$")

            if show_legend:
                ax.legend(loc="upper center")

    plt.savefig(outname, dpi=300)
    plt.close()
    print(f"Merged plot saved to {outname}")

# ------------------------------------------------------------
# SINGLE SCAN plot
# ------------------------------------------------------------
def plot_single(scan_param, entries, outname):
    x_vals, y_vals, y_err = [], [], []

    for e in entries:
        y, err = load_mres_fit(e["filepath"])
        x_vals.append(e[scan_param])
        y_vals.append(y)
        y_err.append(err)

    x_sorted, y_sorted, err_sorted = zip(*sorted(zip(x_vals, y_vals, y_err)))

    ref = entries[0]
    legend_label = make_legend_label(scan_param, ref)
    xlabel = xlabels.get(scan_param, scan_param)

    plt.figure(figsize=(3.5, 2), layout="constrained")
    marker = "o" if ref["Ls"] == 8 else "s"

    plt.errorbar(
        x_sorted, y_sorted, yerr=err_sorted,
        fmt=marker, mec="C0", color="C0",
        label=legend_label
    )

    plt.plot(x_sorted, y_sorted, "--", color="C0")
    plt.xlabel(xlabel)
    plt.ylabel(r"$a m_{\mathrm{res}}$")
    plt.yscale("log")

    if show_legend:
        plt.legend(loc="best")

    plt.savefig(outname)
    plt.close()
    print(f"Plot saved to {outname}")

# ------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------
if args.scan == "merged":
    plot_merged(scan_groups, args.output_filename)
else:
    if args.scan not in scan_groups:
        raise ValueError(f"No scan group for '{args.scan}' in metadata.")
    plot_single(args.scan, scan_groups[args.scan][0], args.output_filename)
