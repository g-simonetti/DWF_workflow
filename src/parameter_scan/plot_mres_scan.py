#!/usr/bin/env python3
"""
plot_mres_scan.py

Default behaviour (parameter scans; unchanged structure):
- Reads m_res_fit.txt files from --mres
- Produces the "merged_m" style 1x4 plot scanning over (alpha, a5, M5, mpv)

scan_beta behaviour:
- Reads m_res.txt files from --mres_t
- Plots am_res(t) vs t/a
- Two panels: left = smaller volume, right = bigger volume
- Requested styling:
  * Symmetrize: duplicate the t/a=0 point at t/a=Nt (same y and err)
  * Mass colours from inferno reference grid (0.01..0.10)
  * Only TWO markers for masses (circle and square)
  * Lines connecting points are only '--' and ':' (like merged_m)
  * Beta text in BLACK, placed just above midpoint of a curve,
    BUT: show each beta only once, on the curve corresponding to the LARGER mass
    (so no overlapping beta labels from multiple masses)
  * Legend shows masses only (if enabled)
"""

import re
import argparse
from collections import defaultdict
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("tableau-colorblind10")


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Plot residual mass scans, grouping from filepath."
)
parser.add_argument("--use", default="merged_m", help="Mode. Use 'scan_beta' for m_res(t) vs t/a.")
parser.add_argument("--mres", nargs="*", default=[], help="List of m_res.json files")
parser.add_argument("--output_filename", required=True, help="Output plot filename")
parser.add_argument("--label", type=str, default="no", help="Set to 'yes' to include legend")
parser.add_argument("--plot_styles", default=None, help="Matplotlib style file to use")
args = parser.parse_args()

show_legend = args.label.lower() == "yes"

if args.plot_styles:
    plt.style.use(args.plot_styles)


# ------------------------------------------------------------
# Regex: parse params from file path
# ------------------------------------------------------------
pattern_mres = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
    r"residual_mass/m_res\.json"
)


# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------
def load_mres_fit(path: str) -> tuple[float, float]:
    with open(path, "r") as f:
        data = json.load(f)
    try:
        y = data["mres_extract"]["value"]
        err = data["mres_extract"]["error"]
    except Exception as e:
        raise ValueError(f"{path} missing mres_extract.value/error") from e
    return float(y), float(err)


def load_mres_t_series(path: str):
    """
    Load m_res(t) series from m_res.json:
      data["mres_series"]["t"], ["mres"], ["mres_err"]
    Returns (t, y, yerr)
    """
    with open(path, "r") as f:
        data = json.load(f)

    try:
        t = np.asarray(data["mres_series"]["t"], dtype=float)
        y = np.asarray(data["mres_series"]["mres"], dtype=float)
        yerr = np.asarray(data["mres_series"]["mres_err"], dtype=float)
    except Exception as e:
        raise ValueError(f"{path} missing mres_series.t/mres/mres_err") from e

    if t.shape != y.shape or t.shape != yerr.shape:
        raise ValueError(f"{path} has inconsistent mres_series array lengths")

    return t, y, yerr


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _k(v):
    return round(v, 12) if isinstance(v, float) else v


def parse_entries(paths, pat):
    entries = []
    for path in paths:
        m = pat.search(path)
        if m is None:
            raise ValueError(f"Cannot parse parameters from path:\n{path}")

        d = m.groupdict()
        entry = {
            k: int(v) if k in {"Nt", "Ns", "Ls"} else float(v)
            for k, v in d.items()
        }
        entry["path"] = path
        entries.append(entry)
    return entries


# ------------------------------------------------------------
# Mass colour mapping (inferno over reference s_masses grid)
# ------------------------------------------------------------
def build_mass_color_map(s_masses: np.ndarray):
    s_masses = np.array(s_masses, dtype=float)
    m_cmap = mpl.cm.inferno(np.linspace(0.1, 0.85, len(s_masses)))
    return s_masses, m_cmap


S_MASSES_DEFAULT = np.round(np.linspace(0.01, 0.10, 10), 12)
_S_MASSES, _M_CMAP = build_mass_color_map(S_MASSES_DEFAULT)


def mass_to_color(mass: float):
    mass = float(mass)
    idx = int(np.argmin(np.abs(_S_MASSES - mass)))
    return _M_CMAP[idx]


# ------------------------------------------------------------
# merged_m grouping (unchanged)
# ------------------------------------------------------------
SCAN_PARAMS = ["alpha", "a5", "M5", "mpv"]
ALL_KEYS = ["Nt", "Ns", "Ls", "beta", "mass", "mpv", "alpha", "a5", "M5"]


def build_merged_m_groups(entries_list):
    groups = defaultdict(list)
    for param in SCAN_PARAMS:
        fixed_keys = [k for k in ALL_KEYS if k != param]

        buckets = defaultdict(list)
        for e in entries_list:
            fixed = tuple((k, _k(e[k])) for k in fixed_keys)
            buckets[fixed].append(e)

        for subset in buckets.values():
            distinct_x = {_k(e[param]) for e in subset}
            if len(distinct_x) >= 2:
                groups[param].append(subset)

    return groups


# ------------------------------------------------------------
# Titles + labels (unchanged)
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
# Plot merged_m (unchanged)
# ------------------------------------------------------------
def plot_merged_m(groups, outname):
    line_styles = ["--", ":"]
    fig, axes = plt.subplots(1, 4, figsize=(7, 2), sharey=True, layout="constrained")
    params = ["alpha", "a5", "M5", "mpv"]

    for i, param in enumerate(params):
        ax = axes[i]

        if param not in groups or len(groups[param]) == 0:
            ax.axis("off")
            continue

        mass_groups = defaultdict(list)
        for subset in groups[param]:
            mass_groups[_k(subset[0]["mass"])].append(subset)

        masses = sorted(mass_groups.keys())

        for j, mass_value in enumerate(masses):
            color = mass_to_color(mass_value)
            linestyle = line_styles[j % len(line_styles)]
            marker = "o" if j == 0 else "s"

            mass_label = rf"$am_0={mass_value}$"

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
# scan_beta plot (m_res(t) vs t/a)
#   - markers encode mass (circle/square)
#   - line styles ONLY '--' and ':' (like merged_m)
#   - beta text in BLACK, shown once per beta on the *largest mass* curve
# ------------------------------------------------------------
def plot_scan_beta_time(entries_t, outname):
    if not entries_t:
        raise ValueError("Mode scan_beta requested, but --mres_t is empty.")

    # group by volume only: (Ns, Ls)
    vol_groups = defaultdict(list)
    for e in entries_t:
        vol_groups[(e["Ns"], e["Ls"])].append(e)

    # choose smallest and largest volumes by Ns^3 * Ls
    vols = sorted(vol_groups.keys(), key=lambda v: (v[0] ** 3) * v[1])
    left_vol = vols[0]
    right_vol = vols[-1] if len(vols) > 1 else vols[0]

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2), sharey=True, layout="constrained")

    line_styles = ["--", ":"]
    mass_markers = ["o", "s"]

    for ax, vol in zip(axes, [left_vol, right_vol]):
        subset = vol_groups[vol]

        # group by mass
        by_mass = defaultdict(list)
        for e in subset:
            by_mass[_k(e["mass"])].append(e)

        masses = sorted(by_mass.keys())
        if not masses:
            continue

        largest_mass = masses[-1]  # <- label betas ONLY on this mass' curves

        # legend: one label per mass
        mass_labeled = {m: False for m in masses}
        # beta labels: one per beta (per panel), placed on the largest_mass curve
        beta_labeled = defaultdict(lambda: False)

        for im, mass in enumerate(masses):
            color = mass_to_color(mass)
            marker = mass_markers[im % len(mass_markers)]
            linestyle = line_styles[im % len(line_styles)]

            entries_mass = by_mass[mass]

            # group by beta
            by_beta = defaultdict(list)
            for e in entries_mass:
                by_beta[_k(e["beta"])].append(e)

            for beta in sorted(by_beta.keys()):
                entries_beta = by_beta[beta]

                # group by Nt (we still plot all)
                by_nt = defaultdict(list)
                for e in entries_beta:
                    by_nt[e["Nt"]].append(e)

                for Nt in sorted(by_nt.keys()):
                    for e in by_nt[Nt]:
                        t, y, yerr = load_mres_t_series(e["path"])

                        # Symmetrize: duplicate t=0 at t=Nt (if needed)
                        if t.size > 0 and np.any(np.isclose(t, 0.0)):
                            idx0 = int(np.where(np.isclose(t, 0.0))[0][0])
                            if not np.any(np.isclose(t, float(Nt))):
                                t = np.append(t, float(Nt))
                                y = np.append(y, y[idx0])
                                if yerr is not None:
                                    yerr = np.append(yerr, yerr[idx0])

                        order = np.argsort(t)
                        t = t[order]
                        y = y[order]
                        if yerr is not None:
                            yerr = yerr[order]

                        # legend: mass only
                        lbl = None
                        if show_legend and (not mass_labeled[mass]):
                            lbl = rf"$am_0={mass}$"
                            mass_labeled[mass] = True

                        # plot curve
                        if yerr is None:
                            ax.plot(
                                t, y,
                                linestyle=linestyle,
                                marker=marker, markersize=3,
                                color=color,
                                label=lbl
                            )
                        else:
                            ax.errorbar(
                                t, y, yerr=yerr,
                                marker=marker, markersize=3,
                                linestyle=linestyle,
                                color=color,
                                label=lbl
                            )

                        # beta label: ONCE per beta, on the LARGEST mass curve only
                        if (mass == largest_mass) and (not beta_labeled[beta]):
                            mid = len(t) // 2
                            xmid = float(t[mid])
                            ymid = float(y[mid])

                            ax.text(
                                xmid,
                                ymid + 0.006,
                                rf"$\beta={beta}$",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                color="black",
                                bbox=dict(facecolor="white", edgecolor="none", alpha=0., pad=0.2),
                            )
                            beta_labeled[beta] = True

        ax.set_xlabel(r"$t/a$")
        ax.set_ylim(0.01, 0.17)

        if show_legend:
            ax.legend(fontsize=7, loc="upper left")

    axes[0].set_ylabel(r"$a m_{\rm res}$")

    plt.savefig(outname, dpi=300)
    plt.close()


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
use = args.use.strip().lower()

entries = parse_entries(args.mres, pattern_mres)
if not entries:
    raise ValueError("No valid m_res.json files provided via --mres.")

if use == "scan_beta":
    plot_scan_beta_time(entries, args.output_filename)
else:
    merged_groups = build_merged_m_groups(entries)
    plot_merged_m(merged_groups, args.output_filename)
