#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("tableau-colorblind10")

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Two-beta plotting utility.\n"
        "1) Ls scan: m_res vs Ls (Shamir + all Möbius points; Möbius line uses min m_res per Ls)\n"
        "2) Time scan: m_res vs (tau_int_plaq * time) (ONLY Shamir points + Möbius points used in the Möbius line)\n"
        "   where time comes from log_hmc_extract.txt and tau_int_plaq is the plaquette autocorrelation time.\n"
    )
)

parser.add_argument("--mres", nargs="+", required=True)
parser.add_argument("--hmc", nargs="+", required=True)
parser.add_argument("--metadata", required=True)  # kept for interface parity

parser.add_argument("--label", default="no")
parser.add_argument("--plot_styles", default=None)

parser.add_argument("--ls_scan", required=True, help="Output PDF for m_res vs Ls (2 betas).")
parser.add_argument("--costs", required=True, help="Output PDF for m_res vs (tau_int_plaq * time) (2 betas).")

# Optional: control fixed x-limits for the time panel (in seconds * tau_int units)
parser.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"))

args = parser.parse_args()
show_legend = args.label.lower() == "yes"

if args.plot_styles:
    plt.style.use(args.plot_styles)

# ------------------------------------------------------------
# Path parser
# ------------------------------------------------------------
pattern = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
)

def parse_params(path: str) -> dict:
    m = pattern.search(path)
    if m is None:
        raise ValueError(f"Cannot parse metadata from path: {path}")
    out = m.groupdict()
    return {k: (int(v) if k in ["Nt", "Ns", "Ls"] else float(v)) for k, v in out.items()}

def params_key(p: dict):
    return tuple(sorted(p.items()))

# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------
def load_mres_fit(path: str):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        return float(arr[0]), float(arr[1])
    return float(arr[0, 0]), float(arr[0, 1])

def load_hmc_extract(path: str):
    """
    Reads ONE-row log_hmc_extract.txt produced by your extractor.

    Expected columns (as you showed):
      ... bcs bcs_err t_traj t_traj_err plaq plaq_err tau_int_plaq tau_int_plaq_err ...

    Returns:
      bcs, bcs_err, t_traj, t_traj_err, plaq, plaq_err, tau_int_plaq, tau_int_plaq_err
    """
    arr = np.genfromtxt(path, names=True, dtype=None, encoding=None)

    # If multiple rows were passed, take the first row for compatibility with old behavior
    if getattr(arr, "ndim", 0) != 0:
        arr = arr[0]

    def get(name: str) -> float:
        if name not in arr.dtype.names:
            raise ValueError(
                f"Missing column '{name}' in {path}. "
                f"Found columns: {arr.dtype.names}"
            )
        return float(arr[name])

    return (
        get("bcs"),
        get("bcs_err"),
        get("t_traj"),
        get("t_traj_err"),
        get("plaq"),
        get("plaq_err"),
        get("tau_int_plaq"),
        get("tau_int_plaq_err"),
    )

def product_with_err(a, da, b, db):
    """
    For z = a*b, assuming a and b are independent:
      dz = sqrt((b*da)^2 + (a*db)^2)
    """
    z = a * b
    dz = np.sqrt((b * da) ** 2 + (a * db) ** 2)
    return float(z), float(dz)

# ------------------------------------------------------------
# Merge MRES and HMC entries (match exactly on parsed params)
# ------------------------------------------------------------
hmc_lookup = {}
for fp in args.hmc:
    p = parse_params(fp)
    hmc_lookup[params_key(p)] = load_hmc_extract(fp)

entries = []
for fp in args.mres:
    p = parse_params(fp)
    k = params_key(p)
    if k not in hmc_lookup:
        raise ValueError(f"No matching HMC file for: {fp}")

    mres, mres_err = load_mres_fit(fp)

    (
        bcs, bcs_err,
        ttraj, ttraj_err,
        plaq, plaq_err,
        taui, taui_err
    ) = hmc_lookup[k]

    # X-axis: tau_int_plaq * t_traj (with propagated error)
    x_eff, x_eff_err = product_with_err(ttraj, ttraj_err, taui, taui_err)

    entries.append(
        {
            **p,
            "filepath": fp,
            "mres": mres,
            "mres_err": mres_err,
            "bcs": bcs,
            "bcs_err": bcs_err,
            "t_traj": ttraj,
            "t_traj_err": ttraj_err,
            "plaq": plaq,
            "plaq_err": plaq_err,
            "tau_int_plaq": taui,
            "tau_int_plaq_err": taui_err,
            "x_eff": x_eff,
            "x_eff_err": x_eff_err,
        }
    )

if len(entries) == 0:
    print("WARNING: no entries found.")
    raise SystemExit(0)

# ------------------------------------------------------------
# Choose the two beta values (ascending)
# ------------------------------------------------------------
betas = sorted({e["beta"] for e in entries})
if len(betas) < 2:
    raise ValueError(f"Need at least 2 beta values, found {betas}")
beta_vals = betas[:2]
entries_by_beta = {b: [e for e in entries if np.isclose(e["beta"], b)] for b in beta_vals}

# ------------------------------------------------------------
# CONSISTENT alpha styles across BOTH betas (as in your plot_Ls)
# ------------------------------------------------------------
all_alphas = sorted({e["alpha"] for e in entries})
non_shamir_alphas = [a for a in all_alphas if not np.isclose(a, 1.0)]

colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
markers = ["s", "^", "v", "D", "P", "X"]

alpha_style = {}
for idx, alpha in enumerate(non_shamir_alphas):
    alpha_style[alpha] = (
        markers[idx % len(markers)],
        colors[idx % len(colors)],
    )

# ------------------------------------------------------------
# Style helpers for TIME plot (match your merged script)
# ------------------------------------------------------------
LS_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

def marker_from_Ls(Ls: int) -> str:
    return LS_MARKERS[hash(Ls) % len(LS_MARKERS)]

def place_label(ax, x, y, text, family, subplot):
    """
    family: "shamir" or "mobius"
    subplot: "t" (time) or "cg" (kept for parity with original)
    """
    dx = 0.03 * (x if x != 0 else 1)
    dy_up = y * 1.06
    dy_down = y * 0.94

    if subplot == "cg":
        if family == "shamir":
            xt, yt = x - dx, dy_down
        else:
            xt, yt = x + dx, dy_up
    else:  # "t"
        if family == "shamir":
            xt, yt = x + dx, dy_up
        else:
            xt, yt = x - dx, dy_down

    ax.text(
        xt,
        yt,
        text,
        fontsize=7,
        ha="left" if xt > x else "right",
        va="bottom" if yt > y else "top",
    )

def connect(ax, xs, ys, Ls, style, color):
    if len(xs) > 1:
        _, xs_sorted, ys_sorted = zip(*sorted(zip(Ls, xs, ys)))
        ax.plot(xs_sorted, ys_sorted, style, color=color)

# ------------------------------------------------------------
# Panel 1: Ls scan 
# ------------------------------------------------------------
def plot_Ls_panel(ax, entries_panel, title):
    if len(entries_panel) == 0:
        print("WARNING: No entries for Ls scan panel")
        ax.set_title(title)
        return

    Ls_groups = defaultdict(list)
    for e in entries_panel:
        Ls_groups[e["Ls"]].append(e)

    Ls_sorted = sorted(Ls_groups.keys())

    dx = 0.12
    seen_labels = set()

    shamir_Ls, shamir_vals = [], []
    mobius_Ls, mobius_vals = [], []

    for Ls in Ls_sorted:
        group = Ls_groups[Ls]
        alphas = np.array([g["alpha"] for g in group], dtype=float)

        y_vals = np.array([g["mres"] for g in group], dtype=float)
        y_errs = np.array([g["mres_err"] for g in group], dtype=float)

        # --- Shamir
        shamir_mask = np.isclose(alphas, 1.0)
        if shamir_mask.any():
            y = y_vals[shamir_mask][0]
            e = y_errs[shamir_mask][0]

            shamir_Ls.append(Ls)
            shamir_vals.append(y)

            label = "Shamir $\\alpha=1$"
            show_label = label not in seen_labels
            seen_labels.add(label)

            ax.errorbar(
                Ls,
                y,
                yerr=e,
                fmt="o",
                color="black",
                label=label if show_label else None,
            )

        # --- Möbius points (all)
        non_indices = sorted(
            [i for i in range(len(alphas)) if not shamir_mask[i]],
            key=lambda i: alphas[i],
        )

        for local_i, idx in enumerate(non_indices):
            a = alphas[idx]
            y = y_vals[idx]
            e = y_errs[idx]

            offset = -dx + local_i * dx
            marker, color = alpha_style.get(a, ("s", "C0"))

            label = rf"$\alpha={a}$"
            show_label = label not in seen_labels
            if show_label:
                seen_labels.add(label)

            ax.errorbar(
                Ls + offset,
                y,
                yerr=e,
                fmt=marker,
                color=color,
                label=label if show_label else None,
            )

        # --- Möbius min-mres for dotted line
        if len(non_indices) > 0:
            y_non = y_vals[non_indices]
            idx_min = non_indices[int(np.argmin(y_non))]
            mobius_Ls.append(Ls)
            mobius_vals.append(y_vals[idx_min])

    if len(shamir_Ls) > 1:
        ax.plot(shamir_Ls, shamir_vals, "--", color="black")

    if len(mobius_Ls) > 1:
        ax.plot(mobius_Ls, mobius_vals, ":", color="C1")

    ax.set_title(title)
    ax.set_xlabel(r"$L_s$")
    ax.set_yscale("log")

    if show_legend:
        ax.legend(loc="upper right", ncol=2)

def make_Ls_scan_figure(outname: str):
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True, layout="constrained")

    for ax, beta in zip(axs, beta_vals):
        panel_entries = entries_by_beta[beta]
        masses = sorted({f"{e['mass']:.8g}" for e in panel_entries})
        mass_str = masses[0] if len(masses) == 1 else ",".join(masses)
        title = rf"$\beta={beta},\; am_0={mass_str}$"
        plot_Ls_panel(ax, panel_entries, title)

    axs[0].set_ylabel(r"$a m_{\rm res}$")

    ys_all = np.array([e["mres"] for e in entries], dtype=float)
    ys_all = ys_all[np.isfinite(ys_all) & (ys_all > 0)]
    if ys_all.size > 0:
        ymin = max(ys_all.min() * 0.4, 1e-12)
        ymax = ys_all.max() * 1.6
        axs[0].set_ylim(ymin, ymax)

    plt.savefig(outname, dpi=300)
    plt.close()
    print(f"Saved Ls scan plot → {outname}")

# ------------------------------------------------------------
# Panel 2: Time scan using x_eff = tau_int_plaq * t_traj
#   (ONLY Shamir + Möbius points used in dotted line)
# ------------------------------------------------------------
def select_mobius_min_points_per_Ls(entries_panel):
    Ls_groups = defaultdict(list)
    for e in entries_panel:
        Ls_groups[e["Ls"]].append(e)

    selected = []
    for Ls, group in Ls_groups.items():
        mobius = [g for g in group if not np.isclose(g["alpha"], 1.0)]
        if not mobius:
            continue
        best = min(mobius, key=lambda g: g["mres"])
        selected.append(best)
    return selected

def plot_time_panel(ax, entries_panel, title):
    if len(entries_panel) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No entries", ha="center", va="center", transform=ax.transAxes)
        return

    shamir_entries = [e for e in entries_panel if np.isclose(e["alpha"], 1.0)]
    mobius_entries = select_mobius_min_points_per_Ls(entries_panel)

    sh_x, sh_y, sh_Ls = [], [], []
    mo_x, mo_y, mo_Ls = [], [], []

    # --- Shamir: plot all
    for e in shamir_entries:
        x, dx = e["x_eff"], e["x_eff_err"]
        y, dy = e["mres"], e["mres_err"]
        Ls = e["Ls"]

        sh_x.append(x)
        sh_y.append(y)
        sh_Ls.append(Ls)

        marker = marker_from_Ls(Ls)
        label_text = rf"$L_s={Ls}$"

        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt=marker, color="black", mec="black")
        place_label(ax, x, y, label_text, "shamir", "t")

    # --- Möbius: plot ONLY the selected ones (same ones used for dotted line)
    for e in mobius_entries:
        x, dx = e["x_eff"], e["x_eff_err"]
        y, dy = e["mres"], e["mres_err"]
        Ls = e["Ls"]
        alpha = e["alpha"]

        mo_x.append(x)
        mo_y.append(y)
        mo_Ls.append(Ls)

        marker = marker_from_Ls(Ls)
        label_text = rf"$L_s={Ls}$" + "\n" + rf"$\alpha={alpha}$"

        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt=marker, color="C1", mec="C1")
        place_label(ax, x, y, label_text, "mobius", "t")

    # connect lines 
    connect(ax, sh_x, sh_y, sh_Ls, "--", "black")
    connect(ax, mo_x, mo_y, mo_Ls, ":", "C1")

    ax.set_title(title)
    ax.set_xlabel(r"$\tau_{\mathrm{int}}^{\mathrm{plaq}} \times \mathrm{Time}$ [s]")
    ax.set_yscale("log")
    #ax.set_xscale("log")

    if show_legend:
        handles = [
            plt.Line2D([], [], linestyle="--", color="black", label="Shamir, $\\alpha=1.0$"),
            plt.Line2D([], [], linestyle=":",  color="C1",    label="Möbius"),
        ]
        ax.legend(handles=handles, loc="upper right")

def make_time_scan_figure(outname: str):
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True, layout="constrained")

    # Set x-limits either from --xlim or from data (per panel)
    for ax, beta in zip(axs, beta_vals):
        panel_entries = entries_by_beta[beta]
        masses = sorted({f"{e['mass']:.8g}" for e in panel_entries})
        mass_str = masses[0] if len(masses) == 1 else ",".join(masses)
        title = rf"$\beta={beta},\; am_0={mass_str}$"
        plot_time_panel(ax, panel_entries, title)

        if args.xlim is not None:
            ax.set_xlim(args.xlim[0], args.xlim[1])
        else:
            xs = np.array([e["x_eff"] for e in panel_entries], dtype=float)
            xs = xs[np.isfinite(xs)]
            if xs.size:
                xmin = max(xs.min() * 0.2, 0.0)
                xmax = xs.max() * 1.3
                if xmax > xmin:
                    ax.set_xlim(xmin, xmax)

    axs[0].set_ylabel(r"$a m_{\rm res}$")

    ys_all = np.array([e["mres"] for e in entries], dtype=float)
    ys_all = ys_all[np.isfinite(ys_all) & (ys_all > 0)]
    if ys_all.size > 0:
        ymin = max(ys_all.min() * 0.3, 1e-12)
        ymax = ys_all.max() * 1.8
        axs[0].set_ylim(ymin, ymax)

    plt.savefig(outname, dpi=300)
    plt.close()
    print(f"Saved time scan plot → {outname}")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
make_Ls_scan_figure(args.ls_scan)
make_time_scan_figure(args.costs)
