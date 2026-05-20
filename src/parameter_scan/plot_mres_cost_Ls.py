#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

plt.style.use("tableau-colorblind10")


# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser(
    description=(
        "Two-beta plotting utility.\n"
        "1) Ls scan: m_res vs Ls (Shamir + all Möbius points; Möbius line uses min m_res per Ls)\n"
        "2) Cost scan: x-axis is m_res and uses the minimum-m_res Möbius alpha per Ls.\n"
        "   For each beta there are three stacked panels: top is full bcs, middle is t_traj, bottom is (tau_int_ptll * t_traj).\n"
        "   Shamir and Möbius are overlaid in the same subplot; left beta=7.4, right beta=7.6 (if present).\n"
    )
)

parser.add_argument("--mres", nargs="+", required=True, help="m_res.json files (one per ensemble).")
parser.add_argument("--hmc", nargs="+", required=True, help="log_hmc_extract.txt files (one per ensemble).")

parser.add_argument("--label", default="no")
parser.add_argument("--plot_styles", default=None)

parser.add_argument("--ls_scan", required=True, help="Output PDF for m_res vs Ls (2 betas).")
parser.add_argument("--costs", required=True, help="Output PDF for combined cost/bcs vs m_res figure.")

args = parser.parse_args()
show_legend = args.label.lower() == "yes"

if args.plot_styles:
    plt.style.use(args.plot_styles)


# ============================================================
# Path parser
# ============================================================
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


# ============================================================
# Loaders
# ============================================================
def load_mres_extract(path: str):
    """
    Read plateau-extracted residual mass from m_res.json:
      data["mres_extract"]["value"], data["mres_extract"]["error"]
      data["mres_extract"]["ptll_tau_int"]["tau_int"], ["tau_int_err"]
    """
    with open(path, "r") as f:
        data = json.load(f)

    try:
        y = data["mres_extract"]["value"]
        err = data["mres_extract"]["error"]
        ptll_tau = data["mres_extract"]["ptll_tau_int"]["tau_int"]
        ptll_tau_err = data["mres_extract"]["ptll_tau_int"]["tau_int_err"]
    except Exception as e:
        raise ValueError(
            f"{path}: expected JSON with mres_extract.value/error and mres_extract.ptll_tau_int.tau_int/tau_int_err"
        ) from e

    return float(y), float(err), float(ptll_tau), float(ptll_tau_err)


def load_hmc_extract(path: str):
    """
    Reads log_hmc_extract.json produced by the new extractor.

    Expected keys:
      data["hmc_extract"]["t_traj"]
      data["hmc_extract"]["t_traj_err"]

    Optional:
      bcs, bcs_err, plaq, plaq_err, tau_int_plaq, tau_int_plaq_err (if present)
    """
    with open(path, "r") as f:
        data = json.load(f)

    if "hmc_extract" not in data or not isinstance(data["hmc_extract"], dict):
        raise ValueError(f"{path}: expected JSON with top-level 'hmc_extract' object")

    h = data["hmc_extract"]

    def get(name: str, required: bool = True, default=np.nan) -> float:
        if name in h and h[name] is not None:
            return float(h[name])
        if required:
            raise ValueError(f"{path}: missing required hmc_extract.{name}")
        return float(default)

    t_traj = get("t_traj", required=True)
    t_traj_err = get("t_traj_err", required=True)
    # optional
    tau_int_plaq = get("tau_int_plaq", required=False)
    tau_int_plaq_err = get("tau_int_plaq_err", required=False)
    bcs = get("bcs", required=False)
    bcs_err = get("bcs_err", required=False)
    plaq = get("plaq", required=False)
    plaq_err = get("plaq_err", required=False)

    return {
        "t_traj": t_traj,
        "t_traj_err": t_traj_err,
        "tau_int_plaq": tau_int_plaq,
        "tau_int_plaq_err": tau_int_plaq_err,
        "bcs": bcs,
        "bcs_err": bcs_err,
        "plaq": plaq,
        "plaq_err": plaq_err,
    }


def product_with_err(a, da, b, db):
    """
    For z = a*b, assuming a and b are independent:
      dz = sqrt((b*da)^2 + (a*db)^2)
    """
    z = a * b
    dz = np.sqrt((b * da) ** 2 + (a * db) ** 2)
    return float(z), float(dz)


# ============================================================
# Merge MRES and HMC entries (match exactly on parsed params)
# ============================================================
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

    mres, mres_err, tau_int_ptll, tau_int_ptll_err = load_mres_extract(fp)
    h = hmc_lookup[k]

    x_eff, x_eff_err = product_with_err(
        h["t_traj"], h["t_traj_err"],
        tau_int_ptll, tau_int_ptll_err,
    )

    entries.append(
        {
            **p,
            "filepath": fp,
            "mres": mres,
            "mres_err": mres_err,
            "tau_int_ptll": tau_int_ptll,
            "tau_int_ptll_err": tau_int_ptll_err,
            **h,
            "x_eff": x_eff,
            "x_eff_err": x_eff_err,
        }
    )

if len(entries) == 0:
    print("WARNING: no entries found.")
    raise SystemExit(0)

ls_marker_order = sorted({e["Ls"] for e in entries})


# ============================================================
# Choose the two beta values: prefer (7.4, 7.6) if present
# ============================================================
betas_present = sorted({e["beta"] for e in entries})

preferred = [7.4, 7.6]
if all(any(np.isclose(b, bp) for bp in betas_present) for b in preferred):
    beta_vals = preferred
else:
    if len(betas_present) < 2:
        raise ValueError(f"Need at least 2 beta values, found {betas_present}")
    beta_vals = betas_present[:2]

entries_by_beta = {b: [e for e in entries if np.isclose(e["beta"], b)] for b in beta_vals}


# ============================================================
# Helpers
# ============================================================
LS_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def marker_from_Ls(Ls: int) -> str:
    if Ls in ls_marker_order:
        return LS_MARKERS[ls_marker_order.index(Ls) % len(LS_MARKERS)]
    return LS_MARKERS[int(Ls) % len(LS_MARKERS)]


def _mass_str(entries_panel) -> str:
    masses = sorted({f"{e['mass']:.8g}" for e in entries_panel})
    return masses[0] if len(masses) == 1 else ",".join(masses)


def place_label(ax, x, y, text, family, panel, Ls=None, beta=None):
    """
    Your custom label placement rules:
      - Möbius: special-case certain (Ls, beta) to avoid overlaps
      - Shamir: Ls 8,12,24 -> top-right; Ls 32 -> bottom-left; others default bottom-left
    """
    dx = 0.03 * (x if x != 0 else 1.0)
    y_up = y * 1.10
    y_dn = y * 0.90

    if panel == "mobius":
        # custom exceptions you added
        if (Ls == 16 and np.isclose(beta, 7.6)) or (Ls in {12, 32} and np.isclose(beta, 7.4)):
            xt, yt = x - dx, y_dn
            ha, va = "right", "top"
        else:
            xt, yt = x + dx, y_up
            ha, va = "left", "bottom"

    else:  # shamir
        if Ls in {8, 12, 24}:
            xt, yt = x + dx, y_up
            ha, va = "left", "bottom"
        elif Ls == 32:
            xt, yt = x - dx, y_dn
            ha, va = "right", "top"
        else:
            xt, yt = x - dx, y_dn
            ha, va = "right", "top"

    ax.text(xt, yt, text, fontsize=7, ha=ha, va=va)


def connect_sorted_by_Ls(ax, xs, ys, Ls, style, color):
    if len(xs) > 1:
        _, xs_sorted, ys_sorted = zip(*sorted(zip(Ls, xs, ys)))
        ax.plot(xs_sorted, ys_sorted, style, color=color)


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


def finite_pair(x, y):
    return np.isfinite(x) and np.isfinite(y)


# ============================================================
# Colour maps (modern Matplotlib API, no get_cmap deprecation)
# ============================================================
# For cost plot: alpha -> color (viridis_r)
all_alphas = sorted({e["alpha"] for e in entries})
non_shamir_alphas = [a for a in all_alphas if not np.isclose(a, 1.0)]
alpha_to_color = {}
if non_shamir_alphas:
    sorted_alphas = sorted(non_shamir_alphas)
    norm = Normalize(vmin=min(sorted_alphas), vmax=max(sorted_alphas))
    cmap_alpha = plt.colormaps.get_cmap("viridis_r").resampled(len(sorted_alphas))
    for a in sorted_alphas:
        alpha_to_color[a] = cmap_alpha(norm(a))


# ============================================================
# Gradient dotted connectors that stay straight on log axes
# ============================================================
def add_viridis_gradient_dotted_line_straight(ax, xs, ys, n_per_segment=120, linewidth=0.6):
    """
    Straight-looking dotted line with smooth viridis gradient, even on log axes.
    Sorts by x (used in Ls scan where x is Ls, so fine).
    """
    if len(xs) < 2:
        return

    xs_sorted, ys_sorted = zip(*sorted(zip(xs, ys)))
    xs_sorted = np.asarray(xs_sorted, dtype=float)
    ys_sorted = np.asarray(ys_sorted, dtype=float)

    # IMPORTANT: use the current transform (so call after scales/limits are set)
    trans = ax.transData
    inv = trans.inverted()

    n_orig = len(xs_sorted) - 1
    segs_all = []

    for i in range(n_orig):
        p0_disp = trans.transform((xs_sorted[i],     ys_sorted[i]))
        p1_disp = trans.transform((xs_sorted[i + 1], ys_sorted[i + 1]))

        t = np.linspace(0.0, 1.0, n_per_segment, endpoint=False)
        pts_disp = p0_disp[None, :] + (p1_disp - p0_disp)[None, :] * t[:, None]

        if i == n_orig - 1:
            pts_disp = np.vstack([pts_disp, p1_disp[None, :]])

        pts_data = inv.transform(pts_disp)
        segs = np.stack([pts_data[:-1], pts_data[1:]], axis=1)
        segs_all.append(segs)

    segs_all = np.concatenate(segs_all, axis=0)
    tcol = np.linspace(0.0, 1.0, segs_all.shape[0])

    lc = LineCollection(
        segs_all,
        cmap=plt.cm.viridis_r,
        array=tcol,
        linewidths=linewidth,
        linestyles=":",
    )
    ax.add_collection(lc)


def add_viridis_gradient_dotted_line_straight_by_Ls(ax, xs, ys, Ls, n_per_segment=120, linewidth=0.6):
    """
    Same as above, but connects points in increasing Ls order (for cost plot).
    """
    if len(xs) < 2:
        return

    _, xs_sorted, ys_sorted = zip(*sorted(zip(Ls, xs, ys)))
    xs_sorted = np.asarray(xs_sorted, dtype=float)
    ys_sorted = np.asarray(ys_sorted, dtype=float)

    trans = ax.transData
    inv = trans.inverted()

    n_orig = len(xs_sorted) - 1
    segs_all = []

    for i in range(n_orig):
        p0_disp = trans.transform((xs_sorted[i],     ys_sorted[i]))
        p1_disp = trans.transform((xs_sorted[i + 1], ys_sorted[i + 1]))

        t = np.linspace(0.0, 1.0, n_per_segment, endpoint=False)
        pts_disp = p0_disp[None, :] + (p1_disp - p0_disp)[None, :] * t[:, None]

        if i == n_orig - 1:
            pts_disp = np.vstack([pts_disp, p1_disp[None, :]])

        pts_data = inv.transform(pts_disp)
        segs = np.stack([pts_data[:-1], pts_data[1:]], axis=1)
        segs_all.append(segs)

    segs_all = np.concatenate(segs_all, axis=0)
    tcol = np.linspace(0.0, 1.0, segs_all.shape[0])

    lc = LineCollection(
        segs_all,
        cmap=plt.cm.viridis_r,
        array=tcol,
        linewidths=linewidth,
        linestyles=":",
    )
    ax.add_collection(lc)


# ============================================================
# Panel 1: Ls scan
# ============================================================
def build_alpha_style_for_panel(entries_panel):
    """
    Möbius points: viridis_r across unique alphas in THIS panel (excluding alpha=1).
    Markers: cycle locally within this panel.
    """
    markers = ["s", "^", "v", "D", "P", "X"]
    panel_alphas = sorted({e["alpha"] for e in entries_panel})
    non_shamir = [a for a in panel_alphas if not np.isclose(a, 1.0)]

    style = {}
    if not non_shamir:
        return style

    n = len(non_shamir)
    cmap_local = plt.colormaps.get_cmap("viridis_r").resampled(n)
    for i, a in enumerate(sorted(non_shamir)):
        style[a] = (markers[i % len(markers)], cmap_local(i))
    return style


def plot_Ls_panel(ax, entries_panel, title):
    if len(entries_panel) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No entries", ha="center", va="center", transform=ax.transAxes)
        return

    alpha_style = build_alpha_style_for_panel(entries_panel)

    # Pre-register Möbius legend entries (with errorbar glyph)
    for a in sorted(alpha_style.keys()):
        marker, color = alpha_style[a]
        label = rf"$\alpha={a}$"
        ax.errorbar([np.nan], [np.nan], yerr=[1.0], fmt=marker, color=color, ecolor=color,
                    linestyle="None", label=label)

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

        shamir_mask = np.isclose(alphas, 1.0)
        if shamir_mask.any():
            y = float(y_vals[shamir_mask][0])
            e = float(y_errs[shamir_mask][0])

            shamir_Ls.append(Ls)
            shamir_vals.append(y)

            label = r"Shamir $\alpha=1$"
            show_label = label not in seen_labels
            seen_labels.add(label)

            ax.errorbar(Ls, y, yerr=e, fmt="o", color="black", label=label if show_label else None)

        non_indices = sorted([i for i in range(len(alphas)) if not shamir_mask[i]], key=lambda i: alphas[i])
        for local_i, idx in enumerate(non_indices):
            a = float(alphas[idx])
            y = float(y_vals[idx])
            e = float(y_errs[idx])

            offset = -dx + local_i * dx
            marker, color = alpha_style.get(a, ("s", "C0"))
            ax.errorbar(Ls + offset, y, yerr=e, fmt=marker, color=color)

        if len(non_indices) > 0:
            y_non = y_vals[non_indices]
            idx_min = non_indices[int(np.argmin(y_non))]
            mobius_Ls.append(Ls)
            mobius_vals.append(float(y_vals[idx_min]))

    ax.set_title(title)
    ax.set_xlabel(r"$L_s$")
    ax.set_yscale("log")

    if len(shamir_Ls) > 1:
        ax.plot(shamir_Ls, shamir_vals, "--", color="black")

    # IMPORTANT: call after yscale is set
    add_viridis_gradient_dotted_line_straight(ax, mobius_Ls, mobius_vals, n_per_segment=120, linewidth=0.6)

    if show_legend:
        ax.legend(loc="best", ncol=2)


def make_Ls_scan_figure(outname: str):
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True, layout="constrained")

    for ax, beta in zip(axs, beta_vals):
        panel_entries = entries_by_beta[beta]
        title = rf"$\beta={beta},\; am_0={_mass_str(panel_entries)}$"
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


# ============================================================
# Panel 2: Combined trajectory/cost/bcs scan (rows are observables, columns are betas)
# ============================================================
def plot_combined_metric_panel(
    ax,
    entries_panel,
    beta,
    y_key,
    y_err_key=None,
    ylabel=None,
    title=None,
):
    ax.set_xscale("log")

    shamir_entries = sorted(
        [e for e in entries_panel if np.isclose(e["alpha"], 1.0)],
        key=lambda e: e["Ls"]
    )
    mobius_entries = sorted(select_mobius_min_points_per_Ls(entries_panel), key=lambda e: e["Ls"])

    sh_x, sh_y, sh_Ls = [], [], []
    mo_x, mo_y, mo_Ls = [], [], []

    for e in shamir_entries:
        x, dx = e["mres"], e["mres_err"]
        y = e[y_key]
        dy = e[y_err_key] if y_err_key is not None else None
        Ls = e["Ls"]

        if not finite_pair(x, y):
            continue

        sh_x.append(x)
        sh_y.append(y)
        sh_Ls.append(Ls)

        marker = marker_from_Ls(Ls)
        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt=marker, color="black", mec="black")

    for e in mobius_entries:
        x, dx = e["mres"], e["mres_err"]
        y = e[y_key]
        dy = e[y_err_key] if y_err_key is not None else None
        Ls = e["Ls"]
        alpha = e["alpha"]

        if not finite_pair(x, y):
            continue

        mo_x.append(x)
        mo_y.append(y)
        mo_Ls.append(Ls)

        marker = marker_from_Ls(Ls)
        color = alpha_to_color.get(alpha, "C1")
        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt=marker, color=color, mec=color)

    connect_sorted_by_Ls(ax, sh_x, sh_y, sh_Ls, "--", "black")
    add_viridis_gradient_dotted_line_straight_by_Ls(ax, mo_x, mo_y, mo_Ls, n_per_segment=120, linewidth=0.6)

    if title is not None:
        ax.set_title(title, pad=28)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def add_panel_mobius_legend(ax, entries_panel):
    mobius_entries = sorted(select_mobius_min_points_per_Ls(entries_panel), key=lambda e: e["Ls"])
    if not mobius_entries:
        return

    handles = []
    labels = []
    for e in mobius_entries:
        Ls = e["Ls"]
        alpha = e["alpha"]
        color = alpha_to_color.get(alpha, "C1")
        handles.append(
            ax.errorbar(
                [np.nan], [np.nan],
                xerr=[1.0], yerr=[1.0],
                fmt=marker_from_Ls(Ls),
                color=color,
                mec=color,
                linestyle="None",
            )
        )
        labels.append(rf"$L_s={Ls},\ \alpha={alpha}$")

    legend = ax.legend(
        handles=handles,
        labels=labels,
        title="Möbius",
        loc="upper right",
        fontsize=6.2,
        title_fontsize=6.5,
        ncol=1,
        frameon=True,
        borderpad=0.3,
        handletextpad=0.35,
        labelspacing=0.3,
    )
    legend.get_frame().set_alpha(0.95)


def add_shared_shamir_legend(fig, top_axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    top_positions = [ax.get_position() for ax in top_axes]
    title_boxes = [
        ax.title.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        for ax in top_axes
    ]

    legend_left = min(pos.x0 for pos in top_positions)
    legend_right = max(pos.x1 for pos in top_positions)
    axes_top = max(pos.y1 for pos in top_positions)
    title_bottom = min(box.y0 for box in title_boxes)

    gap = max(title_bottom - axes_top, 0.06)
    legend_bottom = axes_top + 0.003
    legend_height = max(gap - 0.008, 0.05)

    legend_ax = fig.add_axes(
        [legend_left, legend_bottom, legend_right - legend_left, legend_height],
        frameon=False,
    )
    legend_ax.set_axis_off()

    shamir_entries = sorted({e["Ls"] for e in entries if np.isclose(e["alpha"], 1.0)})

    shamir_handles = [
        legend_ax.errorbar(
            [np.nan], [np.nan],
            xerr=[1.0], yerr=[1.0],
            fmt=marker_from_Ls(Ls),
            color="black",
            mec="black",
            linestyle="None",
        )
        for Ls in shamir_entries
    ]
    inline_header = plt.Line2D([], [], linestyle="None", marker=None, alpha=0.0)
    shamir_handles = [inline_header] + shamir_handles
    shamir_labels = [r"Shamir $\alpha=1$"] + [rf"$L_s={Ls}$" for Ls in shamir_entries]

    legend_shamir = legend_ax.legend(
        handles=shamir_handles,
        labels=shamir_labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        fontsize=6.5,
        ncol=6,
        frameon=True,
        borderpad=0.35,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.35,
    )
    legend_shamir.get_frame().set_alpha(0.95)
    legend_shamir._legend_box.align = "left"


def make_cost_scan_figure(outname: str):
    fig, axs = plt.subplots(
        3, 2,
        figsize=(7, 5.45),
        sharex="col",
        sharey="row",
        layout="constrained"
    )

    xs_all = np.array([e["mres"] for e in entries], dtype=float)
    xs_all = xs_all[np.isfinite(xs_all) & (xs_all > 0)]
    if xs_all.size > 0:
        xmin = max(xs_all.min() * 0.6, 1e-12)
        xmax = xs_all.max() * 1.8
        for ax in axs.ravel():
            ax.set_xlim(xmin, xmax)

    for j, beta in enumerate(beta_vals):
        panel_entries = entries_by_beta[beta]
        title = rf"$\beta={beta}\; am_0={_mass_str(panel_entries)}$"
        plot_combined_metric_panel(
            axs[0, j],
            panel_entries,
            beta,
            y_key="bcs",
            y_err_key="bcs_err",
            ylabel=r"Full bcs" if j == 0 else None,
            title=title,
        )
        add_panel_mobius_legend(axs[0, j], panel_entries)
        plot_combined_metric_panel(
            axs[1, j],
            panel_entries,
            beta,
            y_key="t_traj",
            y_err_key="t_traj_err",
            ylabel=r"$t_{\mathrm{traj}}\;[\mathrm{s}]$" if j == 0 else None,
        )
        plot_combined_metric_panel(
            axs[2, j],
            panel_entries,
            beta,
            y_key="x_eff",
            y_err_key="x_eff_err",
            ylabel=r"$\tau_{\mathrm{int}}^{\mathrm{PS}} \times t_{\mathrm{traj}}\;[\mathrm{s}]$" if j == 0 else None,
        )

    add_shared_shamir_legend(fig, axs[0, :])

    ttraj_vals = np.array([e["t_traj"] for e in entries], dtype=float)
    ttraj_errs = np.array([e["t_traj_err"] for e in entries], dtype=float)
    ttraj_mask = np.isfinite(ttraj_vals) & np.isfinite(ttraj_errs)
    if np.any(ttraj_mask):
        ttraj_top = np.max(ttraj_vals[ttraj_mask] + ttraj_errs[ttraj_mask])
        for ax in axs[1, :]:
            ax.set_ylim(0.0, ttraj_top * 1.05)

    axs[2, 0].set_xlabel(r"$a m_{\rm res}$")
    axs[2, 1].set_xlabel(r"$a m_{\rm res}$")

    plt.savefig(outname, dpi=300)
    plt.close()
    print(f"Saved combined trajectory/cost/bcs plot → {outname}")


# ============================================================
# Run
# ============================================================
make_Ls_scan_figure(args.ls_scan)
make_cost_scan_figure(args.costs)
