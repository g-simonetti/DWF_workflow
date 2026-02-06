#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict

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
        "2) Cost scan: m_res vs (tau_int_plaq * t_traj) with propagated x-error\n"
        "   where t_traj comes from log_hmc_extract.txt and tau_int_plaq is the plaquette autocorrelation time.\n"
        "   COST FIGURE is 2x2: top row Möbius, bottom row Shamir; left beta=7.4, right beta=7.6 (if present).\n"
    )
)

parser.add_argument("--mres", nargs="+", required=True, help="m_res fit files (one per ensemble).")
parser.add_argument("--hmc", nargs="+", required=True, help="log_hmc_extract.txt files (one per ensemble).")
parser.add_argument("--metadata", required=True)  # kept for interface parity

parser.add_argument("--label", default="no")
parser.add_argument("--plot_styles", default=None)

parser.add_argument("--ls_scan", required=True, help="Output PDF for m_res vs Ls (2 betas).")
parser.add_argument("--costs", required=True, help="Output PDF for m_res vs (tau_int_plaq * t_traj) (2 betas).")

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
def load_mres_fit(path: str):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        return float(arr[0]), float(arr[1])
    return float(arr[0, 0]), float(arr[0, 1])


def load_hmc_extract(path: str):
    """
    Reads ONE-row log_hmc_extract.txt produced by your extractor.

    Expected columns include at least:
      t_traj, t_traj_err, tau_int_plaq, tau_int_plaq_err
    """
    arr = np.genfromtxt(path, names=True, dtype=None, encoding=None)

    # If multiple rows were passed, take the first row
    if getattr(arr, "ndim", 0) != 0:
        arr = arr[0]

    names = arr.dtype.names

    def get(name: str) -> float:
        if name not in names:
            raise ValueError(
                f"Missing column '{name}' in {path}. Found columns: {names}"
            )
        return float(arr[name])

    t_traj = get("t_traj")
    t_traj_err = get("t_traj_err")
    tau_int_plaq = get("tau_int_plaq")
    tau_int_plaq_err = get("tau_int_plaq_err")

    # optional
    bcs = float(arr["bcs"]) if "bcs" in names else np.nan
    bcs_err = float(arr["bcs_err"]) if "bcs_err" in names else np.nan
    plaq = float(arr["plaq"]) if "plaq" in names else np.nan
    plaq_err = float(arr["plaq_err"]) if "plaq_err" in names else np.nan

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

    mres, mres_err = load_mres_fit(fp)
    h = hmc_lookup[k]

    x_eff, x_eff_err = product_with_err(
        h["t_traj"], h["t_traj_err"],
        h["tau_int_plaq"], h["tau_int_plaq_err"],
    )

    entries.append(
        {
            **p,
            "filepath": fp,
            "mres": mres,
            "mres_err": mres_err,
            **h,
            "x_eff": x_eff,
            "x_eff_err": x_eff_err,
        }
    )

if len(entries) == 0:
    print("WARNING: no entries found.")
    raise SystemExit(0)


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
LS_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]


def marker_from_Ls(Ls: int) -> str:
    return LS_MARKERS[hash(Ls) % len(LS_MARKERS)]


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
# Panel 2: Cost scan (2x2: top Möbius, bottom Shamir)
# ============================================================
def plot_cost_mobius(ax, entries_panel, beta):
    # yscale MUST be set before gradient connector (transform depends on it)
    ax.set_yscale("log")

    mobius_entries = select_mobius_min_points_per_Ls(entries_panel)

    mo_x, mo_y, mo_Ls = [], [], []

    for e in mobius_entries:
        x, dx = e["x_eff"], e["x_eff_err"]
        y, dy = e["mres"], e["mres_err"]
        Ls = e["Ls"]
        alpha = e["alpha"]

        mo_x.append(x); mo_y.append(y); mo_Ls.append(Ls)

        marker = marker_from_Ls(Ls)
        color = alpha_to_color.get(alpha, "C1")
        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt=marker, color=color, mec=color)

        label_text = rf"$L_s={Ls}$" + "\n" + rf"$\alpha={alpha}$"
        place_label(ax, x, y, label_text, "mobius", "mobius", Ls=Ls, beta=beta)

    # Gradient dotted connector like in first plot, but ordered by Ls
    add_viridis_gradient_dotted_line_straight_by_Ls(ax, mo_x, mo_y, mo_Ls, n_per_segment=120, linewidth=0.6)

    ax.set_title(rf"$\beta={beta}\; am_0={_mass_str(entries_panel)}$")

    if show_legend:
        ax.legend(
            handles=[plt.Line2D([], [], linestyle=":", color="C1",
                                label="Möbius")],
            loc="upper right"
        )


def plot_cost_shamir(ax, entries_panel, beta):
    ax.set_yscale("log")

    shamir_entries = [e for e in entries_panel if np.isclose(e["alpha"], 1.0)]

    sh_x, sh_y, sh_Ls = [], [], []

    for e in shamir_entries:
        x, dx = e["x_eff"], e["x_eff_err"]
        y, dy = e["mres"], e["mres_err"]
        Ls = e["Ls"]

        sh_x.append(x); sh_y.append(y); sh_Ls.append(Ls)

        marker = marker_from_Ls(Ls)
        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt=marker, color="black", mec="black")

        label_text = rf"$L_s={Ls}$"
        place_label(ax, x, y, label_text, "shamir", "shamir", Ls=Ls, beta=beta)

    connect_sorted_by_Ls(ax, sh_x, sh_y, sh_Ls, "--", "black")

    if show_legend:
        ax.legend(
            handles=[plt.Line2D([], [], linestyle="--", color="black",
                                label="Shamir, $\\alpha=1.0$")],
            loc="upper right"
        )


def make_cost_scan_figure(outname: str):
    fig, axs = plt.subplots(
        2, 2,
        figsize=(7, 3.5),
        sharex=True,
        sharey=True,
        layout="constrained"
    )

    # ---- Set limits early so gradient connector uses final transform ----
    # Your requested x-range:
    for ax in axs.ravel():
        ax.set_xlim(50.0, 5700.0)

    ys_all = np.array([e["mres"] for e in entries], dtype=float)
    ys_all = ys_all[np.isfinite(ys_all) & (ys_all > 0)]
    if ys_all.size > 0:
        ymin = max(ys_all.min() * 0.4, 1e-12)
        ymax = ys_all.max() * 2.0
        for ax in axs.ravel():
            ax.set_ylim(ymin, ymax)

    # ---- Plot panels ----
    for j, beta in enumerate(beta_vals):
        panel_entries = entries_by_beta[beta]
        plot_cost_mobius(axs[0, j], panel_entries, beta)  # top row
        plot_cost_shamir(axs[1, j], panel_entries, beta)  # bottom row

    # Labels
    axs[1, 0].set_xlabel(r"$\tau_{\mathrm{int}}^{\mathrm{plaq}} \times t_{\mathrm{traj}}\;[\mathrm{s}]$")
    axs[1, 1].set_xlabel(r"$\tau_{\mathrm{int}}^{\mathrm{plaq}} \times t_{\mathrm{traj}}\;[\mathrm{s}]$")
    axs[0, 0].set_ylabel(r"$a m_{\rm res}$")
    axs[1, 0].set_ylabel(r"$a m_{\rm res}$")

    plt.savefig(outname, dpi=300)
    plt.close()
    print(f"Saved cost scan plot → {outname}")


# ============================================================
# Run
# ============================================================
make_Ls_scan_figure(args.ls_scan)
make_cost_scan_figure(args.costs)
