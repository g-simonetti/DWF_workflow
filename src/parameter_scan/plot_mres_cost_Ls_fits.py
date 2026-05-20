#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from scipy.optimize import OptimizeWarning, curve_fit

plt.style.use("tableau-colorblind10")


# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser(
    description=(
        "Two-beta plotting utility with fixed-nu m_res(Ls) fits.\n"
        "1) Ls scan: m_res vs Ls (Shamir + all Möbius points; fit curves use Shamir nu=1 and Möbius nu=2)\n"
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


def mres_fit_ansatz(Ls, c1, lambda_c, c2, nu):
    Ls = np.asarray(Ls, dtype=float)
    return c1 * np.exp(-lambda_c * Ls) + c2 / np.power(Ls, nu)


def guess_mres_fit_parameters(Ls, y, nu):
    Ls = np.asarray(Ls, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(Ls)
    Ls = Ls[order]
    y = y[order]

    tail_count = min(2, len(Ls))
    c2_guess = float(np.median(y[-tail_count:] * np.power(Ls[-tail_count:], nu)))
    c2_guess = max(c2_guess, 0.0)

    residual = np.maximum(y - c2_guess / np.power(Ls, nu), 1e-12)
    c1_guess = float(max(np.max(residual), y[0], 1e-12))

    if len(Ls) > 1 and not np.isclose(Ls[-1], Ls[0]):
        lambda_guess = np.log(residual[0] / residual[-1]) / (Ls[-1] - Ls[0])
    else:
        lambda_guess = 0.1
    lambda_guess = float(np.clip(lambda_guess, 1e-6, 5.0))

    return [c1_guess, lambda_guess, c2_guess]


def fit_mres_vs_Ls(Ls, y, yerr, nu):
    Ls = np.asarray(Ls, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    finite = np.isfinite(Ls) & np.isfinite(y) & np.isfinite(yerr)
    Ls = Ls[finite]
    y = y[finite]
    yerr = yerr[finite]

    if len(Ls) < 3:
        return None

    sigma_floor = np.maximum(np.abs(y) * 1e-3, 1e-12)
    sigma = np.where(yerr > 0, yerr, sigma_floor)

    p0 = guess_mres_fit_parameters(Ls, y, nu)
    model = lambda x, c1, lambda_c, c2: mres_fit_ansatz(x, c1, lambda_c, c2, nu)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                model,
                Ls,
                y,
                p0=p0,
                sigma=sigma,
                absolute_sigma=True,
                bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
                maxfev=20000,
            )
    except Exception:
        return None

    diag = np.diag(pcov) if pcov is not None else np.array([np.nan, np.nan, np.nan])
    perr = np.sqrt(diag) if np.all(np.isfinite(diag)) else np.full(3, np.nan)

    y_fit = model(Ls, *popt)
    chi2 = float(np.sum(((y - y_fit) / sigma) ** 2))
    dof = int(len(Ls) - len(popt))

    return {
        "nu": nu,
        "params": popt,
        "errors": perr,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": (chi2 / dof) if dof > 0 else np.nan,
        "Ls_min": float(np.min(Ls)),
        "Ls_max": float(np.max(Ls)),
    }


def print_fit_summary(beta, family, fit):
    if fit is None:
        print(f"WARNING: beta={beta} {family} fit failed or had fewer than 3 points.")
        return

    c1, lambda_c, c2 = fit["params"]
    dc1, dlambda_c, dc2 = fit["errors"]
    chi2_dof = fit["chi2_dof"]
    chi2_text = f"{chi2_dof:.3g}" if np.isfinite(chi2_dof) else "n/a"
    print(
        f"beta={beta} {family} fit (nu={fit['nu']}): "
        f"c1={c1:.6g} +/- {dc1:.2g}, "
        f"lambda_c={lambda_c:.6g} +/- {dlambda_c:.2g}, "
        f"c2={c2:.6g} +/- {dc2:.2g}, "
        f"chi2/dof={chi2_text}"
    )


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

BLACK_GREY_CMAP = LinearSegmentedColormap.from_list("black_grey", ["black", "0.65"])
SHAMIR_MARKER_GREY = "0.25"


def build_shamir_shade_map(entries_panel):
    shamir_Ls = sorted({e["Ls"] for e in entries_panel if np.isclose(e["alpha"], 1.0)})
    if not shamir_Ls:
        return {}

    if len(shamir_Ls) == 1:
        shade_positions = [1.0]
    else:
        shade_positions = np.linspace(1.0, 0.0, len(shamir_Ls))

    return {Ls: BLACK_GREY_CMAP(t) for Ls, t in zip(shamir_Ls, shade_positions)}

def add_gradient_fit_curve(ax, xs, ys, cmap, linewidth=0.8, linestyle="solid", reverse=False, zorder=1):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]

    if len(xs) < 2:
        return

    points = np.column_stack([xs, ys])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    tcol = np.linspace(1.0, 0.0, len(segments)) if reverse else np.linspace(0.0, 1.0, len(segments))
    lc = LineCollection(
        segments,
        cmap=plt.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap,
        array=tcol,
        linewidths=linewidth,
        linestyles=linestyle,
        capstyle="round",
        zorder=zorder,
    )
    ax.add_collection(lc)


def add_gradient_connector_straight(ax, xs, ys, cmap, linewidth=0.6, linestyle=":", reverse=False, zorder=1):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]

    if len(xs) < 2:
        return

    trans = ax.transData
    inv = trans.inverted()
    segs_all = []

    for i in range(len(xs) - 1):
        p0_disp = trans.transform((xs[i], ys[i]))
        p1_disp = trans.transform((xs[i + 1], ys[i + 1]))

        t = np.linspace(0.0, 1.0, 120, endpoint=False)
        pts_disp = p0_disp[None, :] + (p1_disp - p0_disp)[None, :] * t[:, None]
        if i == len(xs) - 2:
            pts_disp = np.vstack([pts_disp, p1_disp[None, :]])

        pts_data = inv.transform(pts_disp)
        segs_all.append(np.stack([pts_data[:-1], pts_data[1:]], axis=1))

    segs_all = np.concatenate(segs_all, axis=0)
    tcol = np.linspace(1.0, 0.0, len(segs_all)) if reverse else np.linspace(0.0, 1.0, len(segs_all))
    lc = LineCollection(
        segs_all,
        cmap=plt.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap,
        array=tcol,
        linewidths=linewidth,
        linestyles=linestyle,
        capstyle="round",
        zorder=zorder,
    )
    ax.add_collection(lc)


def make_legend_errorbar(ax, marker, color, include_xerr=True, include_yerr=True):
    kwargs = {
        "fmt": marker,
        "color": color,
        "ecolor": color,
        "mec": color,
        "linestyle": "None",
    }
    if include_xerr:
        kwargs["xerr"] = [1.0]
    if include_yerr:
        kwargs["yerr"] = [1.0]
    return ax.errorbar([np.nan], [np.nan], **kwargs)




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


def plot_Ls_panel(ax, entries_panel, title, beta):
    if len(entries_panel) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No entries", ha="center", va="center", transform=ax.transAxes)
        return

    alpha_style = build_alpha_style_for_panel(entries_panel)

    # Pre-register Möbius legend entries (with errorbar glyph)
    for a in sorted(alpha_style.keys()):
        marker, color = alpha_style[a]
        label = rf"$\alpha={a}$"
        handle = make_legend_errorbar(ax, marker, color, include_xerr=False, include_yerr=True)
        handle.set_label(label)

    Ls_groups = defaultdict(list)
    for e in entries_panel:
        Ls_groups[e["Ls"]].append(e)
    Ls_sorted = sorted(Ls_groups.keys())

    dx = 0.12
    seen_labels = set()

    shamir_Ls, shamir_vals, shamir_errs = [], [], []
    mobius_Ls, mobius_vals, mobius_errs = [], [], []

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
            shamir_errs.append(e)

            label = r"Shamir $\alpha=1$"
            show_label = label not in seen_labels
            seen_labels.add(label)
            ax.errorbar(
                Ls,
                y,
                yerr=e,
                fmt="o",
                color=SHAMIR_MARKER_GREY,
                ecolor=SHAMIR_MARKER_GREY,
                mec=SHAMIR_MARKER_GREY,
                label=label if show_label else None,
                zorder=3,
            )

        non_indices = sorted([i for i in range(len(alphas)) if not shamir_mask[i]], key=lambda i: alphas[i])
        for local_i, idx in enumerate(non_indices):
            a = float(alphas[idx])
            y = float(y_vals[idx])
            e = float(y_errs[idx])

            offset = -dx + local_i * dx
            marker, color = alpha_style.get(a, ("s", "C0"))
            ax.errorbar(Ls + offset, y, yerr=e, fmt=marker, color=color, ecolor=color, mec=color, zorder=3)

        if len(non_indices) > 0:
            y_non = y_vals[non_indices]
            idx_min = non_indices[int(np.argmin(y_non))]
            mobius_Ls.append(Ls)
            mobius_vals.append(float(y_vals[idx_min]))
            mobius_errs.append(float(y_errs[idx_min]))

    ax.set_title(title)
    ax.set_xlabel(r"$L_s$")
    ax.set_yscale("log")

    shamir_fit = fit_mres_vs_Ls(shamir_Ls, shamir_vals, shamir_errs, nu=1)
    mobius_fit = fit_mres_vs_Ls(mobius_Ls, mobius_vals, mobius_errs, nu=2)
    fit_legend_handles = []
    fit_legend_labels = []

    print_fit_summary(beta, "Shamir", shamir_fit)
    print_fit_summary(beta, "Mobius-min", mobius_fit)

    if shamir_fit is not None:
        Ls_fit = np.linspace(shamir_fit["Ls_min"], shamir_fit["Ls_max"], 400)
        y_fit = mres_fit_ansatz(Ls_fit, *shamir_fit["params"], nu=1)
        add_gradient_fit_curve(ax, Ls_fit, y_fit, cmap=BLACK_GREY_CMAP, linewidth=0.8, linestyle="--", reverse=True, zorder=1)
        if show_legend:
            fit_legend_handles.append(plt.Line2D([], [], linestyle="--", color="0.35", linewidth=0.8))
            fit_legend_labels.append(r"Shamir fit ($\nu=1$)")

    if mobius_fit is not None:
        Ls_fit = np.linspace(mobius_fit["Ls_min"], mobius_fit["Ls_max"], 400)
        y_fit = mres_fit_ansatz(Ls_fit, *mobius_fit["params"], nu=2)
        add_gradient_fit_curve(ax, Ls_fit, y_fit, cmap="viridis_r", linewidth=0.8, zorder=1)
        if show_legend:
            fit_legend_handles.append(
                plt.Line2D([], [], linestyle="-", color=plt.colormaps.get_cmap("viridis_r")(0.5), linewidth=0.8)
            )
            fit_legend_labels.append(r"Mobius min fit ($\nu=2$)")

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        legend_loc = "lower left" if np.isclose(beta, 7.4) else "upper right"
        ax.legend(
            handles + fit_legend_handles,
            labels + fit_legend_labels,
            loc=legend_loc,
            fontsize=5.8,
            ncol=2,
            frameon=True,
            borderpad=0.25,
            handletextpad=0.3,
            labelspacing=0.25,
            columnspacing=0.8,
        )


def make_Ls_scan_figure(outname: str):
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True, layout="constrained")

    for ax, beta in zip(axs, beta_vals):
        panel_entries = entries_by_beta[beta]
        title = rf"$\beta={beta},\; am_0={_mass_str(panel_entries)}$"
        plot_Ls_panel(ax, panel_entries, title, beta)

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
    alpha_style = build_alpha_style_for_panel(entries_panel)

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

        ax.errorbar(
            x,
            y,
            xerr=dx,
            yerr=dy,
            fmt="o",
            color=SHAMIR_MARKER_GREY,
            ecolor=SHAMIR_MARKER_GREY,
            mec=SHAMIR_MARKER_GREY,
            zorder=3,
        )

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

        marker, color = alpha_style.get(alpha, ("s", "C1"))
        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt=marker, color=color, ecolor=color, mec=color, zorder=3)

    add_gradient_connector_straight(ax, sh_x, sh_y, cmap=BLACK_GREY_CMAP, linewidth=0.6, linestyle=":", reverse=True, zorder=1)
    add_gradient_connector_straight(ax, mo_x, mo_y, cmap="viridis_r", linewidth=0.6, linestyle=":", zorder=1)

    if title is not None:
        ax.set_title(title, pad=plt.rcParams["axes.titlepad"])
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def add_panel_alpha_legend(ax, entries_panel):
    mobius_entries = sorted(select_mobius_min_points_per_Ls(entries_panel), key=lambda e: e["Ls"])
    alpha_style = build_alpha_style_for_panel(entries_panel)
    beta_values = sorted({e["beta"] for e in entries_panel})
    beta = beta_values[0] if beta_values else np.nan
    has_shamir = any(np.isclose(e["alpha"], 1.0) for e in entries_panel)
    if not mobius_entries and not has_shamir:
        return

    handles = []
    labels = []
    if has_shamir:
        handles.append(make_legend_errorbar(ax, "o", SHAMIR_MARKER_GREY, include_xerr=False, include_yerr=True))
        labels.append(r"Shamir ($\alpha=1$)")

    panel_alphas = sorted({e["alpha"] for e in mobius_entries})
    for alpha in panel_alphas:
        marker, color = alpha_style.get(alpha, ("s", "C1"))
        handles.append(make_legend_errorbar(ax, marker, color, include_xerr=False, include_yerr=True))
        labels.append(rf"$\alpha={alpha}$")

    legend_loc = "lower left" if np.isclose(beta, 7.4) else "upper right"
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc=legend_loc,
        fontsize=5.8,
        ncol=2,
        frameon=True,
        borderpad=0.25,
        handletextpad=0.3,
        labelspacing=0.25,
        columnspacing=0.8,
    )
    legend.get_frame().set_alpha(0.95)


def make_cost_scan_figure(outname: str):
    fig, axs = plt.subplots(
        3, 2,
        figsize=(7, 4.0),
        sharex="col",
        sharey="row",
        layout="constrained"
    )
    fig.set_constrained_layout_pads(hspace=0.02, h_pad=0.02)

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
        add_panel_alpha_legend(axs[0, j], panel_entries)
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

    ttraj_vals = np.array([e["t_traj"] for e in entries], dtype=float)
    ttraj_errs = np.array([e["t_traj_err"] for e in entries], dtype=float)
    ttraj_mask = np.isfinite(ttraj_vals) & np.isfinite(ttraj_errs)
    if np.any(ttraj_mask):
        ttraj_top = np.max(ttraj_vals[ttraj_mask] + ttraj_errs[ttraj_mask])
        for ax in axs[1, :]:
            ax.set_ylim(0.0, ttraj_top * 1.05)

    bcs_vals = np.array([e["bcs"] for e in entries], dtype=float)
    bcs_errs = np.array([e["bcs_err"] for e in entries], dtype=float)
    bcs_mask = np.isfinite(bcs_vals)
    if np.any(bcs_mask):
        bcs_top = np.max(bcs_vals[bcs_mask] + np.where(np.isfinite(bcs_errs[bcs_mask]), bcs_errs[bcs_mask], 0.0))
        for ax in axs[0, :]:
            ax.set_ylim(0.0, bcs_top * 1.05)

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
