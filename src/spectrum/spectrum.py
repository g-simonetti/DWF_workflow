#!/usr/bin/env python3
"""
spectrum.py — LQCD meson spectrum analysis using correlated fits (corrfitter).

This version keeps the overall structure of the original script, but rewrites
PP / VV / simultaneous PP+A0P fits so they follow the corrfitter/gvar.dataset
workflow more closely:

Standard fit:
    dset = ds.Dataset()
    data = ds.avg_data(dset)
    fit = fitter.lsqfit(prior=prior, data=data)

Bootstrap fit:
    bs_datalist = (ds.avg_data(d) for d in ds.bootstrap_iter(dset, n_boot))
    for bs_fit in fitter.bootstrapped_fit_iter(bs_datalist):
        p = bs_fit.pmean
        ... collect bootstrap fit outputs ...
    bs = ds.avg_data(bs, bstrap=True)

Outputs written to JSON:
  - standard_fit: correlated fit on the actual ensemble
  - bootstrap_fit: bootstrap summary for PP, VV, simultaneous PP+A0P
    including per-replica fit parameters under each fit's samples key

Z_A remains a standard correlated plateau fit as in the original code.

Bootstrap fit_stats now mirror the standard_fit structure:
  - chi2: {mean, sdev}
  - dof: integer
  - chi2_over_dof: {mean, sdev}
  - Q: null
  - logGBF: null
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import gvar as gv
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

BOOTSTRAP_DIR = Path(__file__).resolve().parents[1] / "bootstrap"
if str(BOOTSTRAP_DIR) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_DIR))

from bootstrap import bootstrap_from_path, bootstrap_to_jsonable
from fps_Z_fit import (
    bootstrap_ZA_curve,
    compute_weighted_Z_samples,
    fit_with_bootstrap_PP_A0P,
)
from ps_fit import fit_with_bootstrap_PP
from v_fit import fit_with_bootstrap_VV

plt.style.use("tableau-colorblind10")

##############################################################################
#                              DATA READERS                                  #
##############################################################################


def read_meson_corr(filename, name):
    with h5py.File(filename, "r") as f:
        data = f[f"meson/{name}/corr"][:]
        return data["re"]

def read_ps_corr(fname):
    return read_meson_corr(fname, "meson_1")

def read_vx_corr(fname):
    return read_meson_corr(fname, "meson_50")

def read_vy_corr(fname):
    return read_meson_corr(fname, "meson_83")

def read_vz_corr(fname):
    return read_meson_corr(fname, "meson_116")

def read_L_corr(fname):
    return read_meson_corr(fname, "meson_33")  # A0,P

def read_L2_corr(fname):
    return read_meson_corr(fname, "meson_9")   # P,A0

def read_R_corr(fname):
    with h5py.File(fname, "r") as f:
        return f["wardIdentity/PA0"][:]["re"]


##############################################################################
#                              FOLDING HELPERS                               #
##############################################################################


def fold_even(corr):
    """
    Fold an even correlator C(t) = C(T-t).

    Input:
        corr shape (Ncfg, T) or (T,)

    Output:
        folded correlator shape (Ncfg, Tfold) or (Tfold,)
        with times t = 0 .. T//2
    """
    corr = np.asarray(corr)
    was_1d = (corr.ndim == 1)
    if was_1d:
        corr = corr[None, :]

    N, T = corr.shape
    Th = T // 2

    folded = np.empty((N, Th + 1), dtype=corr.dtype)
    folded[:, 0] = corr[:, 0]

    if T % 2 == 0:
        for t in range(1, Th):
            folded[:, t] = 0.5 * (corr[:, t] + corr[:, T - t])
        folded[:, Th] = corr[:, Th]
    else:
        for t in range(1, Th + 1):
            folded[:, t] = 0.5 * (corr[:, t] + corr[:, T - t])

    return folded[0] if was_1d else folded


def fold_odd(corr):
    """
    Fold an odd correlator C(t) = -C(T-t).

    Input:
        corr shape (Ncfg, T) or (T,)

    Output:
        folded correlator shape (Ncfg, Tfold) or (Tfold,)
        with times t = 0 .. T//2
    """
    corr = np.asarray(corr)
    was_1d = (corr.ndim == 1)
    if was_1d:
        corr = corr[None, :]

    N, T = corr.shape
    Th = T // 2

    folded = np.empty((N, Th + 1), dtype=corr.dtype)

    # odd correlator vanishes at t=0
    folded[:, 0] = 0.0

    if T % 2 == 0:
        for t in range(1, Th):
            folded[:, t] = 0.5 * (corr[:, t] - corr[:, T - t])
        # midpoint also vanishes for exact odd symmetry
        folded[:, Th] = 0.0
    else:
        for t in range(1, Th + 1):
            folded[:, t] = 0.5 * (corr[:, t] - corr[:, T - t])

    return folded[0] if was_1d else folded


##############################################################################
#                           EFFECTIVE MASS                                   #
##############################################################################


def eff_mass_hyperbolic(C):
    """
    Effective mass from:
        cosh(m_eff(t)) = (C(t-1) + C(t+1)) / (2 C(t)).
    Defined for t = 1..T-2.
    """
    C = np.asarray(C)
    T = len(C)
    t = np.arange(1, T - 1)
    R = (C[t - 1] + C[t + 1]) / (2 * C[t])
    R = np.maximum(R, 1.0)
    return t, np.arccosh(R)


def bootstrap_effmass(corr, n_boot, boot_idx):
    """
    Bootstrap effective mass for correlators.

    Assumes corr has shape (Ncfg, T).
    Effective mass is computed on the full valid range t = 1..T-2.
    """
    corr = np.asarray(corr)
    N, T = corr.shape

    t_vals = np.arange(1, T - 1)
    nt = len(t_vals)
    m_samples = np.full((n_boot, nt), np.nan, dtype=float)

    for b in range(n_boot):
        Cb = corr[boot_idx[b]].mean(axis=0)

        num = Cb[t_vals - 1] + Cb[t_vals + 1]
        den = 2.0 * Cb[t_vals]

        R = np.full(nt, np.nan)
        good = np.abs(den) > 0
        R[good] = num[good] / den[good]

        good = good & (R >= 1.0) & np.isfinite(R)
        m_samples[b, good] = np.arccosh(R[good])

    mean = np.full(nt, np.nan)
    std = np.full(nt, np.nan)
    for i in range(nt):
        vals = m_samples[:, i]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 2:
            mean[i] = vals.mean()
            std[i] = vals.std(ddof=1)

    return t_vals, mean, std


##############################################################################
#                                  PLOTTING                                  #
##############################################################################


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _ensure_parent_dirs(files):
    for fn in files:
        Path(fn).parent.mkdir(parents=True, exist_ok=True)


def _savefig_multi(fig, outfiles):
    outfiles = _as_list(outfiles)
    _ensure_parent_dirs(outfiles)
    for fn in outfiles:
        fig.savefig(fn, dpi=300)
    plt.close(fig)


def plot_plateau(t, y, e, t0, t1, fit, dfit, outfiles, label_flag, beta, mass, qlabel, ylabel):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    data_label = rf"$\beta={beta},\ am_0={mass}$" if label_flag == "yes" else None
    fit_label = rf"${qlabel} = {fit:.5f}\pm{dfit:.5f}$"

    ax.errorbar(t, y, yerr=e, fmt="o", color="C4", label=data_label)
    ax.axvspan(t0, t1, alpha=0.2, color="C2", label="Plateau")

    ax.fill_between(
        [t0, t1],
        [fit - dfit, fit - dfit],
        [fit + dfit, fit + dfit],
        color="C1",
        alpha=0.25,
    )
    ax.hlines(fit, t0, t1, color="C1", linestyle="--", label=fit_label)

    ax.set_xlabel(r"$t/a$")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    if data_label or fit_label:
        ax.legend()

    if "PS" in qlabel or "V" in qlabel:
        ax.set_ylim(0, 1.0)

    _savefig_multi(fig, outfiles)


def plot_effmass(t, m, e, t0, t1, mfit, dfit, outfiles, label, beta, mass, chan):
    qlab = f"am_{{\\rm {chan}}}"
    plot_plateau(
        t, m, e, t0, t1, mfit, dfit,
        outfiles, label, beta, mass,
        qlab, ylabel=r"$am_{\rm eff}$"
    )


def plot_fps_two_panel(
    tps, meps, eeps,
    ta, mea0p, ea0p,
    t0, t1,
    mps, dmps, fPS, dfPS,
    outfiles, label_flag, beta, mass
):
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(3.5, 2.5),
        sharex=True, layout="constrained"
    )

    data_label = rf"$\beta={beta},\ am_0={mass}$" if label_flag == "yes" else None
    fit_label = (
        rf"$am_{{\rm PS}} = {mps:.5f}\pm{dmps:.5f}$"
        + "\n" +
        rf"$af_{{\rm PS}} = {fPS:.5f}\pm{dfPS:.5f}$"
    )

    ax0.errorbar(tps, meps, yerr=eeps, fmt="o", color="C4", label=data_label)
    ax0.axvspan(t0, t1, alpha=0.2, color="C2", label="Plateau")
    ax0.fill_between(
        [t0, t1],
        [mps - dmps, mps - dmps],
        [mps + dmps, mps + dmps],
        color="C1",
        alpha=0.25,
    )
    ax0.hlines(mps, t0, t1, color="C1", linestyle="--", label=fit_label)
    ax0.set_ylabel(r"$am_{\rm eff}^{\rm PS,PS}$")

    ax1.errorbar(ta, mea0p, yerr=ea0p, fmt="o", color="C4")
    ax1.axvspan(t0, t1, alpha=0.2, color="C2")
    ax1.fill_between(
        [t0, t1],
        [mps - dmps, mps - dmps],
        [mps + dmps, mps + dmps],
        color="C1",
        alpha=0.25,
    )
    ax1.hlines(mps, t0, t1, color="C1", linestyle="--")
    ax1.set_xlabel(r"$t/a$")
    ax1.set_ylabel(r"$am_{\rm eff}^{\rm AV,PS}$")

    for ax in (ax0, ax1):
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    if data_label or fit_label:
        ax0.legend()

    _savefig_multi(fig, outfiles)


##############################################################################
#                       FILE DISCOVERY + SELECTION                           #
##############################################################################

_NUM_PAT_PT = re.compile(r"^pt_ll\.(\d+)\.h5$")
_NUM_PAT_MR = re.compile(r"^mres\.(\d+)\.h5$")


def _extract_num(path: Path, kind: str) -> int:
    name = path.name
    m = _NUM_PAT_PT.match(name) if kind == "pt" else _NUM_PAT_MR.match(name)
    if not m:
        raise RuntimeError(f"Could not parse config number from filename: {name}")
    return int(m.group(1))


def _find_file_maps(mesons_dir: Path):
    """
    Build dicts {num -> Path} for pt_ll.num.h5 and mres.num.h5 and return the sorted
    list of numbers common to both.
    """
    pt_files = sorted(mesons_dir.glob("pt_ll.*.h5"))
    if len(pt_files) == 0:
        raise RuntimeError(f"No pt_ll files found in {mesons_dir}")

    mres_files = sorted(mesons_dir.glob("mres.*.h5"))
    if len(mres_files) == 0:
        mres_files = sorted(mesons_dir.parent.glob("mres.*.h5"))
    if len(mres_files) == 0:
        raise RuntimeError(f"No mres files found in {mesons_dir} or {mesons_dir.parent}")

    pt_map = {_extract_num(p, "pt"): p for p in pt_files}
    mr_map = {_extract_num(p, "mr"): p for p in mres_files}

    common = sorted(set(pt_map.keys()) & set(mr_map.keys()))
    if len(common) == 0:
        raise RuntimeError("No matching numbers between pt_ll.*.h5 and mres.*.h5")

    return pt_map, mr_map, common


def _select_pairs_by_number(pt_map, mr_map, common_nums, therm, delta_traj):
    """
    Select configuration numbers n such that:
      n >= therm  and  (n - therm) % delta_traj == 0
    """
    therm = int(therm)
    delta = int(delta_traj) if int(delta_traj) > 0 else 1

    nums_sel = [n for n in common_nums if (n >= therm) and ((n - therm) % delta == 0)]
    if len(nums_sel) == 0:
        lo, hi = common_nums[0], common_nums[-1]
        raise RuntimeError(
            f"Empty selection by file number: therm={therm}, delta_traj={delta}. "
            f"Available common numbers span [{lo}, {hi}] with {len(common_nums)} total."
        )

    pt_sel = [pt_map[n] for n in nums_sel]
    mr_sel = [mr_map[n] for n in nums_sel]
    return pt_sel, mr_sel, nums_sel


##############################################################################
#                                JSON HELPERS                                #
##############################################################################


def _finite_or_none(x: float) -> float | None:
    return None if (x is None or not np.isfinite(x)) else float(x)

def _gvar_to_obj(g: gv.GVar) -> Dict[str, float]:
    return {"mean": float(gv.mean(g)), "sdev": float(gv.sdev(g))}

def _chi2_over_dof(chi2: float, dof: int) -> float | None:
    if dof is None or dof <= 0:
        return None
    val = float(chi2) / float(dof)
    return _finite_or_none(val)

def _fit_stats(fit) -> Dict[str, Any]:
    return {
        "chi2": float(fit.chi2),
        "dof": int(fit.dof),
        "chi2_over_dof": _chi2_over_dof(float(fit.chi2), int(fit.dof)),
        "Q": _finite_or_none(float(getattr(fit, "Q", np.nan))),
        "logGBF": _finite_or_none(float(getattr(fit, "logGBF", np.nan))),
    }


def _summary_from_sample_list(samples, key) -> Dict[str, float] | None:
    values = np.array(
        [sample[key] for sample in samples if sample is not None and key in sample],
        dtype=float,
    )
    if values.size == 0:
        return None
    return {
        "mean": float(np.mean(values)),
        "sdev": float(np.std(values, ddof=1)) if values.size >= 2 else 0.0,
    }


def _bootstrap_fit_stats_from_samples(samples, dof: int) -> Dict[str, Any]:
    chi2 = _summary_from_sample_list(samples, "chi2")
    if chi2 is None:
        chi2_over_dof = None
    elif dof > 0:
        chi2_over_dof = {
            "mean": chi2["mean"] / float(dof),
            "sdev": chi2["sdev"] / float(dof),
        }
    else:
        chi2_over_dof = None

    return {
        "chi2": chi2,
        "dof": int(dof),
        "chi2_over_dof": chi2_over_dof,
        "Q": None,
        "logGBF": None,
    }


def _bootstrap_meta_from_samples(samples, n_requested: int) -> Dict[str, int]:
    n_success = sum(
        sample is not None and "error" not in sample
        for sample in samples
    )
    return {
        "n_requested": int(n_requested),
        "n_success": int(n_success),
        "n_failed": int(n_requested - n_success),
    }


def _bootstrap_failures_from_samples(samples):
    failures = []
    for i, sample in enumerate(samples):
        if sample is not None and "error" in sample:
            failures.append({
                "index": int(sample.get("index", i)),
                "error": str(sample["error"]),
            })
    return failures

##############################################################################
#                                   MAIN                                     #
##############################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")

    parser.add_argument("--label", default="")
    parser.add_argument("--spectrum_out", required=True)

    parser.add_argument("--plot_ps", required=True, nargs="+")
    parser.add_argument("--plot_v", required=True, nargs="+")
    parser.add_argument("--plot_fps", required=True, nargs="+")
    parser.add_argument("--plot_Z", required=True, nargs="+")

    parser.add_argument("--plot_styles", default="")

    parser.add_argument("--plateau_start_ps", type=float, required=True)
    parser.add_argument("--plateau_end_ps", type=float, required=True)
    parser.add_argument("--plateau_start_v", type=float, required=True)
    parser.add_argument("--plateau_end_v", type=float, required=True)
    parser.add_argument("--plateau_start_fps", type=float, required=True)
    parser.add_argument("--plateau_end_fps", type=float, required=True)
    parser.add_argument("--plateau_start_Z", type=float, required=True)
    parser.add_argument("--plateau_end_Z", type=float, required=True)

    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--mass", type=float, default=0)
    parser.add_argument("--Ns", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=200)
    parser.add_argument("--svdcut", type=float, default=1e-8)

    parser.add_argument("--therm", type=int, default=0)
    parser.add_argument("--delta_traj", type=int, default=1)

    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    ps0, ps1 = int(args.plateau_start_ps), int(args.plateau_end_ps)
    v0, v1 = int(args.plateau_start_v), int(args.plateau_end_v)
    fps0, fps1 = int(args.plateau_start_fps), int(args.plateau_end_fps)
    z0, z1 = int(args.plateau_start_Z), int(args.plateau_end_Z)

    mesons_dir = Path(args.input_dir)

    result: Dict[str, Any] = {
        "ok": False,
        "input": {
            "input_dir": str(mesons_dir),
            "label": args.label,
            "beta": float(args.beta),
            "mass": float(args.mass),
            "Ns": int(args.Ns),
            "therm": int(args.therm),
            "delta_traj": int(args.delta_traj),
            "n_boot": int(args.n_boot),
            "svdcut": float(args.svdcut),
            "plot_styles": args.plot_styles,
        },
        "windows": {
            "ps": {"t0": int(ps0), "t1": int(ps1)},
            "v": {"t0": int(v0), "t1": int(v1)},
            "fps": {"t0": int(fps0), "t1": int(fps1)},
            "Z": {"t0": int(z0), "t1": int(z1)},
        },
        "outputs": {
            "spectrum_out": str(args.spectrum_out),
            "plot_ps": [str(x) for x in args.plot_ps],
            "plot_v": [str(x) for x in args.plot_v],
            "plot_fps": [str(x) for x in args.plot_fps],
            "plot_Z": [str(x) for x in args.plot_Z],
        },
    }

    try:
        pt_map, mr_map, common_nums = _find_file_maps(mesons_dir)
        pt_files, mres_files, nums_used = _select_pairs_by_number(
            pt_map, mr_map, common_nums, therm=args.therm, delta_traj=args.delta_traj
        )

        print(
            f"[select] Using {len(nums_used)} configurations by file number "
            f"(therm={args.therm}, delta_traj={args.delta_traj}). "
            f"First/last used: {nums_used[0]} .. {nums_used[-1]}",
            file=sys.stderr,
        )

        result["selection"] = {
            "n_common": int(len(common_nums)),
            "nums_common_min": int(common_nums[0]),
            "nums_common_max": int(common_nums[-1]),
            "n_used": int(len(nums_used)),
            "nums_used": [int(n) for n in nums_used],
            "pt_files": [str(p) for p in pt_files],
            "mres_files": [str(p) for p in mres_files],
        }

        ps, v, a0p, Ls, Rs = [], [], [], [], []

        for fpt, fmr in zip(pt_files, mres_files):
            ps.append(read_ps_corr(str(fpt)))

            vx = read_vx_corr(str(fpt))
            vy = read_vy_corr(str(fpt))
            vz = read_vz_corr(str(fpt))
            v.append((vx + vy + vz) / 3.0)

            A0_33 = read_L_corr(str(fpt))
            A0_9 = read_L2_corr(str(fpt))
            A0_comb = 0.5 * (A0_33 + A0_9)

            a0p.append(A0_comb)
            Ls.append(A0_comb)
            Rs.append(read_R_corr(str(fmr)))

        ps = np.array(ps, dtype=float)
        v = np.array(v, dtype=float)
        a0p = np.array(a0p, dtype=float)
        Ls = np.array(Ls, dtype=float)
        Rs = np.array(Rs, dtype=float)

        # Original ensemble shape before folding
        N, T_full = ps.shape

        # Fold correlators configuration by configuration
        ps = fold_even(ps)
        v = fold_even(v)
        a0p = fold_odd(a0p)

        N, T = ps.shape
        result["data_shape"] = {
            "Ncfg": int(N),
            "T_full": int(T_full),
            "T_folded": int(T),
            "ps_folded": True,
            "v_folded": True,
            "a0p_folded": True,
            "a0p_fold_type": "odd",
            "Z_folded": False,
        }

        # Sanity checks for folded PS/V windows
        if not (0 <= ps0 < ps1 < T):
            raise RuntimeError(
                f"Invalid PS plateau window after folding: [{ps0}, {ps1}] with folded T={T}"
            )
        if not (0 <= v0 < v1 < T):
            raise RuntimeError(
                f"Invalid V plateau window after folding: [{v0}, {v1}] with folded T={T}"
            )
        if not (0 <= fps0 < fps1 < T):
            raise RuntimeError(
                f"Invalid FPS plateau window after folding: [{fps0}, {fps1}] with folded T={T}"
            )

        bootstrap = bootstrap_from_path(mesons_dir, nums_used, args.n_boot)
        boot_idx = np.asarray(bootstrap["boot_idx"], dtype=int)
        result["bootstrap"] = bootstrap_to_jsonable(bootstrap)

        # Effective masses are now computed from folded PS/V correlators
        tps, meps, eeps = bootstrap_effmass(ps, args.n_boot, boot_idx)
        tv, mev, eev = bootstrap_effmass(v, args.n_boot, boot_idx)
        ta, mea0p, ea0p = bootstrap_effmass(a0p, args.n_boot, boot_idx)

        pp_res = fit_with_bootstrap_PP(
            ps, ps0, ps1, Nt_full=T_full, n_boot=args.n_boot, boot_idx=boot_idx, svdcut=args.svdcut
        )
        vv_res = fit_with_bootstrap_VV(
            v, v0, v1, Nt_full=T_full, n_boot=args.n_boot, boot_idx=boot_idx, svdcut=args.svdcut
        )
        sim_res = fit_with_bootstrap_PP_A0P(
            ps, a0p, fps0, fps1, Ns=args.Ns, Nt_full=T_full, n_boot=args.n_boot, boot_idx=boot_idx, svdcut=args.svdcut
        )

        fit_pp = pp_res["fit"]
        fit_vv = vv_res["fit"]
        fit_sim = sim_res["fit"]
        pp_samples = pp_res["bootstrap_samples"]
        vv_samples = vv_res["bootstrap_samples"]
        sim_samples = sim_res["bootstrap_samples"]
        pp_failures = pp_res.get("bootstrap_failures", _bootstrap_failures_from_samples(pp_samples))
        vv_failures = vv_res.get("bootstrap_failures", _bootstrap_failures_from_samples(vv_samples))
        sim_failures = sim_res.get("bootstrap_failures", _bootstrap_failures_from_samples(sim_samples))

        mps_pp_gv = fit_pp.p["m_ps"][0]
        mv_gv = fit_vv.p["m_v"][0]
        mps_sim_gv = fit_sim.p["m_ps"][0]
        fPS_std_gv = -gv.sqrt(1.0) * fit_sim.p["g"][0] / gv.sqrt(fit_sim.p["m_ps"][0])
        fPS_std_gv /= (args.Ns ** 1.5)

        tZ_plot, Zt_plot, Zerr_plot = bootstrap_ZA_curve(Ls, Rs, args.n_boot, boot_idx)
        fit_Z, z_samples = compute_weighted_Z_samples(
            Ls, Rs, z0, z1, args.n_boot, boot_idx, args.svdcut
        )
        Zplat_gv = fit_Z.p["Z0"]

        # PS plot now uses folded effective mass + folded fit result
        plot_effmass(
            tps, meps, eeps, ps0, ps1,
            float(gv.mean(mps_pp_gv)), float(gv.sdev(mps_pp_gv)),
            args.plot_ps, args.label, args.beta, args.mass, "PS"
        )

        # V plot now uses folded effective mass + folded fit result
        plot_effmass(
            tv, mev, eev, v0, v1,
            float(gv.mean(mv_gv)), float(gv.sdev(mv_gv)),
            args.plot_v, args.label, args.beta, args.mass, "V"
        )

        plot_fps_two_panel(
            tps, meps, eeps,
            ta, mea0p, ea0p,
            fps0, fps1,
            float(gv.mean(mps_sim_gv)), float(gv.sdev(mps_sim_gv)),
            float(gv.mean(fPS_std_gv)), float(gv.sdev(fPS_std_gv)),
            args.plot_fps, args.label, args.beta, args.mass
        )

        tmax = T_full // 2 - 2
        mask = tZ_plot <= tmax

        plot_plateau(
            tZ_plot[mask], Zt_plot[mask], Zerr_plot[mask], z0, z1,
            float(gv.mean(Zplat_gv)), float(gv.sdev(Zplat_gv)),
            args.plot_Z, args.label, args.beta, args.mass, "Z_A", r"$Z_A$"
        )

        result["results"] = {
            "standard_fit": {
                "PP": {
                    "am_ps": _gvar_to_obj(fit_pp.p["m_ps"][0]),
                    "Afit": _gvar_to_obj(fit_pp.p["Afit"][0]),
                    "fit_stats": _fit_stats(fit_pp),
                },
                "VV": {
                    "am_v": _gvar_to_obj(fit_vv.p["m_v"][0]),
                    "AfitV": _gvar_to_obj(fit_vv.p["AfitV"][0]),
                    "fit_stats": _fit_stats(fit_vv),
                },
                "simultaneous_PP_A0P": {
                    "am_ps": _gvar_to_obj(fit_sim.p["m_ps"][0]),
                    "Afit": _gvar_to_obj(fit_sim.p["Afit"][0]),
                    "g": _gvar_to_obj(fit_sim.p["g"][0]),
                    "af_ps": _gvar_to_obj(fPS_std_gv),
                    "fit_stats": _fit_stats(fit_sim),
                },
                "Z_A": {
                    "Z_A": _gvar_to_obj(Zplat_gv),
                    "fit_stats": _fit_stats(fit_Z),
                },
            },
            "bootstrap_fit": {
                "PP": {
                    "am_ps": _summary_from_sample_list(pp_samples, "m_ps"),
                    "Afit": _summary_from_sample_list(pp_samples, "Afit"),
                    "fit_stats": pp_res["bootstrap_fit_stats"],
                    "meta": pp_res["bootstrap_meta"],
                    "samples": pp_samples,
                    "failures": pp_failures,
                },
                "VV": {
                    "am_v": _summary_from_sample_list(vv_samples, "m_v"),
                    "AfitV": _summary_from_sample_list(vv_samples, "AfitV"),
                    "fit_stats": vv_res["bootstrap_fit_stats"],
                    "meta": vv_res["bootstrap_meta"],
                    "samples": vv_samples,
                    "failures": vv_failures,
                },
                "simultaneous_PP_A0P": {
                    "am_ps": _summary_from_sample_list(sim_samples, "m_ps"),
                    "Afit": _summary_from_sample_list(sim_samples, "Afit"),
                    "g": _summary_from_sample_list(sim_samples, "g"),
                    "af_ps": _summary_from_sample_list(sim_samples, "f_ps"),
                    "fit_stats": sim_res["bootstrap_fit_stats"],
                    "meta": sim_res["bootstrap_meta"],
                    "samples": sim_samples,
                    "failures": sim_failures,
                },
                "Z_A": {
                    "Z_A": _summary_from_sample_list(z_samples, "Z_A"),
                    "Z_A_err": _summary_from_sample_list(z_samples, "Z_A_err"),
                    "fit_stats": _bootstrap_fit_stats_from_samples(z_samples, int(fit_Z.dof)),
                    "meta": _bootstrap_meta_from_samples(z_samples, args.n_boot),
                    "samples": z_samples,
                    "failures": _bootstrap_failures_from_samples(z_samples),
                },
            },
            "summary": {
                "am_ps": _gvar_to_obj(mps_sim_gv),
                "am_v": _gvar_to_obj(mv_gv),
                "af_ps": _gvar_to_obj(fPS_std_gv),
                "Z_A": _gvar_to_obj(Zplat_gv),
            },
        }

        result["ok"] = True

    except Exception as e:
        result["ok"] = False
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }

    out_path = Path(args.spectrum_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    if result.get("ok", False):
        print(
            f"[spectrum] Saved {args.spectrum_out} "
            f"with {result['selection']['n_used']} selected cfgs and "
            f"{result['bootstrap']['n_boot']} bootstrap replicas",
            file=sys.stderr,
        )
    else:
        print(
            f"[spectrum] Failed: {result['error']['type']}: {result['error']['message']}",
            file=sys.stderr,
        )

    if not result.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
