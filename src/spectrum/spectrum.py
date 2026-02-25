#!/usr/bin/env python3
"""
spectrum.py — LQCD meson spectrum analysis using correlated fits (corrfitter).

This program reads meson correlators from pt_ll.<num>.h5 and Ward-identity data
from mres.<num>.h5 and computes (with correlated fits):

  • a m_PS   (pseudoscalar mass)              from PP correlator
  • a m_V    (vector mass)                    from VV correlator (spatially averaged)
  • a f_PS   (pseudoscalar decay constant)    from a simultaneous PP + A0P fit
  • Z_A      (axial renormalisation constant) from a correlated constant plateau fit

Plots:
  • Effective masses (bootstrap mean ± std) using the 3-point cosh estimator,
    computed on the full valid range t = 1..T-2.
  • Plateau bands on the chosen fit windows.
  • Z_A(t) curve (bootstrap) and constant-fit plateau.

Configuration selection (NEW):
  The analysis uses only configurations whose *filename number* satisfies:

      num >= therm  and  (num - therm) % delta_traj == 0

  where filenames are exactly:
      pt_ll.<num>.h5
      mres.<num>.h5

  This is NOT list-index based. It uses the integer <num> parsed from filenames.

Fits are unchanged:
  • corrfitter models periodic/antiperiodic structure via tp = +T / tp = -T
  • Fit windows are inclusive: t = plateau_start .. plateau_end
"""


import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import gvar as gv
import corrfitter as cf
import lsqfit

plt.style.use("tableau-colorblind10")

##############################################################################
#                              DATA READERS                                  #
##############################################################################

def read_meson_corr(filename, name):
    with h5py.File(filename, "r") as f:
        data = f[f"meson/{name}/corr"][:]
        return data["re"]

def read_ps_corr(fname):      return read_meson_corr(fname, "meson_1")
def read_vx_corr(fname):      return read_meson_corr(fname, "meson_50")
def read_vy_corr(fname):      return read_meson_corr(fname, "meson_83")
def read_vz_corr(fname):      return read_meson_corr(fname, "meson_116")
def read_L_corr(fname):       return read_meson_corr(fname, "meson_33")  # A0,P
def read_L2_corr(fname):      return read_meson_corr(fname, "meson_9")   # P,A0

def read_R_corr(fname):
    with h5py.File(fname, "r") as f:
        return f["wardIdentity/PA0"][:]["re"]

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
    Bootstrap effective mass for UNFOLDED correlators.

    Assumes corr has shape (Ncfg, T).
    Effective mass is computed on the full valid range t = 1..T-2 (needs t±1 defined).
    """
    corr = np.asarray(corr)
    N, T = corr.shape

    t_vals = np.arange(1, T - 1)  # 1..T-2
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
    std  = np.full(nt, np.nan)
    for i in range(nt):
        vals = m_samples[:, i]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 2:
            mean[i] = vals.mean()
            std[i]  = vals.std(ddof=1)

    return t_vals, mean, std

##############################################################################
#                      CORRFITTER FIT HELPERS                                #
##############################################################################

def _guess_m_from_effmass(cmean, t0, t1):
    teff, meff = eff_mass_hyperbolic(cmean)
    mask = (teff >= t0 + 1) & (teff <= t1 - 1)
    m0 = float(np.mean(meff[mask])) if np.any(mask) else 0.1
    if not np.isfinite(m0) or m0 <= 0:
        m0 = 0.1
    return m0

def _guess_Afit_from_corr(cmean, m0, t_ref):
    """
    For symmetric correlator:
      C(t) ~ Afit^2 * exp(-m t)  (ignoring backward term)
    So Afit ~ sqrt( |C(t)| * exp(m t) ).
    """
    t_ref = int(t_ref)
    t_ref = max(0, min(t_ref, len(cmean) - 1))
    C = float(cmean[t_ref])
    A2 = abs(C) * np.exp(m0 * t_ref)
    return float(np.sqrt(max(A2, 1e-16)))

def fit_PP_only(ps_samples, t0, t1, svdcut=1e-8):
    """
    Correlated PP fit:
      C_PP(t) = Afit^2 * (e^{-mt} + e^{-m(T-t)})
    Fit region is exactly t = t0..t1 inclusive.
    """
    N, T = ps_samples.shape
    data = gv.dataset.avg_data({"PP": ps_samples})
    cmean = ps_samples.mean(axis=0)

    m0 = _guess_m_from_effmass(cmean, t0, t1)
    A0 = _guess_Afit_from_corr(cmean, m0, t_ref=t0)

    prior = gv.BufferDict()
    prior["log(Afit)"] = gv.gvar([np.log(A0)], [3.0])
    prior["log(m_ps)"] = gv.gvar([np.log(m0)], [3.0])

    model = cf.Corr2(
        "PP",
        a="Afit", b="Afit", dE="m_ps",
        tp=+T,
        tdata=np.arange(T),
        tfit=np.arange(t0, t1 + 1),
    )

    return cf.CorrFitter(models=[model]).lsqfit(data=data, prior=prior, svdcut=svdcut)

def fit_VV_only(v_samples, t0, t1, svdcut=1e-8):
    """
    Correlated VV fit:
      C_VV(t) = AfitV^2 * (e^{-m_v t} + e^{-m_v(T-t)})
    Fit region is exactly t = t0..t1 inclusive.
    """
    N, T = v_samples.shape
    data = gv.dataset.avg_data({"VV": v_samples})
    cmean = v_samples.mean(axis=0)

    m0 = _guess_m_from_effmass(cmean, t0, t1)
    A0 = _guess_Afit_from_corr(cmean, m0, t_ref=t0)

    prior = gv.BufferDict()
    prior["log(AfitV)"] = gv.gvar([np.log(A0)], [3.0])
    prior["log(m_v)"]   = gv.gvar([np.log(m0)], [3.0])

    model = cf.Corr2(
        "VV",
        a="AfitV", b="AfitV", dE="m_v",
        tp=+T,
        tdata=np.arange(T),
        tfit=np.arange(t0, t1 + 1),
    )

    return cf.CorrFitter(models=[model]).lsqfit(data=data, prior=prior, svdcut=svdcut)

def fit_simultaneous_PP_A0P(ps_samples, a0p_samples, t0, t1, svdcut=1e-8):
    """
    Simultaneous correlated fit over SAME window [t0,t1] for BOTH PP and A0P.

    PP:  C_PP(t)  = Afit^2 * (exp(-mt) + exp(-m(T-t)))
    A0P: C_A0P(t) = (Afit*g) * (exp(-mt) - exp(-m(T-t)))
    => f_PS = sqrt(2) * g / sqrt(m)
    Fit region is exactly t = t0..t1 inclusive.
    """
    N, T = ps_samples.shape
    data = gv.dataset.avg_data({"PP": ps_samples, "A0P": a0p_samples})
    cmean = ps_samples.mean(axis=0)

    m0 = _guess_m_from_effmass(cmean, t0, t1)
    A0 = _guess_Afit_from_corr(cmean, m0, t_ref=t0)

    prior = gv.BufferDict()
    prior["log(Afit)"] = gv.gvar([np.log(A0)], [3.0])
    prior["log(m_ps)"] = gv.gvar([np.log(m0)], [3.0])
    prior["g"]         = gv.gvar([0.0], [10.0])

    m_pp = cf.Corr2(
        "PP",
        a="Afit", b="Afit", dE="m_ps",
        tp=+T,
        tdata=np.arange(T),
        tfit=np.arange(t0, t1 + 1),
    )

    m_a0p = cf.Corr2(
        "A0P",
        a="Afit", b="g", dE="m_ps",
        tp=-T,
        tdata=np.arange(T),
        tfit=np.arange(t0, t1 + 1),
    )

    return cf.CorrFitter(models=[m_pp, m_a0p]).lsqfit(data=data, prior=prior, svdcut=svdcut)

##############################################################################
#                                   Z_A                                     #
##############################################################################

def compute_ZA(L, R):
    L = np.asarray(L)
    R = np.asarray(R)
    T = len(L)
    tvals = np.arange(1, T - 1)  # 1..T-2
    Z = np.zeros_like(tvals, float)

    for i, t in enumerate(tvals):
        Rf = R[t]
        Rb = R[t - 1]
        Lt = L[t]
        Ltp = L[t + 1]

        term1 = (Rf + Rb) / (2 * Lt)
        term2 = 2 * Rf / (Lt + Ltp)
        Z[i] = 0.5 * (term1 + term2)
    return tvals, Z

def build_Z_samples(Lcorr, Rcorr):
    """
    Build per-configuration Z_A(t) samples WITHOUT folding.

    Returns:
      tvals: t = 1..T-2
      Z_samples: shape (Ncfg, T-2)
    """
    Lcorr = np.asarray(Lcorr)
    Rcorr = np.asarray(Rcorr)
    if Lcorr.shape != Rcorr.shape or Lcorr.ndim != 2:
        raise ValueError("build_Z_samples expects Lcorr and Rcorr with shape (Ncfg, T)")

    N, T = Lcorr.shape
    tvals, _ = compute_ZA(Lcorr[0], Rcorr[0])  # 1..T-2
    nt = len(tvals)

    Z_samples = np.empty((N, nt), dtype=float)
    for i in range(N):
        _, Zi = compute_ZA(Lcorr[i], Rcorr[i])
        Z_samples[i] = Zi

    return tvals, Z_samples

def fit_Z_only(Z_samples, t0, t1, svdcut=1e-8):
    """
    Correlated constant fit to Z(t) on [t0, t1].
    Z_samples: (Ncfg, nt) with nt = T-2 corresponding to t=1..T-2.
    Fit region is exactly t = t0..t1 inclusive (in that t=1..T-2 domain).
    """
    Z_samples = np.asarray(Z_samples)
    N, nt = Z_samples.shape
    tvals = np.arange(1, nt + 1)  # corresponds to t=1..T-2

    Zg = gv.dataset.avg_data({"Z": Z_samples})["Z"]

    mask = (tvals >= t0) & (tvals <= t1)
    if not np.any(mask):
        raise ValueError(f"fit_Z_only: empty fit window [{t0},{t1}] for t=1..{nt}")

    y = Zg[mask]

    prior = gv.BufferDict()
    prior["Z0"] = gv.gvar(1.0, 10.0)

    def fcn(p):
        return p["Z0"] * np.ones(len(y))

    return lsqfit.nonlinear_fit(data=y, prior=prior, fcn=fcn, svdcut=svdcut)

def bootstrap_ZA_curve(Lcorr, Rcorr, n_boot, boot_idx):
    """
    Bootstrap mean/std of Z_A(t) for plotting WITHOUT folding.
    """
    Lcorr = np.asarray(Lcorr)
    Rcorr = np.asarray(Rcorr)
    N, T = Lcorr.shape

    tvals, _ = compute_ZA(Lcorr[0], Rcorr[0])  # 1..T-2
    nt = len(tvals)

    Zb = np.empty((n_boot, nt), dtype=float)

    for b in range(n_boot):
        idx = boot_idx[b]
        Lm = Lcorr[idx].mean(axis=0)
        Rm = Rcorr[idx].mean(axis=0)
        _, Zt = compute_ZA(Lm, Rm)
        Zb[b] = Zt

    return tvals, Zb.mean(axis=0), Zb.std(axis=0, ddof=1)

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
    fit_label  = rf"${qlabel} = {fit:.5f}\pm{dfit:.5f}$"

    ax.errorbar(t, y, yerr=e, fmt="o", color="C4", label=data_label)
    ax.axvspan(t0, t1, alpha=0.2, color="C2", label="Plateau")

    ax.fill_between([t0, t1], [fit - dfit, fit - dfit], [fit + dfit, fit + dfit],
                    color="C1", alpha=0.25)
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
    ta,  mea0p, ea0p,
    t0, t1,
    mps, dmps, fPS, dfPS,
    outfiles, label_flag, beta, mass
):
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(3.5, 2.5),
        sharex=True, layout="constrained"
    )

    data_label = rf"$\beta={beta},\ am_0={mass}$" if label_flag == "yes" else None
    fit_label  = (
        rf"$am_{{\rm PS}} = {mps:.5f}\pm{dmps:.5f}$"
        + "\n" +
        rf"$af_{{\rm PS}} = {fPS:.5f}\pm{dfPS:.5f}$"
    )

    ax0.errorbar(tps, meps, yerr=eeps, fmt="o", color="C4", label=data_label)
    ax0.axvspan(t0, t1, alpha=0.2, color="C2", label="Plateau")
    ax0.fill_between([t0, t1], [mps - dmps, mps - dmps], [mps + dmps, mps + dmps],
                     color="C1", alpha=0.25)
    ax0.hlines(mps, t0, t1, color="C1", linestyle="--", label=fit_label)
    ax0.set_ylabel(r"$am_{\rm eff}^{\rm PS,PS}$")

    ax1.errorbar(ta,  mea0p, yerr=ea0p, fmt="o", color="C4")
    ax1.axvspan(t0, t1, alpha=0.2, color="C2")
    ax1.fill_between([t0, t1], [mps - dmps, mps - dmps], [mps + dmps, mps + dmps],
                     color="C1", alpha=0.25)
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

##############################################################################
#                                   MAIN                                     #
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")  # in rule: .../mesons

    parser.add_argument("--label", default="")
    parser.add_argument("--spectrum_out", required=True)

    # multiext outputs => Snakemake passes multiple filenames; accept 1 or many
    parser.add_argument("--plot_ps",  required=True, nargs="+")
    parser.add_argument("--plot_v",   required=True, nargs="+")
    parser.add_argument("--plot_fps", required=True, nargs="+")
    parser.add_argument("--plot_Z",   required=True, nargs="+")

    parser.add_argument("--plot_styles", default="")

    parser.add_argument("--plateau_start_ps", type=float, required=True)
    parser.add_argument("--plateau_end_ps",   type=float, required=True)
    parser.add_argument("--plateau_start_v",  type=float, required=True)
    parser.add_argument("--plateau_end_v",    type=float, required=True)
    parser.add_argument("--plateau_start_fps", type=float, required=True)
    parser.add_argument("--plateau_end_fps",   type=float, required=True)
    parser.add_argument("--plateau_start_Z",  type=float, required=True)
    parser.add_argument("--plateau_end_Z",    type=float, required=True)

    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--mass", type=float, default=0)
    parser.add_argument("--Ns", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--svdcut", type=float, default=1e-8)

    # selection by filename number
    parser.add_argument("--therm", type=int, default=0)
    parser.add_argument("--delta_traj", type=int, default=1)

    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    ps0, ps1   = int(args.plateau_start_ps),   int(args.plateau_end_ps)
    v0,  v1    = int(args.plateau_start_v),    int(args.plateau_end_v)
    fps0, fps1 = int(args.plateau_start_fps),  int(args.plateau_end_fps)
    z0,  z1    = int(args.plateau_start_Z),    int(args.plateau_end_Z)

    mesons_dir = Path(args.input_dir)

    # Prepare JSON shell early (so failures still produce JSON)
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
            "ps":  {"t0": int(ps0),  "t1": int(ps1)},
            "v":   {"t0": int(v0),   "t1": int(v1)},
            "fps": {"t0": int(fps0), "t1": int(fps1)},
            "Z":   {"t0": int(z0),   "t1": int(z1)},
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

        # Human log -> stderr (stdout stays JSON)
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
            v.append((vx + vy + vz) / 3)

            A0_33 = read_L_corr(str(fpt))
            A0_9  = read_L2_corr(str(fpt))
            A0_comb = 0.5 * (A0_33 + A0_9)

            a0p.append(A0_comb)
            Ls.append(A0_comb)
            Rs.append(read_R_corr(str(fmr)))

        ps  = np.array(ps)
        v   = np.array(v)
        a0p = np.array(a0p)
        Ls  = np.array(Ls)
        Rs  = np.array(Rs)

        N, T = ps.shape
        result["data_shape"] = {"Ncfg": int(N), "T": int(T)}

        rng = np.random.default_rng()
        boot_idx = rng.integers(0, N, size=(args.n_boot, N))

        # Effective masses for plots (we do not dump full arrays by default)
        tps, meps, eeps  = bootstrap_effmass(ps,  args.n_boot, boot_idx)
        tv,  mev,  eev   = bootstrap_effmass(v,   args.n_boot, boot_idx)
        ta,  mea0p, ea0p = bootstrap_effmass(a0p, args.n_boot, boot_idx)

        # Fits
        fit_pp  = fit_PP_only(ps, ps0, ps1, svdcut=args.svdcut)
        fit_vv  = fit_VV_only(v,  v0,  v1,  svdcut=args.svdcut)
        fit_sim = fit_simultaneous_PP_A0P(ps, a0p, fps0, fps1, svdcut=args.svdcut)

        # Masses for plot labels (PP-only for PS plot; SIM for fPS plot)
        mps_pp_gv = fit_pp.p["m_ps"][0]
        mps_sim_gv = fit_sim.p["m_ps"][0]
        mv_gv = fit_vv.p["m_v"][0]

        # f_PS
        g_gv = fit_sim.p["g"][0]
        fPS_gv = gv.sqrt(2.0) * g_gv / gv.sqrt(mps_sim_gv)

        fPS_mean = float(gv.mean(fPS_gv))
        fPS_sdev = float(gv.sdev(fPS_gv))

        if args.Ns <= 0:
            raise RuntimeError("--Ns must be > 0 (needed for f_PS normalisation).")

        # Keep original normalisation/sign
        fPS_mean = -fPS_mean / (args.Ns ** 1.5)
        fPS_sdev =  fPS_sdev / (args.Ns ** 1.5)

        # chi2/dof
        chi2ps  = _chi2_over_dof(float(fit_pp.chi2),  int(fit_pp.dof))
        chi2v   = _chi2_over_dof(float(fit_vv.chi2),  int(fit_vv.dof))
        chi2fps = _chi2_over_dof(float(fit_sim.chi2), int(fit_sim.dof))

        # Z_A
        _, Z_samples = build_Z_samples(Ls, Rs)
        tZ_plot, Zt_plot, Zerr_plot = bootstrap_ZA_curve(Ls, Rs, args.n_boot, boot_idx)

        fit_Z = fit_Z_only(Z_samples, z0, z1, svdcut=args.svdcut)
        chi2Z = _chi2_over_dof(float(fit_Z.chi2), int(fit_Z.dof))
        Zplat_gv = fit_Z.p["Z0"]

        # Plots
        plot_effmass(
            tps, meps, eeps, ps0, ps1,
            float(gv.mean(mps_pp_gv)), float(gv.sdev(mps_pp_gv)),
            args.plot_ps, args.label, args.beta, args.mass, "PS"
        )

        plot_effmass(
            tv,  mev,  eev,  v0,  v1,
            float(gv.mean(mv_gv)), float(gv.sdev(mv_gv)),
            args.plot_v, args.label, args.beta, args.mass, "V"
        )

        plot_fps_two_panel(
            tps, meps, eeps,
            ta,  mea0p, ea0p,
            fps0, fps1,
            float(gv.mean(mps_sim_gv)), float(gv.sdev(mps_sim_gv)),
            fPS_mean, fPS_sdev,
            args.plot_fps, args.label, args.beta, args.mass
        )

        # Force x-range to t <= T/2 - 2 (same as your original code)
        tmax = T // 2 - 2
        mask = tZ_plot <= tmax

        plot_plateau(
            tZ_plot[mask], Zt_plot[mask], Zerr_plot[mask], z0, z1,
            float(gv.mean(Zplat_gv)), float(gv.sdev(Zplat_gv)),
            args.plot_Z, args.label, args.beta, args.mass, "Z_A", r"$Z_A$"
        )

        # Fill JSON results
        result["results"] = {
            "am_ps": _gvar_to_obj(mps_sim_gv),
            "am_v":  _gvar_to_obj(mv_gv),
            "af_ps": {"mean": float(fPS_mean), "sdev": float(fPS_sdev)},
            "Z_A":   _gvar_to_obj(Zplat_gv),
            "chi2_over_dof": {
                "ps": chi2ps,
                "v": chi2v,
                "fps": chi2fps,
                "Z": chi2Z,
            },
        }

        result["ok"] = True

    except Exception as e:
        # Store error as JSON
        result["ok"] = False
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }

    # Write JSON output file (single source of truth)
    out_path = Path(args.spectrum_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    # Print JSON to stdout (and keep human logs on stderr)
    print(json.dumps(result, indent=2, sort_keys=True))

    # Exit nonzero if failed (Snakemake-friendly)
    if not result.get("ok", False):
        raise SystemExit(2)

if __name__ == "__main__":
    main()