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
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import gvar as gv
import gvar.dataset as ds
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
    Bootstrap effective mass for unfolded correlators.

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


def _dataset_from_samples(sample_map):
    """
    Build a gvar.dataset.Dataset from arrays with shape (Ncfg, T).

    Each configuration is appended as one Monte Carlo sample, matching the
    corrfitter/gvar.dataset workflow more closely.
    """
    keys = list(sample_map.keys())
    arrays = {}

    ref_shape = None
    for k, v in sample_map.items():
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{k} must have shape (Ncfg, T), got {arr.shape}")
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f"All sample arrays must have the same shape; "
                f"{k} has {arr.shape}, expected {ref_shape}"
            )
        arrays[k] = arr

    ncfg, _ = ref_shape
    dset = ds.Dataset()
    for icfg in range(ncfg):
        for k in keys:
            dset.append(k, arrays[k][icfg])
    return dset


def _fit_stats(fit) -> Dict[str, Any]:
    return {
        "chi2": float(fit.chi2),
        "dof": int(fit.dof),
        "chi2_over_dof": _chi2_over_dof(float(fit.chi2), int(fit.dof)),
        "Q": _finite_or_none(float(getattr(fit, "Q", np.nan))),
        "logGBF": _finite_or_none(float(getattr(fit, "logGBF", np.nan))),
    }


def _bootstrap_fit_stats(chi2_samples, dof: int) -> Dict[str, Any]:
    """
    Bootstrap analogue of _fit_stats with the same JSON structure where possible.

    Since bootstrap fits produce an ensemble of chi2 values, chi2 and chi2_over_dof
    are reported as {mean, sdev}. Q and logGBF are left as null.
    """
    chi2_samples = np.asarray(chi2_samples, dtype=float)
    chi2_samples = chi2_samples[np.isfinite(chi2_samples)]

    if chi2_samples.size == 0:
        return {
            "chi2": None,
            "dof": int(dof),
            "chi2_over_dof": None,
            "Q": None,
            "logGBF": None,
        }

    chi2_mean = float(np.mean(chi2_samples))
    chi2_sdev = float(np.std(chi2_samples, ddof=1)) if chi2_samples.size >= 2 else 0.0

    if dof is not None and dof > 0:
        chi2_over_dof = {
            "mean": chi2_mean / float(dof),
            "sdev": chi2_sdev / float(dof),
        }
    else:
        chi2_over_dof = None

    return {
        "chi2": {"mean": chi2_mean, "sdev": chi2_sdev},
        "dof": int(dof),
        "chi2_over_dof": chi2_over_dof,
        "Q": None,
        "logGBF": None,
    }


def make_models_PP(T, t0, t1):
    return [
        cf.Corr2(
            datatag="PP",
            a="Afit", b="Afit", dE="m_ps",
            tp=+T,
            tdata=np.arange(T),
            tfit=np.arange(t0, t1 + 1),
        )
    ]


def make_models_VV(T, t0, t1):
    return [
        cf.Corr2(
            datatag="VV",
            a="AfitV", b="AfitV", dE="m_v",
            tp=+T,
            tdata=np.arange(T),
            tfit=np.arange(t0, t1 + 1),
        )
    ]


def make_models_PP_A0P(T, t0, t1):
    return [
        cf.Corr2(
            datatag="PP",
            a="Afit", b="Afit", dE="m_ps",
            tp=+T,
            tdata=np.arange(T),
            tfit=np.arange(t0, t1 + 1),
        ),
        cf.Corr2(
            datatag="A0P",
            a="Afit", b="g", dE="m_ps",
            tp=-T,
            tdata=np.arange(T),
            tfit=np.arange(t0, t1 + 1),
        ),
    ]


def make_prior_PP(ps_samples, t0, t1):
    cmean = np.mean(ps_samples, axis=0)
    m0 = _guess_m_from_effmass(cmean, t0, t1)
    A0 = _guess_Afit_from_corr(cmean, m0, t_ref=t0)

    prior = gv.BufferDict()
    prior["log(Afit)"] = gv.gvar([np.log(A0)], [100.0])
    prior["log(m_ps)"] = gv.gvar([np.log(m0)], [100.0])
    return prior


def make_prior_VV(v_samples, t0, t1):
    cmean = np.mean(v_samples, axis=0)
    m0 = _guess_m_from_effmass(cmean, t0, t1)
    A0 = _guess_Afit_from_corr(cmean, m0, t_ref=t0)

    prior = gv.BufferDict()
    prior["log(AfitV)"] = gv.gvar([np.log(A0)], [100.0])
    prior["log(m_v)"] = gv.gvar([np.log(m0)], [100.0])
    return prior


def make_prior_PP_A0P(ps_samples, t0, t1):
    cmean = np.mean(ps_samples, axis=0)
    m0 = _guess_m_from_effmass(cmean, t0, t1)
    A0 = _guess_Afit_from_corr(cmean, m0, t_ref=t0)

    prior = gv.BufferDict()
    prior["log(Afit)"] = gv.gvar([np.log(A0)], [100.0])
    prior["log(m_ps)"] = gv.gvar([np.log(m0)], [100.0])
    prior["g"] = gv.gvar([0.0], [100.0])
    return prior


def fit_with_bootstrap_PP(ps_samples, t0, t1, n_boot=200, svdcut=1e-8):
    """
    Standard + bootstrap correlated fit for PP using corrfitter.
    """
    ps_samples = np.asarray(ps_samples, dtype=float)
    _, T = ps_samples.shape

    dset = _dataset_from_samples({"PP": ps_samples})
    data = ds.avg_data(dset)

    models = make_models_PP(T, t0, t1)
    prior = make_prior_PP(ps_samples, t0, t1)
    fitter = cf.CorrFitter(models=models)

    fit = fitter.lsqfit(prior=prior, data=data, svdcut=svdcut)

    bs_datalist = (
        ds.avg_data(d)
        for d in ds.bootstrap_iter(dset, n_boot)
    )

    bs = ds.Dataset()
    bs_chi2 = []
    n_success = 0

    for bs_fit in fitter.bootstrapped_fit_iter(datalist=bs_datalist):
        p = bs_fit.pmean
        bs.append("m_ps", [float(np.asarray(p["m_ps"])[0])])
        bs.append("Afit", [float(np.asarray(p["Afit"])[0])])
        bs_chi2.append(float(bs_fit.chi2))
        n_success += 1

    if n_success == 0:
        raise RuntimeError("All bootstrap PP fits failed.")

    bs = ds.avg_data(bs, bstrap=True)

    return {
        "fit": fit,
        "bootstrap": bs,
        "bootstrap_fit_stats": _bootstrap_fit_stats(bs_chi2, int(fit.dof)),
        "bootstrap_meta": {
            "n_requested": int(n_boot),
            "n_success": int(n_success),
            "n_failed": int(n_boot - n_success),
        },
    }


def fit_with_bootstrap_VV(v_samples, t0, t1, n_boot=200, svdcut=1e-8):
    """
    Standard + bootstrap correlated fit for VV using corrfitter.
    """
    v_samples = np.asarray(v_samples, dtype=float)
    _, T = v_samples.shape

    dset = _dataset_from_samples({"VV": v_samples})
    data = ds.avg_data(dset)

    models = make_models_VV(T, t0, t1)
    prior = make_prior_VV(v_samples, t0, t1)
    fitter = cf.CorrFitter(models=models)

    fit = fitter.lsqfit(prior=prior, data=data, svdcut=svdcut)

    bs_datalist = (
        ds.avg_data(d)
        for d in ds.bootstrap_iter(dset, n_boot)
    )

    bs = ds.Dataset()
    bs_chi2 = []
    n_success = 0

    for bs_fit in fitter.bootstrapped_fit_iter(datalist=bs_datalist):
        p = bs_fit.pmean
        bs.append("m_v", [float(np.asarray(p["m_v"])[0])])
        bs.append("AfitV", [float(np.asarray(p["AfitV"])[0])])
        bs_chi2.append(float(bs_fit.chi2))
        n_success += 1

    if n_success == 0:
        raise RuntimeError("All bootstrap VV fits failed.")

    bs = ds.avg_data(bs, bstrap=True)

    return {
        "fit": fit,
        "bootstrap": bs,
        "bootstrap_fit_stats": _bootstrap_fit_stats(bs_chi2, int(fit.dof)),
        "bootstrap_meta": {
            "n_requested": int(n_boot),
            "n_success": int(n_success),
            "n_failed": int(n_boot - n_success),
        },
    }


def fit_with_bootstrap_PP_A0P(ps_samples, a0p_samples, t0, t1, Ns, n_boot=200, svdcut=1e-8):
    """
    Standard + bootstrap simultaneous correlated fit for PP + A0P using corrfitter.
    """
    ps_samples = np.asarray(ps_samples, dtype=float)
    a0p_samples = np.asarray(a0p_samples, dtype=float)

    if ps_samples.shape != a0p_samples.shape:
        raise ValueError(
            f"PP and A0P samples must have the same shape, got "
            f"{ps_samples.shape} and {a0p_samples.shape}"
        )
    if Ns <= 0:
        raise RuntimeError("--Ns must be > 0 (needed for f_PS normalisation).")

    _, T = ps_samples.shape

    dset = _dataset_from_samples({
        "PP": ps_samples,
        "A0P": a0p_samples,
    })
    data = ds.avg_data(dset)

    models = make_models_PP_A0P(T, t0, t1)
    prior = make_prior_PP_A0P(ps_samples, t0, t1)
    fitter = cf.CorrFitter(models=models)

    fit = fitter.lsqfit(prior=prior, data=data, svdcut=svdcut)

    m_ps_std = fit.p["m_ps"][0]
    g_std = fit.p["g"][0]
    fPS_std = -gv.sqrt(2.0) * g_std / gv.sqrt(m_ps_std) / (Ns ** 1.5)

    bs_datalist = (
        ds.avg_data(d)
        for d in ds.bootstrap_iter(dset, n_boot)
    )

    bs = ds.Dataset()
    bs_chi2 = []
    n_success = 0

    for bs_fit in fitter.bootstrapped_fit_iter(datalist=bs_datalist):
        p = bs_fit.pmean

        m_ps_b = float(np.asarray(p["m_ps"])[0])
        Afit_b = float(np.asarray(p["Afit"])[0])
        g_b = float(np.asarray(p["g"])[0])
        f_ps_b = -np.sqrt(2.0) * g_b / np.sqrt(m_ps_b) / (Ns ** 1.5)

        bs.append("m_ps", [m_ps_b])
        bs.append("Afit", [Afit_b])
        bs.append("g", [g_b])
        bs.append("f_ps", [f_ps_b])
        bs_chi2.append(float(bs_fit.chi2))
        n_success += 1

    if n_success == 0:
        raise RuntimeError("All bootstrap simultaneous PP+A0P fits failed.")

    bs = ds.avg_data(bs, bstrap=True)

    return {
        "fit": fit,
        "fPS_fit_gvar": fPS_std,
        "bootstrap": bs,
        "bootstrap_fit_stats": _bootstrap_fit_stats(bs_chi2, int(fit.dof)),
        "bootstrap_meta": {
            "n_requested": int(n_boot),
            "n_success": int(n_success),
            "n_failed": int(n_boot - n_success),
        },
    }


##############################################################################
#                                   Z_A                                      #
##############################################################################


def compute_ZA(L, R):
    L = np.asarray(L)
    R = np.asarray(R)
    T = len(L)
    tvals = np.arange(1, T - 1)
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
    Build per-configuration Z_A(t) samples without folding.

    Returns:
      tvals: t = 1..T-2
      Z_samples: shape (Ncfg, T-2)
    """
    Lcorr = np.asarray(Lcorr)
    Rcorr = np.asarray(Rcorr)
    if Lcorr.shape != Rcorr.shape or Lcorr.ndim != 2:
        raise ValueError("build_Z_samples expects Lcorr and Rcorr with shape (Ncfg, T)")

    N, _ = Lcorr.shape
    tvals, _ = compute_ZA(Lcorr[0], Rcorr[0])
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
    """
    Z_samples = np.asarray(Z_samples)
    _, nt = Z_samples.shape
    tvals = np.arange(1, nt + 1)

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
    Bootstrap mean/std of Z_A(t) for plotting without folding.
    """
    Lcorr = np.asarray(Lcorr)
    Rcorr = np.asarray(Rcorr)

    tvals, _ = compute_ZA(Lcorr[0], Rcorr[0])
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

        N, T = ps.shape
        result["data_shape"] = {"Ncfg": int(N), "T": int(T)}

        rng = np.random.default_rng()
        boot_idx = rng.integers(0, N, size=(args.n_boot, N))

        tps, meps, eeps = bootstrap_effmass(ps, args.n_boot, boot_idx)
        tv, mev, eev = bootstrap_effmass(v, args.n_boot, boot_idx)
        ta, mea0p, ea0p = bootstrap_effmass(a0p, args.n_boot, boot_idx)

        pp_res = fit_with_bootstrap_PP(ps, ps0, ps1, n_boot=args.n_boot, svdcut=args.svdcut)
        vv_res = fit_with_bootstrap_VV(v, v0, v1, n_boot=args.n_boot, svdcut=args.svdcut)
        sim_res = fit_with_bootstrap_PP_A0P(
            ps, a0p, fps0, fps1, Ns=args.Ns, n_boot=args.n_boot, svdcut=args.svdcut
        )

        fit_pp = pp_res["fit"]
        fit_vv = vv_res["fit"]
        fit_sim = sim_res["fit"]

        mps_pp_gv = fit_pp.p["m_ps"][0]
        mv_gv = fit_vv.p["m_v"][0]
        mps_sim_gv = fit_sim.p["m_ps"][0]
        fPS_std_gv = sim_res["fPS_fit_gvar"]

        _, Z_samples = build_Z_samples(Ls, Rs)
        tZ_plot, Zt_plot, Zerr_plot = bootstrap_ZA_curve(Ls, Rs, args.n_boot, boot_idx)

        fit_Z = fit_Z_only(Z_samples, z0, z1, svdcut=args.svdcut)
        Zplat_gv = fit_Z.p["Z0"]

        plot_effmass(
            tps, meps, eeps, ps0, ps1,
            float(gv.mean(mps_pp_gv)), float(gv.sdev(mps_pp_gv)),
            args.plot_ps, args.label, args.beta, args.mass, "PS"
        )

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

        tmax = T // 2 - 2
        mask = tZ_plot <= tmax

        plot_plateau(
            tZ_plot[mask], Zt_plot[mask], Zerr_plot[mask], z0, z1,
            float(gv.mean(Zplat_gv)), float(gv.sdev(Zplat_gv)),
            args.plot_Z, args.label, args.beta, args.mass, "Z_A", r"$Z_A$"
        )

        result["results"] = {
            "standard_fit": {
                "PP": {
                    "am_ps": _gvar_to_obj(pp_res["fit"].p["m_ps"][0]),
                    "Afit": _gvar_to_obj(pp_res["fit"].p["Afit"][0]),
                    "fit_stats": _fit_stats(pp_res["fit"]),
                },
                "VV": {
                    "am_v": _gvar_to_obj(vv_res["fit"].p["m_v"][0]),
                    "AfitV": _gvar_to_obj(vv_res["fit"].p["AfitV"][0]),
                    "fit_stats": _fit_stats(vv_res["fit"]),
                },
                "simultaneous_PP_A0P": {
                    "am_ps": _gvar_to_obj(sim_res["fit"].p["m_ps"][0]),
                    "Afit": _gvar_to_obj(sim_res["fit"].p["Afit"][0]),
                    "g": _gvar_to_obj(sim_res["fit"].p["g"][0]),
                    "af_ps": _gvar_to_obj(sim_res["fPS_fit_gvar"]),
                    "fit_stats": _fit_stats(sim_res["fit"]),
                },
                "Z_A": {
                    "Z_A": _gvar_to_obj(Zplat_gv),
                    "fit_stats": _fit_stats(fit_Z),
                },
            },
            "bootstrap_fit": {
                "PP": {
                    "am_ps": _gvar_to_obj(pp_res["bootstrap"]["m_ps"][0]),
                    "Afit": _gvar_to_obj(pp_res["bootstrap"]["Afit"][0]),
                    "fit_stats": pp_res["bootstrap_fit_stats"],
                    "meta": pp_res["bootstrap_meta"],
                },
                "VV": {
                    "am_v": _gvar_to_obj(vv_res["bootstrap"]["m_v"][0]),
                    "AfitV": _gvar_to_obj(vv_res["bootstrap"]["AfitV"][0]),
                    "fit_stats": vv_res["bootstrap_fit_stats"],
                    "meta": vv_res["bootstrap_meta"],
                },
                "simultaneous_PP_A0P": {
                    "am_ps": _gvar_to_obj(sim_res["bootstrap"]["m_ps"][0]),
                    "Afit": _gvar_to_obj(sim_res["bootstrap"]["Afit"][0]),
                    "g": _gvar_to_obj(sim_res["bootstrap"]["g"][0]),
                    "af_ps": _gvar_to_obj(sim_res["bootstrap"]["f_ps"][0]),
                    "fit_stats": sim_res["bootstrap_fit_stats"],
                    "meta": sim_res["bootstrap_meta"],
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

    print(json.dumps(result, indent=2, sort_keys=True))

    if not result.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()