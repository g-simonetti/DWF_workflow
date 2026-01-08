#!/usr/bin/env python3
"""
spectrum.py — LQCD spectrum analysis using corrfitter (correlated fits).

Computes:
  - pseudoscalar mass m_PS
  - vector mass m_V (avg over X, Y, Z)
  - f_PS from a simultaneous correlated fit of (PP, A0P)
  - Z_A plateau from bootstrap

Correlators used:
  PP:   C_PP(t)  = A^2/(2 m) * ( e^{-m t} + e^{-m (T-t)} )
  A0P:  C_A0P(t) = f_PS*A/2  * ( e^{-m t} - e^{-m (T-t)} )

Implementation note:
  corrfitter's Corr2 model is:
      C(t) = (a*b) * (exp(-E t) + exp(-E (T-t)))   (tp = +T)
      C(t) = (a*b) * (exp(-E t) - exp(-E (T-t)))   (tp = -T)

  To match A^2/(2m), we fit Afit = A/sqrt(2m), so:
      C_PP(t) = (Afit^2) * (exp(-m t) + exp(-m (T-t)))

  For the axial correlator we fit g such that:
      C_A0P(t) = (Afit*g) * (exp(-m t) - exp(-m (T-t)))

  Then:
      f_PS = sqrt(2) * g / sqrt(m)

IMPORTANT USER REQUEST:
  For extracting f_PS (simultaneous fit), use plateau_start_fps/plateau_end_fps
  as the *same* window for BOTH PP and A0P.

USER REQUEST (PLOTTING EDITS ONLY):
  - Effective masses are computed from FOLDED correlators (to match corrfitter's tp folding)
  - Plot effective mass points only for t < T/2
  - Do NOT modify fits, Z_A, or output format/content (beyond the PS/fPS plot labels)
"""

import argparse
import glob
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

import gvar as gv
import corrfitter as cf

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
#                         FOLDING (PLOTS ONLY)                               #
##############################################################################

def fold_periodic(corr):
    """
    Fold a periodic/cosh-like correlator: C(t) = C(T-t).
    Partner index uses (T - t) % T on the last axis.
    """
    C = np.asarray(corr)
    T = C.shape[-1]
    t = np.arange(T)
    partner = (T - t) % T
    return 0.5 * (C + C[..., partner])

def fold_antiperiodic(corr):
    """
    Fold an anti-periodic/sinh-like correlator: C(t) = -C(T-t).
    Partner index uses (T - t) % T on the last axis.
    """
    C = np.asarray(corr)
    T = C.shape[-1]
    t = np.arange(T)
    partner = (T - t) % T
    return 0.5 * (C - C[..., partner])

def _keep_lt_half(t, y, e, T):
    """Keep only points with t < T/2 (plotting only)."""
    half = T // 2
    m = (t < half)
    return t[m], y[m], e[m]

def _finite_triplet(t, y, e):
    """Drop nan/inf points (for safety in plots)."""
    t = np.asarray(t)
    y = np.asarray(y)
    e = np.asarray(e)
    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(e)
    return t[m], y[m], e[m]

##############################################################################
#                           EFFECTIVE MASS                                   #
##############################################################################

def eff_mass_hyperbolic(C):
    """
    Effective mass from:
        cosh(m_eff(t)) = (C(t-1) + C(t+1)) / (2 C(t)).
    """
    C = np.asarray(C)
    T = len(C)
    t = np.arange(1, T - 1)
    R = (C[t - 1] + C[t + 1]) / (2 * C[t])
    R = np.maximum(R, 1.0)
    return t, np.arccosh(R)

def eff_mass_hyperbolic_safe(C, eps=1e-14):
    """
    Safe version for plotting/bootstrap: returns nan where undefined (e.g. C(t)=0).
    """
    C = np.asarray(C)
    T = len(C)
    t = np.arange(1, T - 1)

    num = C[t - 1] + C[t + 1]
    den = 2.0 * C[t]

    R = np.full_like(num, np.nan, dtype=float)
    np.divide(num, den, out=R, where=(np.abs(den) > eps))

    finite = np.isfinite(R)
    R[finite] = np.maximum(R[finite], 1.0)

    m_eff = np.full_like(R, np.nan, dtype=float)
    m_eff[finite] = np.arccosh(R[finite])
    return t, m_eff

def bootstrap_effmass(corr, n_boot, boot_idx):
    """
    Bootstrap effective mass for ALREADY FOLDED correlators.

    Assumes corr has shape (Ncfg, T) and has been folded beforehand.
    Effective mass is computed only on the interior time slices,
    where it is mathematically well-defined.
    """
    N, T = corr.shape

    # After folding, only t = 1 ... T//2 - 1 are meaningful
    t_max = T // 2
    t_vals = np.arange(1, t_max)

    nt = len(t_vals)
    m_samples = np.full((n_boot, nt), np.nan, dtype=float)

    for b in range(n_boot):
        Cb = corr[boot_idx[b]].mean(axis=0)

        # Compute effective mass on the folded interior only
        num = Cb[t_vals - 1] + Cb[t_vals + 1]
        den = 2.0 * Cb[t_vals]

        R = np.full(nt, np.nan)
        good = np.abs(den) > 0
        R[good] = num[good] / den[good]

        # arccosh only defined for R >= 1
        good = good & (R >= 1.0)
        m_samples[b, good] = np.arccosh(R[good])

    # Safe statistics: only where defined
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
    """
    N, T = ps_samples.shape
    data = gv.dataset.avg_data({"PP": ps_samples})
    cmean = ps_samples.mean(axis=0)

    m0 = _guess_m_from_effmass(cmean, t0, t1)
    A0 = _guess_Afit_from_corr(cmean, m0, t_ref=t0)

    prior = gv.BufferDict()
    prior["log(Afit)"] = gv.gvar([np.log(A0)], [3.0])     # broad
    prior["log(m_ps)"] = gv.gvar([np.log(m0)], [3.0])     # broad

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
    Simultaneous correlated fit over the SAME window [t0,t1] for BOTH PP and A0P.

    PP:  C_PP(t)  = Afit^2 * (exp(-mt) + exp(-m(T-t)))
    A0P: C_A0P(t) = (Afit*g) * (exp(-mt) - exp(-m(T-t)))
    => f_PS = sqrt(2) * g / sqrt(m)
    """
    N, T = ps_samples.shape
    data = gv.dataset.avg_data({"PP": ps_samples, "A0P": a0p_samples})
    cmean = ps_samples.mean(axis=0)

    m0 = _guess_m_from_effmass(cmean, t0, t1)
    A0 = _guess_Afit_from_corr(cmean, m0, t_ref=t0)

    prior = gv.BufferDict()
    prior["log(Afit)"] = gv.gvar([np.log(A0)], [3.0])
    prior["log(m_ps)"] = gv.gvar([np.log(m0)], [3.0])
    prior["g"]         = gv.gvar([0.0], [10.0])  # broad; can be ±

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
        tp=-T,  # difference: exp(-mt) - exp(-m(T-t))
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
    tvals = np.arange(1, T-1)
    Z = np.zeros_like(tvals, float)

    for i, t in enumerate(tvals):
        Rf = R[t]
        Rb = R[t-1]
        Lt = L[t]
        Ltp = L[t+1]

        if Lt == 0 or (Lt + Ltp) == 0:
            Z[i] = 0.0
            continue

        term1 = (Rf + Rb) / (2 * Lt)
        term2 = 2 * Rf / (Lt + Ltp)
        Z[i] = 0.5 * (term1 + term2)
    return tvals, Z

def fold_ZA(Z):
    """
    Fold Z_A(t) defined on t = 1..T-2 (length T-2).
    This is a periodic/cosh-like folding on the *reduced* time index:
        Z(t) = Z(T - t)  -> in reduced indexing partner is (nt-1 - i).
    Works with shape (..., nt).
    """
    Z = np.asarray(Z)
    nt = Z.shape[-1]
    i = np.arange(nt)
    partner = (nt - 1 - i)
    return 0.5 * (Z + Z[..., partner])

def bootstrap_ZA(Lcorr, Rcorr, t0, t1, n_boot, boot_idx):
    """
    Bootstrap Z_A plateau, but using the FOLDED Z_A(t) for both:
      - the returned mean/std curve (for plotting)
      - the plateau extraction over [t0,t1]
    """
    Lcorr = np.asarray(Lcorr)
    Rcorr = np.asarray(Rcorr)
    N, T = Lcorr.shape

    tvals0, _ = compute_ZA(Lcorr[0], Rcorr[0])
    mask = (tvals0 >= t0) & (tvals0 <= t1)
    nt = len(tvals0)

    Z_samp = np.zeros((n_boot, nt))
    Zp = np.zeros(n_boot)

    for b in range(n_boot):
        idx = boot_idx[b]
        Lm = Lcorr[idx].mean(0)
        Rm = Rcorr[idx].mean(0)
        _, Zt = compute_ZA(Lm, Rm)

        # Fold Z_A(t) before storing / plateau extraction
        Zt = fold_ZA(Zt)

        Z_samp[b] = Zt
        Zp[b] = Zt[mask].mean()

    return tvals0, Z_samp.mean(0), Z_samp.std(0, ddof=1), Zp.mean(), Zp.std(ddof=1)

##############################################################################
#                                  PLOTTING                                  #
##############################################################################

def plot_plateau(t, y, e, t0, t1, fit, dfit, outfile, label_flag, beta, mass, qlabel, ylabel):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    t, y, e = _finite_triplet(t, y, e)

    data_label = rf"$\beta={beta},\ am_0={mass}$" if label_flag == "yes" else None
    fit_label  = rf"${qlabel} = {fit:.5f}\pm{dfit:.5f}$"

    ax.errorbar(t, y, yerr=e, fmt="o", color="C4", label=data_label)
    ax.axvspan(t0, t1, alpha=0.2, color="C2", label="Plateau")

    ax.fill_between([t0, t1], [fit-dfit, fit-dfit], [fit+dfit, fit+dfit],
                    color="C1", alpha=0.25)
    ax.hlines(fit, t0, t1, color="C1", linestyle="--", label=fit_label)

    ax.set_xlabel(r"$t/a$")
    ax.set_ylabel(ylabel)

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    if data_label or fit_label:
        ax.legend()

    if "PS" in qlabel or "V" in qlabel:
        ax.set_ylim(0, 1.0)

    plt.savefig(outfile, dpi=300)
    plt.close()
    print("Saved", outfile)

def plot_effmass(t, m, e, t0, t1, mfit, dfit, outfile, label, beta, mass, chan):
    qlab = f"am_{{\\rm {chan}}}"
    plot_plateau(
        t, m, e, t0, t1, mfit, dfit,
        outfile, label, beta, mass,
        qlab, ylabel=r"$am_{\rm eff}$"
    )

def plot_fps_two_panel(
    tps, meps, eeps,
    ta,  mea0p, ea0p,
    t0, t1,
    mps, dmps, fPS, dfPS,
    outfile, label_flag, beta, mass
):
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(3.5, 2.5),
        sharex=True, layout="constrained"
    )

    tps, meps, eeps  = _finite_triplet(tps, meps, eeps)
    ta,  mea0p, ea0p = _finite_triplet(ta, mea0p, ea0p)

    # Legend labels (same logic as plot_plateau)
    data_label = rf"$\beta={beta},\ am_0={mass}$" if label_flag == "yes" else None
    fit_label  = (
        rf"$am_{{\rm PS}} = {mps:.5f}\pm{dmps:.5f}$"
        + "\n" +
        rf"$af_{{\rm PS}} = {fPS:.5f}\pm{dfPS:.5f}$"
    )

    # --- Top panel (PP) ---
    ax0.errorbar(tps, meps, yerr=eeps, fmt="o", color="C4", label=data_label)
    ax0.axvspan(t0, t1, alpha=0.2, color="C2", label="Plateau")
    ax0.fill_between(
        [t0, t1],
        [mps - dmps, mps - dmps],
        [mps + dmps, mps + dmps],
        color="C1", alpha=0.25
    )
    ax0.hlines(mps, t0, t1, color="C1", linestyle="--", label=fit_label)
    ax0.set_ylabel(r"$am_{\rm eff}^{\rm PS,PS}$")

    # --- Bottom panel (A0P) ---
    ax1.errorbar(ta, mea0p, yerr=ea0p, fmt="o", color="C4")
    ax1.axvspan(t0, t1, alpha=0.2, color="C2")
    ax1.fill_between(
        [t0, t1],
        [mps - dmps, mps - dmps],
        [mps + dmps, mps + dmps],
        color="C1", alpha=0.25
    )
    ax1.hlines(mps, t0, t1, color="C1", linestyle="--")
    ax1.set_xlabel(r"$t/a$")
    ax1.set_ylabel(r"$am_{\rm eff}^{\rm AV,PS}$")

    for ax in (ax0, ax1):
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # Plateau-style legend
    if data_label or fit_label:
        ax0.legend()

    plt.savefig(outfile, dpi=300)
    plt.close()
    print("Saved", outfile)


##############################################################################
#                                   MAIN                                     #
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--label", default="")
    parser.add_argument("--output_file", required=True)

    parser.add_argument("--plot_ps", required=True)
    parser.add_argument("--plot_v", required=True)
    parser.add_argument("--plot_fps", required=True)
    parser.add_argument("--plot_Z", required=True)
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
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--svdcut", type=float, default=1e-8)

    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    ps0, ps1   = int(args.plateau_start_ps),  int(args.plateau_end_ps)
    v0,  v1    = int(args.plateau_start_v),   int(args.plateau_end_v)
    fps0, fps1 = int(args.plateau_start_fps), int(args.plateau_end_fps)
    z0,  z1    = int(args.plateau_start_Z),   int(args.plateau_end_Z)

    pt_files   = sorted(glob.glob(os.path.join(args.input_dir, "pt_ll.*.h5")))
    mres_files = sorted(glob.glob(os.path.join(args.input_dir, "mres.*.h5")))

    if len(pt_files) == 0:
        raise RuntimeError("No pt_ll files found.")
    if len(mres_files) == 0:
        raise RuntimeError("No mres files found.")
    if len(pt_files) != len(mres_files):
        raise RuntimeError("pt_ll and mres files must match configuration by configuration.")

    ps, v, a0p, Ls, Rs = [], [], [], [], []

    for fpt, fmr in zip(pt_files, mres_files):
        ps.append(read_ps_corr(fpt))

        vx = read_vx_corr(fpt)
        vy = read_vy_corr(fpt)
        vz = read_vz_corr(fpt)
        v.append((vx + vy + vz) / 3)

        A0_33 = read_L_corr(fpt)
        A0_9  = read_L2_corr(fpt)
        A0_comb = 0.5 * (A0_33 + A0_9)  

        a0p.append(A0_comb)
        Ls.append(A0_comb)
        Rs.append(read_R_corr(fmr))

    ps  = np.array(ps)
    v   = np.array(v)
    a0p = np.array(a0p)
    Ls  = np.array(Ls)
    Rs  = np.array(Rs)

    N, T = ps.shape

    rng = np.random.default_rng()
    boot_idx = rng.integers(0, N, size=(args.n_boot, N))

    # Effective masses for plots: use folded correlators, and keep only t < T/2
    ps_fold  = fold_periodic(ps)
    v_fold   = fold_periodic(v)
    a0p_fold = fold_antiperiodic(a0p)

    tps, meps, eeps  = bootstrap_effmass(ps_fold,  args.n_boot, boot_idx)
    tv,  mev,  eev   = bootstrap_effmass(v_fold,   args.n_boot, boot_idx)
    ta,  mea0p, ea0p = bootstrap_effmass(a0p_fold, args.n_boot, boot_idx)

    tps, meps, eeps  = _keep_lt_half(tps, meps, eeps, T)
    tv,  mev,  eev   = _keep_lt_half(tv,  mev,  eev,  T)
    ta,  mea0p, ea0p = _keep_lt_half(ta,  mea0p, ea0p, T)

    # Fits unchanged (corrfitter folds internally via tp)
    fit_pp  = fit_PP_only(ps, ps0, ps1, svdcut=args.svdcut)
    fit_vv  = fit_VV_only(v,  v0,  v1,  svdcut=args.svdcut)
    fit_sim = fit_simultaneous_PP_A0P(ps, a0p, fps0, fps1, svdcut=args.svdcut)
    

    # Masses for plot labels (PP-only for PS plot; SIM for fPS plot)
    mps_pp_gv = fit_pp.p["m_ps"][0]
    mps_pp  = float(gv.mean(mps_pp_gv))
    dmps_pp = float(gv.sdev(mps_pp_gv))

    mps_sim_gv = fit_sim.p["m_ps"][0]
    mps_sim  = float(gv.mean(mps_sim_gv))
    dmps_sim = float(gv.sdev(mps_sim_gv))

    mv_gv = fit_vv.p["m_v"][0]
    mv  = float(gv.mean(mv_gv))
    dmv = float(gv.sdev(mv_gv))

    # f_PS (use SIM mass)
    g_gv = fit_sim.p["g"][0]
    fPS_gv = gv.sqrt(2.0) * g_gv / gv.sqrt(mps_sim_gv)
    fPS  = float(gv.mean(fPS_gv))
    fPS  = - fPS / args.Ns**1.5 #Factor 1/ sqrt(V) comes from Z2 normalisation
    dfPS = float(gv.sdev(fPS_gv))
    dfPS  = dfPS / args.Ns**1.5

    # chi2/dof unchanged
    chi2ps  = float(fit_pp.chi2  / fit_pp.dof)  if fit_pp.dof  > 0 else np.nan
    chi2v   = float(fit_vv.chi2  / fit_vv.dof)  if fit_vv.dof  > 0 else np.nan
    chi2fps = float(fit_sim.chi2 / fit_sim.dof) if fit_sim.dof > 0 else np.nan

    # Z_A unchanged (except now bootstrap uses folded Z_A(t) and returns folded curve)
    tZ, Zt, Zerr, Zplat, dZ = bootstrap_ZA(Ls, Rs, z0, z1, args.n_boot, boot_idx)

    # Restrict Z_A plot to t in [1, T/2 - 1)
    z_plot_mask = tZ < (T // 2 - 1)
    tZ   = tZ[z_plot_mask]
    Zt   = Zt[z_plot_mask]
    Zerr = Zerr[z_plot_mask]

    # Output unchanged (still writes SIM mps as your original code behavior)
    with open(args.output_file, "w") as f:
        f.write("#am_ps am_ps_err am_v am_v_err chi2_ps chi2_v  "
                "af_ps af_ps_err chi2_fps  Z_A Z_A_err\n")
        f.write(
            f"{mps_sim:.6e} {dmps_sim:.6e}  "
            f"{mv:.6e} {dmv:.6e}  "
            f"{chi2ps:.4e} {chi2v:.4e}  "
            f"{fPS:.6e} {dfPS:.6e} {chi2fps:.4e}  "
            f"{Zplat:.6e} {dZ:.6e}\n"
        )

    # Plots (meff points are folded and limited to t < T/2)
    plot_effmass(tps, meps, eeps, ps0, ps1, mps_pp, dmps_pp,
                 args.plot_ps, args.label, args.beta, args.mass, "PS")

    plot_effmass(tv,  mev,  eev,  v0,  v1,  mv,  dmv,
                 args.plot_v, args.label, args.beta, args.mass, "V")

    plot_fps_two_panel(
        tps, meps, eeps,
        ta,  mea0p, ea0p,
        fps0, fps1,
        mps_sim, dmps_sim, fPS, dfPS,
        args.plot_fps, args.label, args.beta, args.mass
    )

    # Z_A plot now shows folded Z_A(t) returned by bootstrap_ZA
    plot_plateau(tZ, Zt, Zerr, z0, z1, Zplat, dZ, args.plot_Z,
                 args.label, args.beta, args.mass,
                 "Z_A", r"$Z_A$")


if __name__ == "__main__":
    main()
