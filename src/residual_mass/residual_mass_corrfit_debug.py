#!/usr/bin/env python3
import argparse
import glob
import os
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.style.use("tableau-colorblind10")


# ============================================================
# File Readers
# ============================================================
def read_mres_file(filename: str) -> np.ndarray:
    """Read real part of PJ5q from mres HDF5 file. Returns shape (T,)."""
    with h5py.File(filename, "r") as f:
        return f["wardIdentity/PJ5q"][:]["re"]


def read_ptll_file(filename: str, n_elems: int | None = None) -> np.ndarray:
    """Read real part of the corr dataset for meson_1. Returns shape (T,)."""
    with h5py.File(filename, "r") as f:
        data = f["meson/meson_1/corr"][:]
        if n_elems is None or len(data) == n_elems:
            return data["re"]
    raise ValueError(f"corr dataset does not have {n_elems} entries in {filename}")


# ============================================================
# Folding
# ============================================================
def fold_time_axis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Fold along time using C_f(t)=1/2*(C(t)+C((T-t) mod T)),
    returning only t = 0..T//2 (inclusive).

    Input:
      x: shape (Ncfg, T) or (T,)

    Output:
      x_f: shape (Ncfg, T//2+1) or (T//2+1,)
      t_f: array of folded time indices, 0..T//2
    """
    arr = np.asarray(x)
    squeeze_back = False
    if arr.ndim == 1:
        arr = arr[None, :]
        squeeze_back = True
    if arr.ndim != 2:
        raise ValueError(f"fold_time_axis expects 1D or 2D array, got shape {arr.shape}")

    ncfg, T = arr.shape
    Tf = T // 2 + 1
    t_f = np.arange(Tf, dtype=int)
    partner = (T - t_f) % T

    folded = 0.5 * (arr[:, t_f] + arr[:, partner])

    if squeeze_back:
        return folded[0], t_f
    return folded, t_f


# ============================================================
# Bootstrap: ratio of means (NOT mean of ratios)
# ============================================================
def bootstrap_ratio_of_means(
    num_cfg: np.ndarray,
    den_cfg: np.ndarray,
    n_boot: int = 2000,
    tiny: float = 0.0,
):
    """
    Bootstrap the estimator R(t) = mean_k N_k(t) / mean_k D_k(t).

    num_cfg, den_cfg: shape (Ncfg, T)

    Returns:
      ratio_mean: (T,)  from full-sample means
      ratio_err : (T,)  bootstrap std of ratio-of-means replicas
      boot_rep  : (n_boot, T) ratio-of-means replicas (useful for cov estimation)
    """
    num_cfg = np.asarray(num_cfg, dtype=float)
    den_cfg = np.asarray(den_cfg, dtype=float)
    if num_cfg.shape != den_cfg.shape:
        raise ValueError(
            f"num_cfg and den_cfg must have same shape, got {num_cfg.shape} vs {den_cfg.shape}"
        )

    Ncfg, T = num_cfg.shape
    boot_rep = np.empty((n_boot, T), dtype=float)

    for b in range(n_boot):
        idx = np.random.randint(0, Ncfg, size=Ncfg)
        num_b = num_cfg[idx].mean(axis=0)
        den_b = den_cfg[idx].mean(axis=0)
        boot_rep[b] = num_b / (den_b + tiny)

    num_mean = num_cfg.mean(axis=0)
    den_mean = den_cfg.mean(axis=0)
    ratio_mean = num_mean / (den_mean + tiny)
    ratio_err = boot_rep.std(axis=0, ddof=1)
    return ratio_mean, ratio_err, boot_rep


# ============================================================
# Autocorrelation
# ============================================================
def integrated_autocorrelation_time(x, c: float = 5.0, M_max: int = 50):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 0.5, 0.0

    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return 0.5, 0.0

    acf = np.correlate(x, x, mode="full")[n - 1 :] / (var * np.arange(n, 0, -1))
    candidate_M = range(1, min(M_max, n - 1) + 1)
    tau_int_M = np.array([0.5 + np.sum(acf[1 : M + 1]) for M in candidate_M])

    valid = [M for M, tau in zip(candidate_M, tau_int_M) if M >= c * tau]
    M_selected = min(valid) if valid else list(candidate_M)[-1]

    tau = 0.5 + np.sum(acf[1 : M_selected + 1])
    tau_err = np.sqrt(2.0 * (M_selected + 1) / n) * tau
    return tau, tau_err


# ============================================================
# Trajectory numbers
# ============================================================
def extract_trajectory_numbers(file_list, pattern=r".*\.([0-9]+)\.h5"):
    numbers = []
    for f in file_list:
        m = re.match(pattern, os.path.basename(f))
        if not m:
            raise ValueError(f"Cannot extract trajectory number from filename: {f}")
        numbers.append(int(m.group(1)))
    return sorted(numbers)


# ============================================================
# Covariance inversion helper
# ============================================================
def _stable_inverse_cov(S, ridge: float = 0.0, eps_rel: float = 1e-12):
    """
    Symmetrize S, apply optional ridge, then invert via eigen-decomposition
    with eigenvalue floor eps_rel * max_eig to stabilize near-singular matrices.
    """
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)

    if ridge and ridge > 0.0:
        d = np.diag(S).copy()
        if np.all(d == 0):
            S = S + ridge * np.eye(S.shape[0])
        else:
            S = S + ridge * np.diag(d)

    w, V = np.linalg.eigh(S)
    w_max = float(np.max(w)) if np.max(w) > 0 else 1.0
    floor = eps_rel * w_max
    w_clip = np.where(w > floor, w, floor)
    Sinv = (V * (1.0 / w_clip)) @ V.T

    n_clip = int(np.sum(w <= floor))
    info = (
        f"eig_min={w.min():.3e}, eig_max={w.max():.3e}, "
        f"floor={floor:.3e}, n_clip={n_clip}/{len(w)}"
    )
    return Sinv, info


# ============================================================
# Correlation/Covariance check report (your requested format)
# ============================================================
def write_corr_cov_checks_report(
    out_path: str,
    Sigma: np.ndarray,
    ridge_for_Sigma_inv: float,
    eps_rel: float = 1e-12,
):
    """
    Writes a report formatted like:

      # std (sqrt(diag(Sigma))):
      # --- Covariance matrix Sigma ---
      # --- Correlation matrix R = D^{-1} Sigma D^{-1} ---
      # --- (Sigma - Sigma via correlation matrix) ---
      max_abs_diff = ...
      # --- ( inv(Sigma) - inv(Sigma via correlation matrix) )---
      max_abs_diff = ...

    with short definitions:
      std_i = sqrt(Sigma_ii)
      D = diag(std)
      R = D^{-1} Sigma D^{-1}
      Sigma_via_R = D R D
      inv(Sigma via R) = D^{-1} inv(R) D^{-1}
    """
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T)

    var = np.diag(Sigma).copy()
    if np.any(var < 0):
        raise ValueError("Sigma has negative diagonal entries; cannot form std.")
    std = np.sqrt(var)
    if np.any(std == 0):
        bad = np.where(std == 0)[0]
        raise ValueError(f"Zero std at indices {bad.tolist()}; cannot form correlation.")

    invstd = 1.0 / std
    Dinv = np.diag(invstd)

    R = (Sigma * invstd[None, :]) * invstd[:, None]
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    Sigma_via_R = (R * std[None, :]) * std[:, None]
    Sigma_via_R = 0.5 * (Sigma_via_R + Sigma_via_R.T)

    dSigma = Sigma - Sigma_via_R
    max_abs_dSigma = float(np.max(np.abs(dSigma)))

    Omega_direct, _ = _stable_inverse_cov(Sigma, ridge=ridge_for_Sigma_inv, eps_rel=eps_rel)
    Rinv, _ = _stable_inverse_cov(R, ridge=0.0, eps_rel=eps_rel)
    Omega_via_R = Dinv @ Rinv @ Dinv
    Omega_via_R = 0.5 * (Omega_via_R + Omega_via_R.T)

    dOmega = Omega_direct - Omega_via_R
    max_abs_dOmega = float(np.max(np.abs(dOmega)))

    with open(out_path, "w") as f:
        f.write("# D = diag(std), where std_i = sqrt(Sigma_ii)\n")
        f.write("# R = D^{-1} Sigma D^{-1}\n")
        f.write("# Sigma via correlation matrix = D R D\n")
        f.write("# inv(Sigma via correlation matrix) = D^{-1} inv(R) D^{-1}\n\n")

        f.write("# std (sqrt(diag(Sigma))):\n")
        f.write("# " + " ".join(f"{x:.16e}" for x in std.tolist()) + "\n\n")

        f.write("# --- Covariance matrix Sigma ---\n")
        np.savetxt(f, Sigma, fmt="%.16e")
        f.write("\n")

        f.write("# --- Correlation matrix R = D^{-1} Sigma D^{-1} ---\n")
        np.savetxt(f, R, fmt="%.16e")
        f.write("\n")

        f.write("# --- (Sigma - Sigma via correlation matrix) ---\n")
        np.savetxt(f, dSigma, fmt="%.16e")
        f.write("\n")
        f.write(f"max_abs_diff = {max_abs_dSigma:.12e}\n\n")

        f.write("# --- ( inv(Sigma) - inv(Sigma via correlation matrix) )---\n")
        np.savetxt(f, dOmega, fmt="%.16e")
        f.write("\n")
        f.write(f"max_abs_diff = {max_abs_dOmega:.12e}\n")


# ============================================================
# Plateau fit (correlated GLS) + return window covariance
# ============================================================
def plateau_fit_correlated_analytic_err_from_bootrep(
    ratio_mean: np.ndarray,
    ratio_bootrep: np.ndarray,
    t_vals: np.ndarray,
    tmin: int,
    tmax: int,
    ridge: float = 0.0,
):
    """
    Correlated constant (plateau) fit on [tmin, tmax].

    - Use ratio_bootrep (bootstrap replicas of ratio-of-means) to estimate Σ in the window
    - Use ybar = ratio_mean in the window
    - GLS constant + analytic GLS error

    Returns:
      avg, err, chi2, red_chi2, cov_info, Sigma_win, ybar_win, mask
    """
    ratio_mean = np.asarray(ratio_mean, dtype=float)
    ratio_bootrep = np.asarray(ratio_bootrep, dtype=float)
    t_vals = np.asarray(t_vals, dtype=int)

    mask = (t_vals >= tmin) & (t_vals <= tmax)
    if not np.any(mask):
        raise ValueError(f"No points in plateau range {tmin}–{tmax}")

    tw = int(np.sum(mask))
    one = np.ones(tw, dtype=float)

    Y = ratio_bootrep[:, mask]  # (Nboot, tw)
    ybar = ratio_mean[mask]     # (tw,)

    Sigma = np.cov(Y, rowvar=False, ddof=1)
    Sinv, cov_info = _stable_inverse_cov(Sigma, ridge=ridge, eps_rel=1e-12)

    denom = float(one @ Sinv @ one)
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Invalid denominator 1^T Σ^{-1} 1 (non-finite or non-positive).")

    avg = float(one @ Sinv @ ybar) / denom

    r = ybar - avg * one
    chi2 = float(r @ Sinv @ r)
    dof = max(tw - 1, 1)
    red_chi2 = chi2 / dof

    err = float(np.sqrt(1.0 / denom))
    return avg, err, chi2, red_chi2, cov_info, Sigma, ybar, mask


# ============================================================
# Eigenbasis diagnostics (ALWAYS prints eigenvectors + orthonormality)
# (unchanged from your script)
# ============================================================
def covariance_eigen_diagnostics(
    Sigma: np.ndarray,
    ybar: np.ndarray,
    fit_value: float,
    t_window: np.ndarray,
    out_prefix: str,
    n_show_vecs: int = 6,
    stdout_print_modes: int = 6,
):
    Sigma = np.asarray(Sigma, dtype=float)
    ybar = np.asarray(ybar, dtype=float)
    t_window = np.asarray(t_window, dtype=int)

    tw = Sigma.shape[0]
    if Sigma.shape != (tw, tw) or ybar.shape != (tw,) or t_window.shape != (tw,):
        raise ValueError("Shape mismatch in covariance_eigen_diagnostics.")

    Sigma = 0.5 * (Sigma + Sigma.T)

    lam, V = np.linalg.eigh(Sigma)
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    V = V[:, order]

    one = np.ones(tw, dtype=float)
    r = ybar - fit_value * one

    zeta = V.T @ r
    lam_safe = np.where(lam > 0, lam, np.nan)
    w = zeta / np.sqrt(lam_safe)
    chi2_modes = w**2

    lam_pos = lam[lam > 0]
    cond = (lam_pos.max() / lam_pos.min()) if lam_pos.size else np.inf

    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    G = V.T @ V
    I = np.eye(tw)
    ortho_err = G - I

    max_abs_diag = float(np.max(np.abs(np.diag(G) - 1.0))) if tw else 0.0
    off = ortho_err - np.diag(np.diag(ortho_err))
    max_abs_offdiag = float(np.max(np.abs(off))) if tw else 0.0
    frob = float(np.linalg.norm(ortho_err, ord="fro")) if tw else 0.0

    EV_res = Sigma @ V - V * lam
    rel_res = float(np.linalg.norm(EV_res, ord="fro") / max(np.linalg.norm(Sigma, ord="fro"), 1e-300))

    with open(out_prefix + "_orthonorm.txt", "w") as f:
        f.write("# Orthonormality + eigendecomposition residual checks\n")
        f.write("# For symmetric Σ, eigenvectors can be chosen orthonormal: V^T V = I.\n")
        f.write(f"# max |diag(V^T V) - 1|      = {max_abs_diag:.12e}\n")
        f.write(f"# max |offdiag(V^T V)|       = {max_abs_offdiag:.12e}\n")
        f.write(f"# Frobenius ||V^T V - I||_F  = {frob:.12e}\n")
        f.write(f"# rel ||ΣV - VΛ||_F / ||Σ||_F = {rel_res:.12e}\n")

    print("=== Eigenvector orthonormality check ===")
    print(f"max |diag(V^T V)-1|      = {max_abs_diag:.3e}")
    print(f"max |offdiag(V^T V)|     = {max_abs_offdiag:.3e}")
    print(f"||V^T V - I||_F          = {frob:.3e}")
    print(f"rel ||ΣV - VΛ||/||Σ||_F   = {rel_res:.3e}")
    print("")

    with open(out_prefix + "_eigvals.txt", "w") as f:
        f.write("# i  eigenvalue  log10(eigenvalue)\n")
        for i, x in enumerate(lam):
            f.write(f"{i:4d}  {x:.12e}  {np.log10(x) if x > 0 else np.nan:.6f}\n")
        f.write(f"# condition_number (pos eigs) = {cond:.6e}\n")

    with open(out_prefix + "_modes.txt", "w") as f:
        f.write("# i  eigenvalue  zeta  whitened_w  chi2_mode\n")
        for i in range(tw):
            f.write(f"{i:4d}  {lam[i]:.12e}  {zeta[i]:.12e}  {w[i]:.6e}  {chi2_modes[i]:.6e}\n")
        f.write(f"# sum chi2_modes (nan ignored) = {np.nansum(chi2_modes):.8f}\n")

    with open(out_prefix + "_eigvecs.txt", "w") as f:
        f.write("# Eigenvectors V (columns), sorted to match lam in _eigvals.txt\n")
        f.write("# Rows correspond to t_window entries listed below.\n")
        f.write("# t_window:\n")
        f.write("# " + " ".join(map(str, t_window.tolist())) + "\n")
        f.write("# Row i: V[i, 0..tw-1]\n")
        for i in range(tw):
            f.write(" ".join(f"{V[i, j]: .12e}" for j in range(tw)) + "\n")

    nprint = max(1, min(int(stdout_print_modes), tw)) if tw > 0 else 0
    for j in range(nprint):
        vj = V[:, j]
        print(f"mode {j}: lambda={lam[j]:.6e}, ||v||={np.linalg.norm(vj):.6f}")
        print("  v =", " ".join(f"{x:+.6e}" for x in vj))
    if tw > nprint:
        print(f"... remaining {tw - nprint} eigenvectors written to {out_prefix}_eigvecs.txt")

    fig, ax = plt.subplots(figsize=(4.2, 3.0), layout="constrained")
    ax.plot(np.arange(tw), lam, marker="o", linestyle="-")
    ax.set_yscale("log")
    ax.set_xlabel("Mode index (sorted by eigenvalue)")
    ax.set_ylabel("Eigenvalue λ")
    ax.set_title(f"Covariance eigenvalues (cond~{cond:.2e})")
    fig.savefig(out_prefix + "_eigvals.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), layout="constrained")
    ax.plot(np.arange(tw), chi2_modes, marker="o", linestyle="-")
    ax.set_yscale("log")
    ax.set_xlabel("Mode index (sorted)")
    ax.set_ylabel(r"Mode contribution $w_i^2$")
    ax.set_title(r"Mode contributions to $\chi^2$")
    fig.savefig(out_prefix + "_chi2_modes.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), layout="constrained")
    ax.plot(np.arange(tw), w, marker="o", linestyle="-")
    ax.axhline(0.0, linewidth=1)
    ax.axhline(3.0, linestyle="--", linewidth=1)
    ax.axhline(-3.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Mode index (sorted)")
    ax.set_ylabel(r"Whitened residual $w_i=\zeta_i/\sqrt{\lambda_i}$")
    ax.set_title("Whitened residuals (|w|>3 are suspicious)")
    fig.savefig(out_prefix + "_whitened.png", dpi=200)
    plt.close(fig)

    finite = np.isfinite(chi2_modes)
    if np.any(finite):
        idx_sorted = np.argsort(chi2_modes[finite])[::-1]
        bad_modes = np.where(finite)[0][idx_sorted[: min(n_show_vecs, np.sum(finite))]]

        fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
        for j in bad_modes:
            ax.plot(
                t_window,
                V[:, j],
                marker="o",
                linestyle="-",
                label=f"mode {j} (w^2={chi2_modes[j]:.1f})",
            )
        ax.set_xlabel("t (plateau window)")
        ax.set_ylabel("Eigenvector component")
        ax.set_title("Eigenvectors of largest χ²-contributing modes")
        ax.legend(fontsize=7)
        fig.savefig(out_prefix + "_bad_eigvecs.png", dpi=200)
        plt.close(fig)

    return cond, lam, w, chi2_modes


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute residual mass from HDF5 data (optional fold correlators, then ratio-of-means)."
    )
    parser.add_argument("input_dir")

    parser.add_argument("--label", default="", help="yes → include β, am0 label on plot")
    parser.add_argument("--output_file1", required=True)
    parser.add_argument("--output_file2", required=True)
    parser.add_argument("--plot_file", required=True)
    parser.add_argument("--plot_styles", default="")

    parser.add_argument("--plateau_start", type=float, required=True)
    parser.add_argument("--plateau_end", type=float, required=True)

    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--mass", type=float, default=None)

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--a5", type=float, required=True)
    parser.add_argument("--m5", type=float, required=True)
    parser.add_argument("--mpv", type=float, required=True)

    parser.add_argument("--n_boot", type=int, default=2000, help="number of bootstrap replicas")
    parser.add_argument("--cov_ridge", type=float, default=0.0, help="ridge regularization for cov inverse")

    parser.add_argument("--fold", action="store_true", help="fold correlators in time, then form ratio")
    parser.add_argument("--no-fold", dest="fold", action="store_false", help="do not fold; ratio on full T")
    parser.set_defaults(fold=True)

    parser.add_argument(
        "--eigvec_stdout",
        type=int,
        default=6,
        help="number of eigenvectors to print to stdout (always writes all to file)",
    )

    args = parser.parse_args()

    plateau_start = int(round(args.plateau_start))
    plateau_end = int(round(args.plateau_end))

    # ----------------------------------------------------------
    # Load HDF5 Data
    # ----------------------------------------------------------
    mres_files = sorted(glob.glob(os.path.join(args.input_dir, "mres.*.h5")))
    ptll_files = sorted(glob.glob(os.path.join(args.input_dir, "pt_ll.*.h5")))

    if not mres_files or not ptll_files:
        raise FileNotFoundError("Missing mres.*.h5 or pt_ll.*.h5 files.")

    mres_data = np.array([read_mres_file(f) for f in mres_files])  # (Ncfg, T)
    T_full = mres_data.shape[1]
    ptll_data = np.array([read_ptll_file(f, n_elems=T_full) for f in ptll_files])

    min_len = min(len(mres_data), len(ptll_data))
    mres_data = mres_data[:min_len]
    ptll_data = ptll_data[:min_len]

    # ----------------------------------------------------------
    # Fold correlators first (optional)
    # ----------------------------------------------------------
    if args.fold:
        mres_cfg, t_vals = fold_time_axis(mres_data)
        ptll_cfg, _ = fold_time_axis(ptll_data)

        n_times = mres_cfg.shape[1]
        t_max = n_times - 1

        if plateau_end > t_max:
            print(f"⚠️  plateau_end={plateau_end} exceeds T/2={t_max}; clamping plateau_end → {t_max}")
            plateau_end = t_max
        if plateau_start > t_max:
            raise ValueError(f"plateau_start={plateau_start} exceeds T/2={t_max}; no data after folding.")
    else:
        mres_cfg = mres_data
        ptll_cfg = ptll_data
        t_vals = np.arange(T_full, dtype=int)
        n_times = T_full

    tiny = 1e-300 if np.any(ptll_cfg == 0.0) else 0.0

    # ----------------------------------------------------------
    # Ratio of means + bootstrap error
    # ----------------------------------------------------------
    ratio_mean, ratio_err, ratio_bootrep = bootstrap_ratio_of_means(
        mres_cfg, ptll_cfg, n_boot=args.n_boot, tiny=tiny
    )

    # ----------------------------------------------------------
    # Autocorrelation (diagnostic: per-config ratio series)
    # ----------------------------------------------------------
    ratios_per_cfg = (mres_cfg / (ptll_cfg + tiny)).astype(float)
    tau_ints = np.empty(n_times, dtype=float)
    tau_errs = np.empty(n_times, dtype=float)
    for i in range(n_times):
        tau_ints[i], tau_errs[i] = integrated_autocorrelation_time(ratios_per_cfg[:, i])

    # ----------------------------------------------------------
    # Trajectory numbers
    # ----------------------------------------------------------
    traj_numbers = extract_trajectory_numbers(mres_files[:min_len])
    traj_spacing = int(round(np.mean(np.diff(traj_numbers)))) if len(traj_numbers) > 1 else 0

    # ----------------------------------------------------------
    # Save m_res.txt
    # ----------------------------------------------------------
    with open(args.output_file1, "w") as f:
        f.write("#t\tmres\tmres_err\ttau_int\ttau_int_err\ttraj_spacing\tn_traj\n")
        for i, t in enumerate(t_vals[:n_times]):
            f.write(
                f"{t}\t{ratio_mean[i]:.6e}\t{ratio_err[i]:.6e}\t"
                f"{tau_ints[i]:.6f}\t{tau_errs[i]:.6f}\t"
                f"{traj_spacing}\t{min_len}\n"
            )

    # ==========================================================
    # Plateau Fit via correlated covariance + ANALYTIC error
    # ==========================================================
    avg, err, chi2, red_chi2, cov_info, Sigma_win, ybar_win, mask_win = (
        plateau_fit_correlated_analytic_err_from_bootrep(
            ratio_mean=ratio_mean,
            ratio_bootrep=ratio_bootrep,
            t_vals=t_vals[:n_times],
            tmin=plateau_start,
            tmax=plateau_end,
            ridge=args.cov_ridge,
        )
    )

    # ==========================================================
    # Write correlation/covariance checks file next to plot
    # ==========================================================
    plot_dir = os.path.dirname(os.path.abspath(args.plot_file)) or "."
    check_path = os.path.join(plot_dir, f"plateau_t{plateau_start}_t{plateau_end}_corr_cov_checks.txt")
    write_corr_cov_checks_report(
        out_path=check_path,
        Sigma=Sigma_win,
        ridge_for_Sigma_inv=args.cov_ridge,
        eps_rel=1e-12,
    )
    print(f"✓ Saved correlation/covariance checks → {check_path}")

    # ----------------------------------------------------------
    # Save m_res_fit.txt
    # ----------------------------------------------------------
    with open(args.output_file2, "w") as f:
        f.write("#mres_fit\tmres_fit_err\tchi2\treduced_chi2\tplateau_start\tplateau_end\n")
        f.write(
            f"{avg:.6e}\t{err:.6e}\t{chi2:.6e}\t{red_chi2:.6e}\t{plateau_start}\t{plateau_end}\n"
        )

    # ==========================================================
    # Eigenbasis diagnostics (saved next to plot)
    # ==========================================================
    diag_prefix = os.path.join(plot_dir, f"plateau_t{plateau_start}_t{plateau_end}")
    t_window = t_vals[:n_times][mask_win]
    cond, lam, w, chi2_modes = covariance_eigen_diagnostics(
        Sigma=Sigma_win,
        ybar=ybar_win,
        fit_value=avg,
        t_window=t_window,
        out_prefix=diag_prefix,
        n_show_vecs=6,
        stdout_print_modes=args.eigvec_stdout,
    )

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    if args.plot_styles:
        plt.style.use(args.plot_styles)

    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    data_label = None
    if args.label == "yes":
        if args.beta is None or args.mass is None:
            data_label = r"$\beta,\ am_0$"
        else:
            data_label = rf"$\beta={args.beta},\ am_0={args.mass}$"

    fit_label = rf"$am_{{\rm res}}^{{\rm fit}} = {avg:.5f}\,\pm\,{err:.5f}$"

    title_str = (
        rf"$\alpha = {args.alpha},\ a_5/a = {args.a5},\ "
        rf"am_5 = {args.m5},\ am_{{\rm PV}} = {args.mpv}$"
    )
    ax.set_title(title_str, fontsize=10)

    ax.errorbar(
        t_vals[:n_times],
        ratio_mean,
        yerr=ratio_err,
        fmt="o",
        color="C4",
        label=data_label,
    )
    ax.axvspan(plateau_start, plateau_end, color="C2", alpha=0.2, label="Plateau range")

    ax.fill_between(
        [plateau_start, plateau_end],
        [avg - err, avg - err],
        [avg + err, avg + err],
        color="C1",
        alpha=0.25,
        linewidth=0,
    )
    ax.hlines(avg, plateau_start, plateau_end, color="C1", linestyle="--", label=fit_label)

    ax.set_xlabel("$t/a$")
    ax.set_ylabel("$am_{\\rm res}$")

    if avg > 0:
        ax.set_ylim(0.1 * avg, 5 * avg)

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if data_label or fit_label:
        ax.legend()

    plt.savefig(args.plot_file, dpi=300)
    plt.close()

    # ----------------------------------------------------------
    # Prints
    # ----------------------------------------------------------
    print(f"✓ Saved plot → {args.plot_file}")
    print(f"✓ Saved m_res.txt → {args.output_file1}")
    print(f"✓ Saved m_res_fit.txt → {args.output_file2}")
    print(f"✓ Plateau fit: chi^2 = {chi2:.4f}, reduced chi^2 = {red_chi2:.4f}")
    print(f"  Cov inversion info: {cov_info}")
    print(f"  Folding: {'ON' if args.fold else 'OFF'}")
    if args.fold:
        print(f"  Effective time extent after folding: 0..{n_times-1} (from full T={T_full})")
    print(f"  Plateau used: [{plateau_start}, {plateau_end}]")
    print("  Ratio estimator: mean(numerator)/mean(denominator)")
    print(f"  Fit error: analytic GLS sqrt(1 / (1^T Σ^-1 1))  (Σ from bootstrap replicas, n_boot={args.n_boot})")
    print("")
    print("✓ Eigenbasis diagnostics (saved next to the plot):")
    print(f"  prefix: {diag_prefix}*")
    print(f"  condition number ~ {cond:.3e}")
    top5 = np.sort(chi2_modes[np.isfinite(chi2_modes)])[::-1][:5]
    print(f"  top mode χ² contributions: {top5}")
    print("  files:")
    print(f"    {diag_prefix}_eigvals.txt/.png")
    print(f"    {diag_prefix}_modes.txt")
    print(f"    {diag_prefix}_chi2_modes.png")
    print(f"    {diag_prefix}_whitened.png")
    print(f"    {diag_prefix}_bad_eigvecs.png")
    print(f"    {diag_prefix}_eigvecs.txt")
    print(f"    {diag_prefix}_orthonorm.txt")
    print("")
    print("✓ Correlation/covariance check file:")
    print(f"    {check_path}")


if __name__ == "__main__":
    main()
