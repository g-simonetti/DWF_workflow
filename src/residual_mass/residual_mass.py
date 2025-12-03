#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import glob
import os
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from collections import Counter

plt.style.use("tableau-colorblind10")

# ============================================================
# File Readers
# ============================================================
def read_mres_file(filename):
    """Read real part of PJ5q from mres HDF5 file."""
    with h5py.File(filename, "r") as f:
        return f["wardIdentity/PJ5q"][:]["re"]


def read_ptll_file(filename, n_elems=None):
    """Read real part of the corr dataset for meson_1."""
    with h5py.File(filename, "r") as f:
        data = f["meson/meson_1/corr"][:]
        if n_elems is None or len(data) == n_elems:
            return data["re"]
    raise ValueError(f"corr dataset does not have {n_elems} entries in {filename}")


# ============================================================
# Bootstrap
# ============================================================
def bootstrap_ratio(data1, data2, n_boot=2000):

    Ncfg = data1.shape[0]
    T = data1.shape[1]

    ratios_boot = np.empty((n_boot, T))

    for b in range(n_boot):
        # sample configuration indices with replacement
        idx = np.random.randint(0, Ncfg, size=Ncfg)

        # bootstrap means
        mean1 = data1[idx].mean(axis=0)
        mean2 = data2[idx].mean(axis=0)

        # ratio of bootstrap means
        ratios_boot[b] = mean1 / mean2

    # central value = ratio of original means
    ratio_mean = data1.mean(axis=0) / data2.mean(axis=0)

    # bootstrap standard deviation
    ratio_err = ratios_boot.std(axis=0, ddof=1)

    return ratio_mean, ratio_err



# ============================================================
# Madras–Sokal Autocorrelation
# ============================================================
def integrated_autocorrelation_time(x, c=5.0, M_max=50):
    """Madras–Sokal automatic window selection for τ_int."""
    x = np.asarray(x)
    n = len(x)
    if n < 2:
        return 0.5, 0.0

    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return 0.5, 0.0

    acf = np.correlate(x, x, mode="full")[n - 1:] / (var * np.arange(n, 0, -1))
    candidate_M = range(1, min(M_max, n - 1) + 1)
    tau_int_M = np.array([0.5 + np.sum(acf[1:M + 1]) for M in candidate_M])

    valid = [M for M, tau in zip(candidate_M, tau_int_M) if M >= c * tau]
    M_selected = min(valid) if valid else candidate_M[-1]

    tau = 0.5 + np.sum(acf[1:M_selected + 1])
    tau_err = np.sqrt(2.0 * (M_selected + 1) / n) * tau
    return tau, tau_err


# ============================================================
# Trajectory Number
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
# Plateau Fit
# ============================================================
def plateau_fit(t, y, yerr, tmin, tmax):
    """Weighted average + reduced chi²."""
    mask = (t >= tmin) & (t <= tmax)
    if not np.any(mask):
        raise ValueError(f"No points in plateau range {tmin}–{tmax}")

    y_sel = y[mask]
    yerr_sel = yerr[mask]

    weights = 1 / yerr_sel**2
    avg = np.average(y_sel, weights=weights)
    err = np.sqrt(1 / weights.sum())

    chi2 = np.sum(((y_sel - avg) / yerr_sel) ** 2)
    dof = len(y_sel) - 1
    chi2_red = chi2 / dof if dof > 0 else np.nan

    if chi2_red > 1:
        err *= np.sqrt(chi2_red)

    return avg, err, chi2_red


# ============================================================
# Main Script
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Compute residual mass from HDF5 data.")
    parser.add_argument("input_dir")
    parser.add_argument("--label", default="", help="yes → include β, am0 label on plot")
    parser.add_argument("--output_file1", required=True)
    parser.add_argument("--output_file2", required=True)
    parser.add_argument("--plot_file", required=True)
    parser.add_argument("--plot_styles", default="")
    parser.add_argument("--plateau_start", type=float, required=True)
    parser.add_argument("--plateau_end", type=float, required=True)
    parser.add_argument("--beta", type=float, default="")
    parser.add_argument("--mass", type=float, default="")

    # ====================================================
    # NEW: parameters for title (minimal change)
    # ====================================================
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--a5", type=float, required=True)
    parser.add_argument("--m5", type=float, required=True)
    parser.add_argument("--mpv", type=float, required=True)

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

    mres_data = np.array([read_mres_file(f) for f in mres_files])
    n_times = mres_data.shape[1]
    ptll_data = np.array([read_ptll_file(f, n_elems=n_times) for f in ptll_files])

    # Match lengths
    min_len = min(len(mres_data), len(ptll_data))
    mres_data = mres_data[:min_len]
    ptll_data = ptll_data[:min_len]

    # ----------------------------------------------------------
    # Bootstrap Ratio
    # ----------------------------------------------------------
    ratio_mean, ratio_err = bootstrap_ratio(mres_data, ptll_data)

    # ----------------------------------------------------------
    # Autocorrelation
    # ----------------------------------------------------------
    ratio_t = mres_data / ptll_data
    tau_ints, tau_errs = [], []
    for t in range(n_times):
        tau, err = integrated_autocorrelation_time(ratio_t[:, t])
        tau_ints.append(tau)
        tau_errs.append(err)
    tau_ints = np.array(tau_ints)
    tau_errs = np.array(tau_errs)

    # ----------------------------------------------------------
    # Trajectories
    # ----------------------------------------------------------
    traj_numbers = extract_trajectory_numbers(mres_files[:min_len])
    traj_spacing = int(round(np.mean(np.diff(traj_numbers)))) if len(traj_numbers) > 1 else 0

    # ----------------------------------------------------------
    # Save m_res.txt
    # ----------------------------------------------------------
    with open(args.output_file1, "w") as f:
        f.write("#t\tmres\tmres_err\ttau_int\ttau_int_err\ttraj_spacing\tn_traj\n")
        for t in range(n_times):
            f.write(
                f"{t}\t{ratio_mean[t]:.6e}\t{ratio_err[t]:.6e}\t"
                f"{tau_ints[t]:.6f}\t{tau_errs[t]:.6f}\t"
                f"{traj_spacing}\t{min_len}\n"
            )

    # ----------------------------------------------------------
    # Plateau Fit
    # ----------------------------------------------------------
    t_vals = np.arange(n_times)
    avg, err, chi2_red = plateau_fit(t_vals, ratio_mean, ratio_err, plateau_start, plateau_end)

    with open(args.output_file2, "w") as f:
        f.write("#mres_fit\tmres_fit_err\treduced_chi2\tplateau_start\tplateau_end\n")
        f.write(f"{avg:.6e}\t{err:.6e}\t{chi2_red:.3f}\t{plateau_start}\t{plateau_end}\n")

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    if args.plot_styles:
        plt.style.use(args.plot_styles)

    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    data_label = rf"$\beta={args.beta},\ am_0={args.mass}$" if args.label == "yes" else None
    fit_label = rf"$am_{{\rm res}}^{{\rm fit}} = {avg:.5f}\,\pm\,{err:.5f}$"

    # ====================================================
    # Title
    # ====================================================
    title_str = (
        rf"$\alpha = {args.alpha},\ a_5/a = {args.a5},\ "
        rf"am_5 = {args.m5},\ am_{{\rm PV}} = {args.mpv}$"
    )
    ax.set_title(title_str, fontsize=10)

    ax.errorbar(t_vals, ratio_mean, yerr=ratio_err, fmt="o", color="C4", label=data_label)
    ax.axvspan(plateau_start, plateau_end, color="C2", alpha=0.2, label="Plateau range")

    ax.fill_between([plateau_start, plateau_end], [avg - err, avg - err], [avg + err, avg + err],
        color="C1", alpha=0.25, linewidth=0)

    ax.hlines(avg, plateau_start, plateau_end, color="C1", linestyle="--", label=fit_label)

    ax.set_xlabel("$t/a$")
    ax.set_ylabel("$am_{\\rm res}$")

    # ---- Scientific notation for clean axis ----
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    #ax.set_ylim(0.3e-3, 2.3e-3)

    if data_label or fit_label:
        ax.legend()

    plt.savefig(args.plot_file, dpi=300)
    plt.close()

    print(f"✓ Saved plot → {args.plot_file}")
    print(f"✓ Saved m_res.txt → {args.output_file1}")
    print(f"✓ Saved m_res_fit.txt → {args.output_file2}")


if __name__ == "__main__":
    main()
