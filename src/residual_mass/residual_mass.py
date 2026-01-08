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

    ratios_cfg = data1 / data2
    Ncfg, T = ratios_cfg.shape
    ratios_boot = np.empty((n_boot, T))

    for b in range(n_boot):
        idx = np.random.randint(0, Ncfg, size=Ncfg)
        ratios_boot[b] = ratios_cfg[idx].mean(axis=0)

    ratio_mean = ratios_cfg.mean(axis=0)
    ratio_err = ratios_boot.std(axis=0, ddof=1)

    return ratio_mean, ratio_err, ratios_cfg


# ============================================================
# Autocorrelation
# ============================================================
def integrated_autocorrelation_time(x, c=5.0, M_max=50):
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
# Bootstrap plateau fit 
# ============================================================
def bootstrap_plateau(ratios_cfg, t_vals, tmin, tmax, n_boot=2000):
    """
    Bootstrap over configurations to get plateau value and error.
    No covariance matrix needed.
    """
    Ncfg, T = ratios_cfg.shape
    mask = (t_vals >= tmin) & (t_vals <= tmax)
    if not np.any(mask):
        raise ValueError(f"No points in plateau range {tmin}–{tmax}")

    A = np.empty(n_boot)

    for b in range(n_boot):
        idx = np.random.randint(0, Ncfg, size=Ncfg)
        Rb = ratios_cfg[idx].mean(axis=0)      # mean R(t) for this bootstrap sample
        A[b] = Rb[mask].mean()                 # plateau average

    return A.mean(), A.std(ddof=1)


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

    # NEW TITLE ARGS
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

    min_len = min(len(mres_data), len(ptll_data))
    mres_data = mres_data[:min_len]
    ptll_data = ptll_data[:min_len]

    # ----------------------------------------------------------
    # Bootstrap Ratio 
    # ----------------------------------------------------------
    ratio_mean, ratio_err, ratios_cfg = bootstrap_ratio(mres_data, ptll_data)

    # ----------------------------------------------------------
    # Autocorrelation 
    # ----------------------------------------------------------
    ratio_t = ratios_cfg
    tau_ints, tau_errs = [], []
    for t in range(n_times):
        tau, err = integrated_autocorrelation_time(ratio_t[:, t])
        tau_ints.append(tau)
        tau_errs.append(err)
    tau_ints = np.array(tau_ints)
    tau_errs = np.array(tau_errs)

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
        for t in range(n_times):
            f.write(
                f"{t}\t{ratio_mean[t]:.6e}\t{ratio_err[t]:.6e}\t"
                f"{tau_ints[t]:.6f}\t{tau_errs[t]:.6f}\t"
                f"{traj_spacing}\t{min_len}\n"
            )

    # ==========================================================
    # Plateau Fit via Bootstrap 
    # ==========================================================
    t_vals = np.arange(n_times)
    avg, err = bootstrap_plateau(ratios_cfg, t_vals, plateau_start, plateau_end)

    # ----------------------------------------------------------
    # Save m_res_fit.txt
    # ----------------------------------------------------------
    with open(args.output_file2, "w") as f:
        f.write("#mres_fit\tmres_fit_err\treduced_chi2\tplateau_start\tplateau_end\n")
        f.write(f"{avg:.6e}\t{err:.6e}\tNaN\t{plateau_start}\t{plateau_end}\n")

    # ----------------------------------------------------------
    # Plot 
    # ----------------------------------------------------------
    if args.plot_styles:
        plt.style.use(args.plot_styles)

    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    data_label = rf"$\beta={args.beta},\ am_0={args.mass}$" if args.label == "yes" else None
    fit_label = rf"$am_{{\rm res}}^{{\rm fit}} = {avg:.5f}\,\pm\,{err:.5f}$"

    title_str = (
        rf"$\alpha = {args.alpha},\ a_5/a = {args.a5},\ "
        rf"am_5 = {args.m5},\ am_{{\rm PV}} = {args.mpv}$"
    )
    ax.set_title(title_str, fontsize=10)

    ax.errorbar(t_vals, ratio_mean, yerr=ratio_err, fmt="o", color="C4", label=data_label)
    ax.axvspan(plateau_start, plateau_end, color="C2", alpha=0.2, label="Plateau range")

    ax.fill_between(
        [plateau_start, plateau_end], [avg - err, avg - err], [avg + err, avg + err],
        color="C1", alpha=0.25, linewidth=0
    )

    ax.hlines(avg, plateau_start, plateau_end, color="C1", linestyle="--", label=fit_label)

    ax.set_xlabel("$t/a$")
    ax.set_ylabel("$am_{\\rm res}$")

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if data_label or fit_label:
        ax.legend()

    plt.savefig(args.plot_file, dpi=300)
    plt.close()

    print(f"✓ Saved plot → {args.plot_file}")
    print(f"✓ Saved m_res.txt → {args.output_file1}")
    print(f"✓ Saved m_res_fit.txt → {args.output_file2}")


if __name__ == "__main__":
    main()
