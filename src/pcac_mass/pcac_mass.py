#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import glob
import os
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.style.use("tableau-colorblind10")

# ============================================================
# File Readers
# ============================================================
def read_PA0_file(filename):
    """Read real part of PA0 from mres HDF5 file. Shape: (T,)"""
    with h5py.File(filename, "r") as f:
        return f["wardIdentity/PA0"][:]["re"]


def read_ptll_file(filename, n_elems=None):
    """Read real part of the corr dataset for meson_1. Shape: (T,)"""
    with h5py.File(filename, "r") as f:
        data = f["meson/meson_1/corr"][:]
        if n_elems is None or len(data) == n_elems:
            return data["re"]
    raise ValueError(f"corr dataset does not have {n_elems} entries in {filename}")


# ============================================================
# CENTRAL DERIVATIVE (O(a^4))
# ============================================================
def central_time_derivative_oa4(data):
    """
    4th-order accurate CENTRAL time derivative along axis=1.
    f'(t) = [-f(t+2) + 8 f(t+1) - 8 f(t-1) + f(t-2)] / 12
    Periodic BC, a=1.
    data shape: (Ncfg, T)
    """
    return (
        -1.0 * np.roll(data, -2, axis=1)
        + 8.0 * np.roll(data, -1, axis=1)
        - 8.0 * np.roll(data,  1, axis=1)
        + 1.0 * np.roll(data,  2, axis=1)
    ) / 12.0


# ============================================================
# Build two half-correlators per cfg, length Nt/2 (drop t=0)
# ============================================================
def make_half_dataset_len_T2_from_full_ratio(ratio_cfg_full):
    """
    ratio_cfg_full: shape (Ncfg, T) with T even (Nt).
    Constructs two half-correlators per configuration, each of length T/2,
    using ORIGINAL timeslices t=1..T/2 inclusive (drops t=0).

    half1(k) = R(t=1+k) for k=0..T/2-1
    half2(k) = R(t=T-(1+k)) = R(T-1-k) for k=0..T/2-1
             = mapped backward half, aligned with forward half indexing.

    Returns:
      samples shape (2*Ncfg, T/2)
      t_half index array 0..T/2-1 (corresponds to original t=1..T/2)
      t_phys array of original timeslices (1..T/2)
    """
    if ratio_cfg_full.ndim != 2:
        raise ValueError("ratio_cfg_full must have shape (Ncfg, T)")
    Ncfg, T = ratio_cfg_full.shape
    if T % 2 != 0:
        raise ValueError(f"Nt (T={T}) must be even for half-length Nt/2 construction.")

    half = T // 2
    t_phys = np.arange(1, half + 1, dtype=int)      # original timeslices kept
    # Reindexed half-time (0..half-1) for plotting / plateau inputs
    t_half = np.arange(half, dtype=int)

    half1 = ratio_cfg_full[:, t_phys]               # (Ncfg, half)
    half2 = ratio_cfg_full[:, (T - t_phys) % T]     # (Ncfg, half) gives T-1..T-half

    samples = np.concatenate([half1, half2], axis=0)  # (2*Ncfg, half)
    return samples, t_half, t_phys


# ============================================================
# Bootstrap mean and error from samples
# ============================================================
def bootstrap_mean_err_from_samples(samples, n_boot=2000, seed=None):
    """
    samples shape: (Nmeas, T)
    Bootstrap resamples measurements and returns mean and std error per timeslice.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        randint = rng.integers
    else:
        randint = np.random.randint

    Nmeas, T = samples.shape
    boot_means = np.empty((n_boot, T), dtype=np.float64)
    for b in range(n_boot):
        idx = randint(0, Nmeas, size=Nmeas)
        boot_means[b] = samples[idx].mean(axis=0)

    mean = samples.mean(axis=0)
    err = boot_means.std(axis=0, ddof=1)
    return mean, err


# ============================================================
# Bootstrap plateau (from samples)
# ============================================================
def bootstrap_plateau_from_samples(samples, t_half, tmin, tmax, n_boot=2000, seed=None):
    """
    Plateau on reindexed half-time t_half = 0..half-1.
    samples shape: (Nmeas, half)
    """
    mask = (t_half >= tmin) & (t_half <= tmax)
    if not np.any(mask):
        raise ValueError(f"No points in plateau range {tmin}–{tmax}")
    if seed is not None:
        rng = np.random.default_rng(seed)
        randint = rng.integers
    else:
        randint = np.random.randint

    Nmeas, _ = samples.shape
    A = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = randint(0, Nmeas, size=Nmeas)
        mean_b = samples[idx].mean(axis=0)
        A[b] = mean_b[mask].mean()
    return A.mean(), A.std(ddof=1)


# ============================================================
# Trajectory extraction
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
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute PCAC ratio R(t)=0.5*d_t PA0(t)/P(t), then create two "
            "independent half-correlators per cfg of length Nt/2 by dropping t=0 "
            "and taking t=1..Nt/2. Bootstrap over the 2*Ncfg half-correlators."
        )
    )
    parser.add_argument("input_dir")
    parser.add_argument("--label", default="", help="yes → include β, am0 label on plot")
    parser.add_argument("--output_file1", required=True)
    parser.add_argument("--output_file2", required=True)
    parser.add_argument("--plot_file", required=True)
    parser.add_argument("--plot_styles", default="")
    parser.add_argument("--plateau_start", type=float, required=True, help="in half-time index (0..Nt/2-1)")
    parser.add_argument("--plateau_end", type=float, required=True, help="in half-time index (0..Nt/2-1)")
    parser.add_argument("--beta", type=float, default="")
    parser.add_argument("--mass", type=float, default="")

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--a5", type=float, required=True)
    parser.add_argument("--m5", type=float, required=True)
    parser.add_argument("--mpv", type=float, required=True)

    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    plateau_start = int(round(args.plateau_start))
    plateau_end = int(round(args.plateau_end))

    # ----------------------------------------------------------
    # Load files
    # ----------------------------------------------------------
    pa0_files = sorted(glob.glob(os.path.join(args.input_dir, "mres.*.h5")))
    ptll_files = sorted(glob.glob(os.path.join(args.input_dir, "pt_ll.*.h5")))
    if not pa0_files or not ptll_files:
        raise FileNotFoundError("Missing mres.*.h5 or pt_ll.*.h5 files.")

    PA0_data = np.array([read_PA0_file(f) for f in pa0_files])  # (Ncfg, Nt)
    Nt = PA0_data.shape[1]
    P_data = np.array([read_ptll_file(f, n_elems=Nt) for f in ptll_files])  # (Ncfg, Nt)

    min_len = min(len(PA0_data), len(P_data))
    PA0_data = PA0_data[:min_len]
    P_data = P_data[:min_len]

    if Nt % 2 != 0:
        raise ValueError(f"Nt must be even for your requested construction; got Nt={Nt}")

    half = Nt // 2
    plateau_end = half - 1
    # Your constraint #2:
    if plateau_end > half - 1:
        raise ValueError(f"plateau_end={plateau_end} exceeds max allowed {half-1} (Nt/2-1).")
    if plateau_start < 0 or plateau_start > plateau_end:
        raise ValueError("Invalid plateau range. Need 0 <= plateau_start <= plateau_end <= Nt/2-1.")

    # ----------------------------------------------------------
    # Ratio per configuration (full Nt)
    # ----------------------------------------------------------
    dPA0_data = central_time_derivative_oa4(PA0_data)      # (Ncfg, Nt)
    ratio_cfg_full = 0.5 * dPA0_data / P_data              # (Ncfg, Nt)

    # ----------------------------------------------------------
    # Build half dataset: 2*Ncfg samples, each of length Nt/2 (drop t=0)
    # ----------------------------------------------------------
    samples, t_half, t_phys = make_half_dataset_len_T2_from_full_ratio(ratio_cfg_full)

    # ----------------------------------------------------------
    # Bootstrap mean & error (over 2*Ncfg samples)
    # ----------------------------------------------------------
    ratio_mean, ratio_err = bootstrap_mean_err_from_samples(
        samples, n_boot=args.n_boot, seed=args.seed
    )

    # ----------------------------------------------------------
    # Output file 1
    #   We write both half-index (0..half-1) and corresponding original t (1..half)
    # ----------------------------------------------------------
    traj_numbers = extract_trajectory_numbers(pa0_files[:min_len])
    traj_spacing = int(round(np.mean(np.diff(traj_numbers)))) if len(traj_numbers) > 1 else 0

    with open(args.output_file1, "w") as f:
        f.write("#t_half\t t_phys\t pcac\t pcac_err\t traj_spacing\t n_traj\n")
        for i in range(half):
            f.write(
                f"{t_half[i]}\t{t_phys[i]}\t{ratio_mean[i]:.6e}\t{ratio_err[i]:.6e}\t"
                f"{traj_spacing}\t{min_len}\n"
            )

    # ----------------------------------------------------------
    # Plateau
    # ----------------------------------------------------------
    avg, err = bootstrap_plateau_from_samples(
        samples, t_half, plateau_start, plateau_end, n_boot=args.n_boot, seed=args.seed
    )

    with open(args.output_file2, "w") as f:
        f.write("#mpcac_fit\tmpcac_fit_err\treduced_chi2\tplateau_start\tplateau_end\n")
        f.write(f"{avg:.6e}\t{err:.6e}\tNaN\t{plateau_start}\t{plateau_end}\n")

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    if args.plot_styles:
        plt.style.use(args.plot_styles)

    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    data_label = rf"$\beta={args.beta},\ am_0={args.mass}$" if args.label == "yes" else None
    fit_label = rf"$am_{{\rm PCAC}}^{{\rm fit}} = {avg:.5f}\,\pm\,{err:.5f}$"

    title_str = (
        rf"$\alpha = {args.alpha},\ a_5/a = {args.a5},\ "
        rf"am_5 = {args.m5},\ am_{{\rm PV}} = {args.mpv}$"
    )
    ax.set_title(title_str, fontsize=10)

    ax.errorbar(t_half, ratio_mean, yerr=ratio_err, fmt="o", color="C4", label=data_label)
    ax.axvspan(plateau_start, plateau_end, color="C2", alpha=0.2, label="Plateau range")

    ax.fill_between(
        [plateau_start, plateau_end],
        [avg - err, avg - err],
        [avg + err, avg + err],
        color="C1", alpha=0.25, linewidth=0
    )
    ax.hlines(avg, plateau_start, plateau_end, color="C1", linestyle="--", label=fit_label)

    ax.set_xlabel(r"$t/a$")
    ax.set_ylabel(r"$am_{\rm PCAC}$")

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if data_label or fit_label:
        ax.legend()

    plt.savefig(args.plot_file, dpi=300)
    plt.close()

    print(f"✓ Saved plot → {args.plot_file}")
    print(f"✓ Saved half-dataset mpcac(t) → {args.output_file1}")
    print(f"✓ Saved plateau fit → {args.output_file2}")


if __name__ == "__main__":
    main()
