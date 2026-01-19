#!/usr/bin/env python3
"""
Extract HMC and MD statistics from Grid log-*.zst files.

For each ensemble directory, this script:
  - Reads all log-*.zst files
  - Extracts integrator info, acceptance ratio, Full BCs, trajectory times
  - Extracts unsmeared plaquette as a function of trajectory
  - Computes bootstrap means and errors
  - Computes plaquette autocorrelation time (Madras–Sokal) after thermalization
  - Applies thermalization (therm) and subsampling (delta_traj)
  - Outputs one line of statistics

Output columns:
  name fullbcs fullbcs_err therm delta_traj n_conf
  bcs bcs_err t_traj t_traj_err
  plaq plaq_err
  tau_int_plaq tau_int_plaq_err
  length_traj n_steps accept_ratio
"""

import argparse
import glob
import io
import os
import re
import numpy as np
import zstandard as zstd


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def read_zst_lines(path):
    """Yield decoded lines from a .zst file."""
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                yield line.rstrip("\n")


def bootstrap_mean_err(x, n_boot=1000, rng=None):
    """Bootstrap mean and 1-sigma error."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(means.std(ddof=1))


def slice_therm_delta(x, therm, delta, n_conf):
    """Apply thermalization cut + subsampling."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or n_conf <= 0:
        return x[:0]

    start = min(therm, x.size)
    idx = start + delta * np.arange(n_conf, dtype=int)
    idx = idx[idx < x.size]
    return x[idx]



def ms_autocorr_time(x, c=4.0):
    """Madras–Sokal integrated autocorrelation time (self-consistent window)."""
    x = np.asarray(x, np.float64)
    n = x.size
    print(n)
    if n < 8:
        return np.nan, np.nan

    # Center
    x = x - np.mean(x)

    # Variance check (use gamma(0) from data)
    gamma0 = np.mean(x * x)
    if not np.isfinite(gamma0) or gamma0 <= 1e-15:
        return np.nan, np.nan

    # FFT autocovariance (raw sums), then convert to unbiased autocovariance gamma(t)
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, nfft)
    acov_raw = np.fft.irfft(fx * np.conjugate(fx), nfft)[:n].real

    # unbiased: gamma(t) = (sum_{i=0}^{n-t-1} x_i x_{i+t}) / (n - t)
    denom = (n - np.arange(n)).astype(np.float64)
    gamma = acov_raw / denom

    # Normalized autocorrelation
    rho = gamma / gamma[0]
    rho[~np.isfinite(rho)] = 0.0
    rho = np.clip(rho, -1.0, 1.0)

    # Self-consistent window iteration: W = floor(c * tau)
    tau = 0.5
    W = 1
    for _ in range(1000):
        W = int(max(1, np.floor(c * tau)))
        if W >= n:
            W = n - 1
            break

        new_tau = 0.5 + np.sum(rho[1:W + 1])

        if not np.isfinite(new_tau) or new_tau <= 0:
            return np.nan, np.nan

        if abs(new_tau - tau) < 1e-5:
            tau = new_tau
            break

        tau = new_tau

    if not np.isfinite(tau) or tau <= 0:
        return np.nan, np.nan

    # Same style as your original error estimate
    tau_err = tau * np.sqrt((4.0 * W + 2.0) / n)
    return float(tau), float(tau_err)



# -----------------------------------------------------------------------------
# Log parsing
# -----------------------------------------------------------------------------

def extract_from_logs(log_dir):
    """Scan all log-*.zst files and extract observables."""

    zst_files = glob.glob(os.path.join(log_dir, "log-*.zst"))

    if not zst_files:
        raise FileNotFoundError(f"No log-*.zst files found under {log_dir}")

    # Sort by (jobid, run)
    zst_paths = sorted(
        zst_files,
        key=lambda s: (
            int(os.path.basename(s)
                .removeprefix("log-")
                .split("-", 1)[0]),
            int(os.path.basename(s)
                .removeprefix("log-")
                .split("-", 1)[1]
                .split(".", 1)[0])
        )
    )


    acc, rej = 0, 0
    fullbcs_raw, fullbcs_incr = [], []
    traj_times = []
    unsmeared_plaq = []
    traj_numbers = []
    traj_length, md_steps = None, None

    # Regexes
    re_acc = re.compile(r"Metropolis_test\s*--\s*ACCEPTED")
    re_rej = re.compile(r"Metropolis_test\s*--\s*REJECTED")
    re_fullbcs = re.compile(r"Full BCs\s*:\s*(\d+)")
    re_traj_time = re.compile(r"Total time for trajectory \(s\)\s*:\s*([0-9.eE+-]+)")
    re_traj_len = re.compile(r"\[Integrator\]\s*Trajectory length\s*:\s*([0-9.eE+-]+)")
    re_md_steps = re.compile(r"\[Integrator\]\s*Number of MD steps\s*:\s*(\d+)")
    re_plaq_val = re.compile(r"Plaquette:\s*\[\s*(\d+)\s*\]\s*([0-9.eE+-]+)")
    re_traj_num = re.compile(r"#\s*Trajectory\s*=\s*(\d+)")

    want_unsmeared = False
    last_bcs = None

    for path in zst_paths:
        for line in read_zst_lines(path):

            if re_acc.search(line):
                acc += 1
                continue
            if re_rej.search(line):
                rej += 1
                continue

            if (m := re_traj_len.search(line)):
                traj_length = float(m.group(1))
                continue
            if (m := re_md_steps.search(line)):
                md_steps = int(m.group(1))
                continue

            if (m := re_fullbcs.search(line)):
                val = int(m.group(1))
                fullbcs_raw.append(val)
                if last_bcs is not None:
                    diff = val - last_bcs
                    if diff >= 0:
                        fullbcs_incr.append(diff)
                last_bcs = val
                continue

            if (m := re_traj_time.search(line)):
                traj_times.append(float(m.group(1)))
                continue

            if (m := re_traj_num.search(line)):
                traj_numbers.append(int(m.group(1)))
                continue

            if "Unsmeared plaquette" in line:
                want_unsmeared = True
                continue
            if "Smeared plaquette" in line:
                want_unsmeared = False
                continue
            if want_unsmeared and (m := re_plaq_val.search(line)):
                ti = int(m.group(1))
                pv = float(m.group(2))
                unsmeared_plaq.append((ti, pv))
                continue

    accept_ratio = acc / (acc + rej) if (acc + rej) > 0 else np.nan

    return {
        "accept": acc,
        "reject": rej,
        "accept_ratio": accept_ratio,
        "fullbcs_raw": np.array(fullbcs_raw, float),
        "fullbcs_incr": np.array(fullbcs_incr, float),
        "traj_times": np.array(traj_times, float),
        "plaq": np.array(unsmeared_plaq, dtype=object),
        "traj_length": traj_length,
        "md_steps": md_steps,
        "traj_numbers": np.array(traj_numbers, int),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract HMC statistics from log-*.zst files.")
    parser.add_argument("log_dir")
    parser.add_argument("--name", required=True)
    parser.add_argument("--therm", type=int, required=True)
    parser.add_argument("--delta_traj", type=int, required=True)

    # NEW OUTPUT FILES
    parser.add_argument("--hmc_extract", required=True)
    parser.add_argument("--hmc_plaq", required=True)

    args = parser.parse_args()

    data = extract_from_logs(args.log_dir)

    # --- Determine trajectory range
    traj_numbers = data["traj_numbers"]
    if traj_numbers.size > 0:
        t_min = int(traj_numbers.min())
        t_max = int(traj_numbers.max())
        n_traj_total = t_max - t_min + 1
    else:
        lengths = [data["fullbcs_raw"].size, data["traj_times"].size]
        n_traj_total = int(max(lengths)) if lengths else 0
        t_min = 0

    # --- Build full plaquette series
    plaq_pairs = data["plaq"]
    if plaq_pairs.size > 0:
        traj_idx = plaq_pairs[:, 0].astype(int)
        plaq_vals = plaq_pairs[:, 1].astype(float)

        full_series = np.full(n_traj_total, np.nan)
        for ti, pv in zip(traj_idx, plaq_vals):
            j = ti - t_min
            if 0 <= j < n_traj_total:
                full_series[j] = pv

        # Forward fill
        for i in range(1, n_traj_total):
            if np.isnan(full_series[i]):
                full_series[i] = full_series[i - 1]

        # Fill early segment if needed
        if np.isnan(full_series[0]):
            first_valid = np.where(~np.isnan(full_series))[0]
            if first_valid.size > 0:
                full_series[:first_valid[0]] = full_series[first_valid[0]]

        # Post-therm for statistics
        plaq_post_therm_vals = full_series[args.therm:]
    else:
        full_series = np.array([])
        plaq_post_therm_vals = np.array([])

    # -------------------------------------------------------------------------
    # Write plaquette history = full series AFTER therm 
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.hmc_plaq), exist_ok=True)

    if args.therm < len(full_series):
        plaq_after_therm = full_series[args.therm:]
    else:
        plaq_after_therm = np.array([])

    mc_times = np.arange(args.therm, args.therm + len(plaq_after_therm)) + t_min

    with open(args.hmc_plaq, "w") as fpl:
        for t, pv in zip(mc_times, plaq_after_therm):
            if np.isfinite(pv):
                fpl.write(f"{t} {pv:.10g}\n")

    plaq_mean, plaq_err = bootstrap_mean_err(plaq_post_therm_vals)

    usable = max(0, n_traj_total - args.therm)
    n_conf = usable // args.delta_traj

    fullbcs_incr_s = slice_therm_delta(data["fullbcs_incr"], args.therm, args.delta_traj, n_conf)
    traj_times_s = slice_therm_delta(data["traj_times"], args.therm, args.delta_traj, n_conf)

    fullbcs_mean, fullbcs_err = bootstrap_mean_err(fullbcs_incr_s)
    bcs_mean, bcs_err = bootstrap_mean_err(fullbcs_incr_s)
    ttraj_mean, ttraj_err = bootstrap_mean_err(traj_times_s)

    tau_est, tau_err = ms_autocorr_time(plaq_post_therm_vals)

    length_traj = data["traj_length"] if data["traj_length"] is not None else np.nan
    n_steps = data["md_steps"] if data["md_steps"] is not None else np.nan
    accept_ratio = data["accept_ratio"]

    # --- Write stats file
    os.makedirs(os.path.dirname(args.hmc_extract), exist_ok=True)
    with open(args.hmc_extract, "w") as f:
        f.write(
            "name fullbcs fullbcs_err therm delta_traj n_conf "
            "bcs bcs_err t_traj t_traj_err "
            "plaq plaq_err "
            "tau_int_plaq tau_int_plaq_err "
            "length_traj n_steps accept_ratio\n"
        )
        f.write(
            f"{args.name} "
            f"{fullbcs_mean:.6g} {fullbcs_err:.6g} "
            f"{args.therm} {args.delta_traj} {n_conf} "
            f"{bcs_mean:.6g} {bcs_err:.6g} "
            f"{ttraj_mean:.6g} {ttraj_err:.6g} "
            f"{plaq_mean:.6g} {plaq_err:.6g} "
            f"{tau_est:.6g} {tau_err:.6g} "
            f"{length_traj:.6g} {n_steps} {accept_ratio:.6g}\n"
        )

    # Console summary
    print(f"[log_ensembles_extract] wrote {args.hmc_extract}")
    print(f"[log_ensembles_extract] wrote {args.hmc_plaq}")


if __name__ == "__main__":
    main()
