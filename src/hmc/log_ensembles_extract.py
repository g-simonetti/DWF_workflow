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
    """
    Apply thermalization cut, subsample by delta, and return up to n_conf entries.

    therm and delta are in "trajectory units" (same as in the Snakefile).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0 or n_conf <= 0:
        return x[:0]

    # Drop the first `therm` entries
    start = min(therm, x.size)

    # Indices: start, start + delta, start + 2*delta, ...
    idx = start + delta * np.arange(n_conf, dtype=int)
    idx = idx[idx < x.size]  # safety if x is shorter than expected

    return x[idx]


def ms_autocorr_time(x, c=5.0):
    """
    Compute the Madras–Sokal integrated autocorrelation time and its analytic error.
    """
    x = np.asarray(x, np.float64)
    n = x.size
    if n < 8:
        return np.nan, np.nan

    x = x - np.mean(x)
    var = np.var(x)
    if not np.isfinite(var) or var <= 1e-15:
        return np.nan, np.nan

    # FFT-based autocovariance
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(fx * np.conjugate(fx), nfft)[:n]
    rho = np.real(acov / acov[0])
    rho[np.isnan(rho)] = 0.0
    rho = np.clip(rho, -1.0, 1.0)

    # Cut negative tail
    neg = np.where(rho < 0)[0]
    if len(neg) > 0:
        rho[neg[0]:] = 0.0

    tau = 0.5
    for _ in range(1000):
        W = int(max(1, np.floor(c * tau)))
        if W >= n:
            break
        new_tau = 0.5 + np.sum(rho[1:W + 1])
        if abs(new_tau - tau) < 1e-5:
            tau = new_tau
            break
        tau = new_tau

    if not np.isfinite(tau) or tau <= 0.0:
        return np.nan, np.nan

    tau_err = tau * np.sqrt((4 * (2 * W + 1)) / n)
    return float(tau), float(tau_err)


# -----------------------------------------------------------------------------
# Log parsing
# -----------------------------------------------------------------------------

def extract_from_logs(log_dir):
    """Scan all log-*.zst files and extract observables."""
    zst_paths = sorted(glob.glob(os.path.join(log_dir, "log-*.zst")))
    if not zst_paths:
        raise FileNotFoundError(f"No log-*.zst files found under {log_dir}")

    acc, rej = 0, 0
    fullbcs_raw, fullbcs_incr = [], []
    traj_times = []
    unsmeared_plaq = []        # list of (traj_idx, plaq_value)
    traj_numbers = []          # list of trajectory numbers from "# Trajectory = N"
    traj_length, md_steps = None, None

    # Regexes
    re_acc = re.compile(r"Metropolis_test\s*--\s*ACCEPTED")
    re_rej = re.compile(r"Metropolis_test\s*--\s*REJECTED")
    re_fullbcs = re.compile(r"Full BCs\s*:\s*(\d+)")
    re_traj_time = re.compile(r"Total time for trajectory \(s\)\s*:\s*([0-9.eE+-]+)")
    re_traj_len = re.compile(r"\[Integrator\]\s*Trajectory length\s*:\s*([0-9.eE+-]+)")
    re_md_steps = re.compile(r"\[Integrator\]\s*Number of MD steps\s*:\s*(\d+)")
    re_plaq_val = re.compile(r"Plaquette:\s*\[\s*(\d+)\s*\]\s*([0-9.eE+-]+)")
    re_traj_num = re.compile(r"#\s*Trajectory\s*=\s*(\d+)")  # from "Grid : HMC : ... # Trajectory = N"

    want_unsmeared = False
    last_bcs = None

    for path in zst_paths:
        for line in read_zst_lines(path):
            # Metropolis acceptance / rejection
            if re_acc.search(line):
                acc += 1
                continue
            if re_rej.search(line):
                rej += 1
                continue

            # Integrator info
            if (m := re_traj_len.search(line)):
                traj_length = float(m.group(1))
                continue
            if (m := re_md_steps.search(line)):
                md_steps = int(m.group(1))
                continue

            # Full BCs
            if (m := re_fullbcs.search(line)):
                val = int(m.group(1))
                fullbcs_raw.append(val)
                if last_bcs is not None:
                    diff = val - last_bcs
                    if diff >= 0:
                        fullbcs_incr.append(diff)
                last_bcs = val
                continue

            # Trajectory time
            if (m := re_traj_time.search(line)):
                traj_times.append(float(m.group(1)))
                continue

            # Trajectory number
            if (m := re_traj_num.search(line)):
                traj_numbers.append(int(m.group(1)))
                continue

            # Plaquette sections
            if "Unsmeared plaquette" in line:
                want_unsmeared = True
                continue
            if "Smeared plaquette" in line:
                want_unsmeared = False
                continue
            if want_unsmeared and (m := re_plaq_val.search(line)):
                traj_idx = int(m.group(1))
                val = float(m.group(2))
                unsmeared_plaq.append((traj_idx, val))
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
# Main entry point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract HMC statistics from log-*.zst files.")
    parser.add_argument("log_dir", help="Directory containing log-*.zst files")
    parser.add_argument("--name", required=True, help="Ensemble name (added to output row).")
    parser.add_argument("--therm", type=int, required=True, help="Thermalization cut (trajectories)")
    parser.add_argument("--delta_traj", type=int, required=True, help="Subsampling stride (trajectories)")
    parser.add_argument("--output_file", required=True, help="Output file path")
    args = parser.parse_args()

    data = extract_from_logs(args.log_dir)

    # --- Determine total number of trajectories from "# Trajectory = N" ---
    traj_numbers = data["traj_numbers"]
    if traj_numbers.size > 0:
        t_min = int(traj_numbers.min())
        t_max = int(traj_numbers.max())
        n_traj_total = t_max - t_min + 1
    else:
        # Fallback: use lengths of other arrays (not ideal, but better than nothing)
        lengths = [
            data["fullbcs_raw"].size,
            data["traj_times"].size,
        ]
        n_traj_total = int(max(lengths)) if lengths else 0
        t_min = 0

    # --- Build full plaquette time series including rejected steps ---
    plaq_pairs = data["plaq"]
    if plaq_pairs.size > 0:
        traj_idx = plaq_pairs[:, 0].astype(int)
        plaq_vals = plaq_pairs[:, 1].astype(float)

        # Map trajectory indices to [0, n_traj_total)
        # We assume plaq indices are consistent with traj_numbers
        full_series = np.full(n_traj_total, np.nan, dtype=float)
        for ti, pv in zip(traj_idx, plaq_vals):
            j = ti - t_min
            if 0 <= j < n_traj_total:
                full_series[j] = pv

        # Forward-fill missing plaquette entries
        for i in range(1, n_traj_total):
            if np.isnan(full_series[i]):
                full_series[i] = full_series[i - 1]

        # Drop any leading NaNs (if we never saw a plaquette at the very start)
        if np.isnan(full_series[0]):
            first_valid = np.where(~np.isnan(full_series))[0]
            if first_valid.size > 0:
                full_series[:first_valid[0]] = full_series[first_valid[0]]

        # Apply thermalization cut on trajectory index
        if args.therm < n_traj_total:
            plaq_post_therm_vals = full_series[args.therm:]
        else:
            plaq_post_therm_vals = np.array([], dtype=float)
    else:
        plaq_post_therm_vals = np.array([], dtype=float)

    # Average plaquette and bootstrap error (post-therm)
    plaq_mean, plaq_err = bootstrap_mean_err(plaq_post_therm_vals)

    # --- Compute n_conf based on trajectory count, therm and delta_traj ---
    usable = max(0, n_traj_total - args.therm)
    n_conf = usable // args.delta_traj

    # Slice other observables with the same therm and delta_traj
    fullbcs_incr_s = slice_therm_delta(
        data["fullbcs_incr"], args.therm, args.delta_traj, n_conf
    )
    traj_times_s = slice_therm_delta(
        data["traj_times"], args.therm, args.delta_traj, n_conf
    )

    fullbcs_mean, fullbcs_err = bootstrap_mean_err(fullbcs_incr_s)
    bcs_mean, bcs_err = bootstrap_mean_err(fullbcs_incr_s)  # same quantity here
    ttraj_mean, ttraj_err = bootstrap_mean_err(traj_times_s)

    # Plaquette autocorrelation time (post-therm)
    tau_est, tau_err = ms_autocorr_time(plaq_post_therm_vals)

    length_traj = data["traj_length"] if data["traj_length"] is not None else np.nan
    n_steps = data["md_steps"] if data["md_steps"] is not None else np.nan
    accept_ratio = data["accept_ratio"]

    # --- Write output file ---
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
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

    # --- Console summary ---
    print(f"[log_ensembles_extract] wrote {args.output_file}")
    print(f"  Ensemble name: {args.name}")
    print(f"  Accept/Reject: {data['accept']}/{data['reject']} (ratio={accept_ratio:.3f})")
    print(f"  Trajectories: t_min={t_min}, n_traj_total={n_traj_total}")
    print(
        f"  therm={args.therm}, delta_traj={args.delta_traj}, "
        f"usable={usable}, n_conf={n_conf}"
    )
    print(
        f"  fullbcs(incr)={fullbcs_mean:.6g}±{fullbcs_err:.6g}, "
        f"bcs(incr)={bcs_mean:.6g}±{bcs_err:.6g}, "
        f"t_traj={ttraj_mean:.6g}±{ttraj_err:.6g}, "
        f"plaq={plaq_mean:.6g}±{plaq_err:.6g}, "
        f"tau_plaq={tau_est:.6g}±{tau_err:.6g}, "
        f"length_traj={length_traj}, n_steps={n_steps}, "
        f"accept_ratio={accept_ratio:.3f}"
    )

    if np.isfinite(tau_est) and tau_est < args.delta_traj:
        print(f"  ✅ decorrelation OK: tau_int={tau_est:.3g} < delta_traj={args.delta_traj}")
    else:
        print(f"  ⚠️ possible under-decorrelation: tau_int={tau_est:.3g}, delta_traj={args.delta_traj}")


if __name__ == "__main__":
    main()