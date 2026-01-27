#!/usr/bin/env python3
"""
Extract HMC/MD statistics from Grid log-*.zst files and compute plaquette tau_int
using Berg's BINNING method with a hard-coded stopping criterion.

Stopping rule (HARD-CODED, applied to Berg tau):
  - c = 4.0
  - Choose FIRST bin size Nb such that:
        Nb > c * tau_BERG(Nb)
    (skip Nb=1 automatically)
  - If no such Nb exists, fall back to last valid point.

REPORTING CONVENTION (ONLY at the very end / outputs):
  - For comparison with Madras–Sokal convention, we REPORT:
        tau_report = tau_BERG / 2
        err_report = err_BERG / 2
  - IMPORTANT: the stopping-rule Nb selection uses tau_BERG (undivided),
    so Nb does NOT change due to this reporting conversion.

Always produced outputs:
  1) Writes plaquette history after thermalization to --hmc_plaq
  2) ALWAYS writes a binning table next to --hmc_plaq:
        tau_int_binning_table.txt
     and ALSO prints the same table to stdout.
  3) ALWAYS writes a plot next to --hmc_plaq:
        tau_int_vs_Nb.png
     plotting tau_report versus Nb (block size), including error bars.

PLOT STYLE (requested):
  - Tableau colorblind style
  - LaTeX-like Computer Modern math text for axis labels AND tick numbers
    (no external LaTeX required)
  - Dotted horizontal plateau line drawn OVER the points
  - Smaller markers
"""

import argparse
import glob
import io
import os
import re

import numpy as np
import zstandard as zstd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# -----------------------------------------------------------------------------
# Global plotting style (Tableau + LaTeX-like fonts + LaTeX-like tick numbers)
# -----------------------------------------------------------------------------
plt.style.use("tableau-colorblind10")
plt.rcParams.update({
    # Keep portable: no external LaTeX dependency
    "text.usetex": False,
    # Use serif + Computer Modern math look
    "font.family": "serif",
    "mathtext.fontset": "cm",
    # Make minus sign look right with serif fonts
    "axes.unicode_minus": False,
    # Sensible sizes for small figures
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def read_zst_lines(path: str):
    """Yield decoded lines from a .zst file."""
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                yield line.rstrip("\n")


def bootstrap_mean_err(x, n_boot: int = 1000, rng=None):
    """Bootstrap mean and 1-sigma error."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(means.std(ddof=1))


def slice_therm_delta(x, therm: int, delta: int, n_conf: int):
    """Apply thermalization cut + subsampling."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or n_conf <= 0:
        return x[:0]
    start = min(int(therm), x.size)
    idx = start + int(delta) * np.arange(int(n_conf), dtype=int)
    idx = idx[idx < x.size]
    return x[idx]


def _sample_var(x):
    """Unbiased sample variance; NaN if too short."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.var(x, ddof=1))


# -----------------------------------------------------------------------------
# Berg binning tau_int: series + stopping rule + outputs (always on)
# -----------------------------------------------------------------------------

def berg_binning_tau_series_berg(x, min_nbs: int = 4):
    """
    Compute arrays for Nb=1..Nb_max where Nbs=floor(N/Nb) >= min_nbs:
      - Nb_list
      - Nbs_list
      - tau_berg_list
      - err_berg_list

    Using Berg:
      tau_BERG(Nb) = (Var(binmeans)/Nbs) / (Var(raw)/N)
      err_BERG ~ tau_BERG * sqrt(2/(Nbs-1))
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = x.size
    if N < 4:
        return [], [], [], []

    var_f = _sample_var(x)
    if not np.isfinite(var_f) or var_f <= 0:
        return [], [], [], []

    s2_f = var_f / N  # Berg variance of mean (unbinned)

    Nb_list, Nbs_list, tau_list, err_list = [], [], [], []

    Nb_max = N // min_nbs
    for Nb in range(1, Nb_max + 1):
        Nbs = N // Nb
        if Nbs < min_nbs:
            break

        trimmed = x[: Nbs * Nb]
        bin_means = trimmed.reshape(Nbs, Nb).mean(axis=1)

        var_bin = _sample_var(bin_means)
        if not np.isfinite(var_bin) or var_bin <= 0:
            continue

        s2_f_Nb = var_bin / Nbs
        tau_berg = s2_f_Nb / s2_f
        err_berg = abs(tau_berg) * np.sqrt(2.0 / (Nbs - 1))

        Nb_list.append(int(Nb))
        Nbs_list.append(int(Nbs))
        tau_list.append(float(tau_berg))
        err_list.append(float(err_berg))

    return Nb_list, Nbs_list, tau_list, err_list


def find_first_nb_exceeding_c_tau_berg(Nb_list, tau_berg_list):
    """
    HARD-CODED stopping rule (applied to Berg tau):
      c = 4.0
      choose FIRST k (skipping Nb=1) such that:
          Nb[k] > c * tau_berg[k]
    Returns k, or None if not found.
    """
    C_PLATEAU = 4.0

    Nb = np.asarray(Nb_list, dtype=float)
    tau = np.asarray(tau_berg_list, dtype=float)

    if Nb.size == 0 or tau.size == 0 or Nb.size != tau.size:
        return None

    # skip Nb=1
    for k in range(1, Nb.size):
        if np.isfinite(Nb[k]) and np.isfinite(tau[k]) and tau[k] > 0:
            if Nb[k] > C_PLATEAU * tau[k]:
                return int(k)

    return None


def write_tau_table_and_plot(
    x,
    out_dir: str,
    base_name: str = "tau_int",
    min_nbs: int = 4,
):
    """
    Always:
      - compute tau_BERG series
      - choose Nb using stopping rule on tau_BERG
      - WRITE/PRINT/PLOT tau_REPORT = tau_BERG/2 (and err/2)

    Returns (reported values):
      tau_report, err_report, Nb_est, Nbs_est, found(bool)
    """
    os.makedirs(out_dir, exist_ok=True)

    Nb_list, Nbs_list, tauB_list, errB_list = berg_binning_tau_series_berg(x, min_nbs=min_nbs)

    table_path = os.path.join(out_dir, f"{base_name}_binning_table.txt")
    plot_path = os.path.join(out_dir, f"{base_name}_vs_Nb.png")

    if len(tauB_list) == 0:
        with open(table_path, "w") as f:
            f.write("# tau_int binning table\n")
            f.write("# No valid data to compute tau_int.\n")
        print("\n# tau_int binning table")
        print("# No valid data to compute tau_int.\n")
        return np.nan, np.nan, np.nan, np.nan, False

    k_choice = find_first_nb_exceeding_c_tau_berg(Nb_list, tauB_list)

    if k_choice is None:
        k_use = len(tauB_list) - 1
        found = False
    else:
        k_use = k_choice
        found = True

    # --- Reporting conversion ONLY here (and in table/plot)
    tauR_list = [0.5 * t for t in tauB_list]
    errR_list = [0.5 * e for e in errB_list]

    tau_est = tauR_list[k_use]
    tau_err = errR_list[k_use]
    Nb_est = Nb_list[k_use]
    Nbs_est = Nbs_list[k_use]

    header = [
        "# Integrated autocorrelation time (REPORTED = Berg/2)",
        "# Internally computed Berg tau_BERG(Nb) = s^2_{f_Nb} / s^2_f",
        "# where s^2_f     = Var(f)/N and s^2_{f_Nb} = Var(f_Nb)/Nbs",
        "# and err_BERG ≈ tau_BERG * sqrt(2/(Nbs-1))",
        "#",
        "# We report: tau = tau_BERG/2  and  err = err_BERG/2",
        "# NOTE: stopping rule uses tau_BERG (undivided), so Nb choice is unchanged.",
        "#",
        "# Stopping rule (on Berg tau): first k with Nb[k] > c*tau_BERG[k], c=4.0 (HARD-CODED)",
        "# Columns: Nb  Nbs  tau_int  err",
        "#",
    ]

    with open(table_path, "w") as f:
        f.write("\n".join(header) + "\n")
        for i, (Nb, Nbs, tau, er) in enumerate(zip(Nb_list, Nbs_list, tauR_list, errR_list)):
            mark = ""
            if i == k_use and found:
                mark = "  # <-- first Nb > 4*tau_BERG (USED); tau shown = tau_BERG/2"
            elif i == k_use and (not found):
                mark = "  # <-- fallback last point (USED); tau shown = tau_BERG/2"
            f.write(f"{Nb:4d} {Nbs:6d} {tau:14.6g} {er:14.6g}{mark}\n")
        f.write("#\n")
        if found:
            f.write(f"# USED: Nb={Nb_est}, Nbs={Nbs_est}, tau={tau_est:.6g} ± {tau_err:.6g}\n")
        else:
            f.write(f"# USED (no Nb > 4*tau_BERG; last point): Nb={Nb_est}, Nbs={Nbs_est}, tau={tau_est:.6g} ± {tau_err:.6g}\n")

    # Print to stdout
    print("\n# tau_int binning table (also written to file)")
    print("# Nb    Nbs        tau_int          err")
    for i, (Nb, Nbs, tau, er) in enumerate(zip(Nb_list, Nbs_list, tauR_list, errR_list)):
        tag = ""
        if i == k_use and found:
            tag = "  <-- USED (Nb chosen from Berg; tau shown = Berg/2)"
        elif i == k_use and (not found):
            tag = "  <-- USED fallback (tau shown = Berg/2)"
        print(f"{Nb:4d} {Nbs:6d} {tau:14.6g} {er:14.6g}{tag}")

    if found:
        print(f"# USED: Nb={Nb_est}, Nbs={Nbs_est}, tau={tau_est:.6g} ± {tau_err:.6g}\n")
    else:
        print(f"# USED (no Nb > 4*tau_BERG; last point): Nb={Nb_est}, Nbs={Nbs_est}, tau={tau_est:.6g} ± {tau_err:.6g}\n")

    # -------------------------------------------------------------------------
    # Plot (reporting values) — save as PDF, plateau line OVER points, same color
    # as highlighted point, and actually visible
    # -------------------------------------------------------------------------
    plot_path = os.path.join(out_dir, f"{base_name}_vs_Nb.pdf")

    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    # Pick explicit colors so nothing depends on matplotlib's cycle behavior
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    main_color = cycle[0]              # used for all points
    hi_color   = cycle[1] if len(cycle) > 1 else "C1"   # used for highlight + plateau

    # Points + error bars (force a single color for all points)
    ax.errorbar(
        Nb_list,
        tauR_list,
        yerr=errR_list,
        fmt="o",
        color=main_color,          # <-- forces marker/line color
        ecolor=main_color,         # <-- forces errorbar color
        markersize=3.0,
        markerfacecolor="none",
        capsize=2,
        elinewidth=1,
        linestyle="none",
        zorder=2,
    )

    # Chosen point: force distinct color (NOT taken from cycle)
    ax.plot(
        [Nb_est],
        [tau_est],
        marker="o",
        markersize=3.0,
        linestyle="none",
        color=hi_color,
        zorder=7,
    )

    # Plateau line: draw a "halo" underlay so dots stay visible on top of points
    dash = (0, (1.2, 2.0))  # dotted but with stronger spacing

    ax.axhline(
        tau_est,
        color=hi_color,
        linewidth=1.,       # actual visible line
        linestyle="--",
        zorder=6,            # above points/errorbars
    )

    ax.set_xlabel(r"$N_b$")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$")

    fmt = ScalarFormatter(useMathText=True)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    ax.set_xlim(0, 40.5)
    #ax.grid(True, which="major", alpha=0.25)

    fig.savefig(plot_path)   # PDF by extension
    plt.close(fig)

    print(f"[tau_int] wrote {plot_path}")
    print(f"[tau_int] wrote {table_path}")

    return tau_est, tau_err, Nb_est, Nbs_est, found


# -----------------------------------------------------------------------------
# Log parsing
# -----------------------------------------------------------------------------

def extract_from_logs(log_dir: str):
    """Scan all log-*.zst files and extract observables."""
    zst_files = glob.glob(os.path.join(log_dir, "log-*.zst"))
    if not zst_files:
        raise FileNotFoundError(f"No log-*.zst files found under {log_dir}")

    def _key(p: str):
        base = os.path.basename(p)
        stem = base.removeprefix("log-")
        a, b = stem.split("-", 1)
        jobid = int(a)
        run = int(b.split(".", 1)[0])
        return jobid, run

    zst_paths = sorted(zst_files, key=_key)

    acc, rej = 0, 0
    fullbcs_raw, fullbcs_incr = [], []
    traj_times = []
    unsmeared_plaq = []
    traj_numbers = []
    traj_length, md_steps = None, None

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
        "plaq_pairs": np.array(unsmeared_plaq, dtype=object),
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

    parser.add_argument("--hmc_extract", required=True)
    parser.add_argument("--hmc_plaq", required=True)

    args = parser.parse_args()

    data = extract_from_logs(args.log_dir)

    traj_numbers = data["traj_numbers"]
    if traj_numbers.size > 0:
        t_min = int(traj_numbers.min())
        t_max = int(traj_numbers.max())
        n_traj_total = t_max - t_min + 1
    else:
        lengths = [data["fullbcs_raw"].size, data["traj_times"].size]
        n_traj_total = int(max(lengths)) if lengths else 0
        t_min = 0

    # --- Build full plaquette series (dense per-trajectory array)
    plaq_pairs = data["plaq_pairs"]
    if plaq_pairs.size > 0 and n_traj_total > 0:
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
        if n_traj_total > 0 and np.isnan(full_series[0]):
            first_valid = np.where(~np.isnan(full_series))[0]
            if first_valid.size > 0:
                full_series[:first_valid[0]] = full_series[first_valid[0]]

        plaq_post_therm_vals = full_series[args.therm:] if args.therm < n_traj_total else np.array([])
    else:
        full_series = np.array([])
        plaq_post_therm_vals = np.array([])

    # -------------------------------------------------------------------------
    # Write plaquette history = full series AFTER therm
    # -------------------------------------------------------------------------
    plaq_dir = os.path.dirname(os.path.abspath(args.hmc_plaq)) or "."
    os.makedirs(plaq_dir, exist_ok=True)

    plaq_after_therm = full_series[args.therm:] if args.therm < len(full_series) else np.array([])
    mc_times = np.arange(args.therm, args.therm + len(plaq_after_therm)) + t_min

    with open(args.hmc_plaq, "w") as fpl:
        for t, pv in zip(mc_times, plaq_after_therm):
            if np.isfinite(pv):
                fpl.write(f"{t} {pv:.10g}\n")

    print(f"[log_ensembles_extract] wrote {args.hmc_plaq}")

    # -------------------------------------------------------------------------
    # tau_int diagnostics ALWAYS written next to plaquette history
    # -------------------------------------------------------------------------
    tau_est, tau_err, Nb_est, Nbs_est, found = write_tau_table_and_plot(
        plaq_post_therm_vals,
        out_dir=plaq_dir,
        base_name="tau_int",
        min_nbs=4,
    )

    # -------------------------------------------------------------------------
    # Other stats
    # -------------------------------------------------------------------------
    plaq_mean, plaq_err = bootstrap_mean_err(plaq_post_therm_vals)

    usable = max(0, n_traj_total - args.therm)
    n_conf = usable // args.delta_traj

    fullbcs_incr_s = slice_therm_delta(data["fullbcs_incr"], args.therm, args.delta_traj, n_conf)
    traj_times_s = slice_therm_delta(data["traj_times"], args.therm, args.delta_traj, n_conf)

    fullbcs_mean, fullbcs_err = bootstrap_mean_err(fullbcs_incr_s)
    bcs_mean, bcs_err = bootstrap_mean_err(fullbcs_incr_s)
    ttraj_mean, ttraj_err = bootstrap_mean_err(traj_times_s)

    length_traj = data["traj_length"] if data["traj_length"] is not None else np.nan
    n_steps = data["md_steps"] if data["md_steps"] is not None else np.nan
    accept_ratio = data["accept_ratio"]

    # -------------------------------------------------------------------------
    # Write stats file (tau_est/tau_err are already "reported" = Berg/2)
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.hmc_extract)) or ".", exist_ok=True)
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

    print(f"[log_ensembles_extract] wrote {args.hmc_extract}")
    if found:
        print(f"[tau_int] USED first Nb > 4*tau_BERG (Nb fixed by Berg): Nb={Nb_est}, Nbs={Nbs_est}, tau_report={tau_est:.6g} ± {tau_err:.3g}")
    else:
        print(f"[tau_int] No Nb > 4*tau_BERG found; USED last point: Nb={Nb_est}, Nbs={Nbs_est}, tau_report={tau_est:.6g} ± {tau_err:.3g}")


if __name__ == "__main__":
    main()
