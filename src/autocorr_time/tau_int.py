#!/usr/bin/env python3
"""
Reusable integrated autocorrelation time (tau_int) via Berg BINNING method
with a hard-coded stopping rule:

  choose FIRST Nb (skipping Nb=1) such that:
      Nb > 4 * tau_BERG(Nb)

Reporting:
  tau_report = tau_BERG / 2
  err_report = err_BERG / 2

I/O behavior (UPDATED):
  - Produces ONE JSON output file containing what used to be in the TXT table
    plus the final estimate and scan data:
        {out_dir}/{base_name}_results.json

  - Still produces plots:
        {out_dir}/{base_name}_vs_Nb.pdf
        {out_dir}/{base_name}_vs_n_therm.pdf

No .txt output files are produced.

Public API:
  - compute_tau_from_file(input_file, out_dir, therm, plot_styles=None, base_name="tau_int")

CLI:
  python3 tau_int.py <input_file> <out_dir> <therm> [--plot_styles ...]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------

def apply_plot_styles(plot_styles_arg: str | None):
    if not plot_styles_arg:
        return
    parts = [p.strip() for p in str(plot_styles_arg).split(",") if p.strip()]
    if parts:
        plt.style.use(parts)


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def read_series_file(path: str) -> np.ndarray:
    """
    Reads a series from a text file.
      - If line has 1 token: y
      - If line has >=2 tokens: interpret as (t, y, ...) and use SECOND token as y
    Ignores blank lines and lines starting with '#'.
    """
    ys: list[float] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            try:
                y = float(parts[0]) if len(parts) == 1 else float(parts[1])
            except ValueError:
                continue
            if np.isfinite(y):
                ys.append(y)
    return np.asarray(ys, dtype=float)


# -----------------------------------------------------------------------------
# Berg binning tau_int
# -----------------------------------------------------------------------------

def _sample_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.var(x, ddof=1))


def berg_binning_tau_series_berg(x: np.ndarray, min_nbs: int = 4):
    """
    For Nb=1..Nb_max where Nbs=floor(N/Nb) >= min_nbs compute:
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

    s2_f = var_f / N

    Nb_list: list[int] = []
    Nbs_list: list[int] = []
    tau_list: list[float] = []
    err_list: list[float] = []

    Nb_max = N // int(min_nbs)

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


def find_first_nb_exceeding_c_tau_berg(
    Nb_list: list[int],
    tau_berg_list: list[float],
    c_plateau: float = 4.0,
):
    Nb = np.asarray(Nb_list, dtype=float)
    tau = np.asarray(tau_berg_list, dtype=float)
    if Nb.size == 0 or tau.size == 0 or Nb.size != tau.size:
        return None
    for k in range(1, Nb.size):  # skip Nb=1
        if np.isfinite(Nb[k]) and np.isfinite(tau[k]) and tau[k] > 0:
            if Nb[k] > c_plateau * tau[k]:
                return int(k)
    return None


def compute_tau_from_series(
    x: np.ndarray,
    min_nbs: int = 4,
    c_plateau: float = 4.0,
):
    """
    Returns:
      tau_est, tau_err, Nb_est, Nbs_est, found, used_index,
      (Nb_list, Nbs_list, tauR_list, errR_list)

    where tauR = tau_BERG/2, errR = err_BERG/2
    """
    Nb_list, Nbs_list, tauB_list, errB_list = berg_binning_tau_series_berg(x, min_nbs=min_nbs)
    if len(tauB_list) == 0:
        return np.nan, np.nan, None, None, False, None, ([], [], [], [])

    k_choice = find_first_nb_exceeding_c_tau_berg(Nb_list, tauB_list, c_plateau=c_plateau)
    if k_choice is None:
        k_use = len(tauB_list) - 1
        found = False
    else:
        k_use = k_choice
        found = True

    tauR_list = [0.5 * t for t in tauB_list]
    errR_list = [0.5 * e for e in errB_list]

    tau_est = float(tauR_list[k_use])
    tau_err = float(errR_list[k_use])
    Nb_est = int(Nb_list[k_use])
    Nbs_est = int(Nbs_list[k_use])

    return tau_est, tau_err, Nb_est, Nbs_est, found, k_use, (Nb_list, Nbs_list, tauR_list, errR_list)


# -----------------------------------------------------------------------------
# Plot outputs + JSON payload assembly
# -----------------------------------------------------------------------------

def plot_tau_vs_nb(
    Nb_list: list[int],
    tauR_list: list[float],
    errR_list: list[float],
    Nb_est: int,
    tau_est: float,
    plot_path: str,
):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    ax.errorbar(
        Nb_list, tauR_list, yerr=errR_list,
        fmt="o", markersize=3.0, markerfacecolor="none",
        capsize=2, elinewidth=1, linestyle="none", zorder=2
    )
    ax.plot([Nb_est], [tau_est], marker="o", markersize=3.0, linestyle="none", zorder=4)
    ax.axhline(tau_est, linewidth=1.0, linestyle="--", zorder=3)

    ax.set_xlabel(r"$N_b$")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$")

    fmt = ScalarFormatter(useMathText=True)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    fig.savefig(plot_path)
    plt.close(fig)


def scan_tau_vs_n_therm(
    y_full: np.ndarray,
    min_nbs: int = 4,
    c_plateau: float = 4.0,
    scan_step: int = 10,
    scan_min_points: int = 40,
):
    """
    Scan n_therm = 0, scan_step, 2*scan_step, ...
    up to max_therm = N - scan_min_points.

    Returns JSON-friendly dict, does not plot.
    """
    y_full = np.asarray(y_full, dtype=float)
    y_full = y_full[np.isfinite(y_full)]
    N = y_full.size
    if N < scan_min_points:
        return {
            "skipped": True,
            "reason": "not_enough_points",
            "scan_step": int(scan_step),
            "scan_min_points": int(scan_min_points),
            "points": [],
        }

    max_therm = max(0, N - scan_min_points)

    points: list[dict[str, Any]] = []
    for n_therm in range(0, max_therm + 1, scan_step):
        y = y_full[n_therm:]
        if y.size < scan_min_points:
            continue
        tau_est, tau_err, *_ = compute_tau_from_series(y, min_nbs=min_nbs, c_plateau=c_plateau)
        if np.isfinite(tau_est) and np.isfinite(tau_err):
            points.append({"n_therm": int(n_therm), "tau_int": float(tau_est), "err": float(tau_err)})

    if len(points) < 2:
        return {
            "skipped": True,
            "reason": "not_enough_valid_scan_points",
            "scan_step": int(scan_step),
            "scan_min_points": int(scan_min_points),
            "points": [],
        }

    return {
        "skipped": False,
        "scan_step": int(scan_step),
        "scan_min_points": int(scan_min_points),
        "points": points,
    }


def plot_scan(
    scan_payload: dict[str, Any],
    plot_path: str,
):
    if scan_payload.get("skipped", True):
        return

    pts = scan_payload.get("points", [])
    if not pts:
        return

    therms = [p["n_therm"] for p in pts]
    taus = [p["tau_int"] for p in pts]
    errs = [p["err"] for p in pts]

    fig, ax = plt.subplots(figsize=(3.7, 2.6), layout="constrained")
    ax.errorbar(
        therms, taus, yerr=errs,
        fmt="o", markersize=3.0, markerfacecolor="none",
        capsize=2, elinewidth=1, linestyle="none"
    )
    ax.set_xlabel(r"$n_{\mathrm{therm}}$")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$")

    fmt = ScalarFormatter(useMathText=True)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    fig.savefig(plot_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def compute_tau_from_file(
    input_file: str,
    out_dir: str,
    therm: int,
    plot_styles: str | None = None,
    base_name: str = "tau_int",
):
    """
    Main reusable entry point.

    Reads series from input_file, applies therm cut, writes outputs into out_dir:
      - {base_name}_results.json          (contains binning table, estimate, scan)
      - {base_name}_vs_Nb.pdf
      - {base_name}_vs_n_therm.pdf

    Returns:
      tau_est, tau_err, Nb_est, Nbs_est, found
    """
    apply_plot_styles(plot_styles)

    y_full = read_series_file(input_file)
    if y_full.size == 0:
        raise ValueError(f"No finite data read from {input_file}")

    therm = int(max(0, therm))
    y = y_full[therm:]

    os.makedirs(out_dir, exist_ok=True)

    # Compute main estimate + full "table" series
    tau_est, tau_err, Nb_est, Nbs_est, found, used_index, (Nb_list, Nbs_list, tauR_list, errR_list) = \
        compute_tau_from_series(y, min_nbs=4, c_plateau=4.0)

    # Paths
    plot_nb_path = os.path.join(out_dir, f"{base_name}_vs_Nb.pdf")
    plot_scan_path = os.path.join(out_dir, f"{base_name}_vs_n_therm.pdf")
    json_path = os.path.join(out_dir, f"{base_name}_results.json")

    # Build binning "table" as JSON array (this replaces the old TXT file)
    binning_table: list[dict[str, Any]] = []
    for i, (Nb, Nbs, tau, er) in enumerate(zip(Nb_list, Nbs_list, tauR_list, errR_list)):
        binning_table.append({
            "Nb": int(Nb),
            "Nbs": int(Nbs),
            "tau_int": float(tau),
            "err": float(er),
            "used": bool(used_index is not None and i == used_index),
        })

    if used_index is None:
        used_reason = None
    else:
        used_reason = "first_nb_exceeds_c_tau_berg" if found else "fallback_last_point"

    # Scan
    scan_payload = scan_tau_vs_n_therm(
        y_full,
        min_nbs=4,
        c_plateau=4.0,
        scan_step=10,
        scan_min_points=40,
    )

    # Plots (only if main estimate exists)
    if np.isfinite(tau_est) and (Nb_est is not None) and Nb_list:
        plot_tau_vs_nb(Nb_list, tauR_list, errR_list, Nb_est, tau_est, plot_nb_path)

    # Always attempt scan plot (matches original behavior)
    plot_scan(scan_payload, plot_scan_path if not scan_payload.get("skipped", True) else plot_scan_path)

    # Final JSON payload (single source of truth)
    results: dict[str, Any] = {
        "ok": bool(np.isfinite(tau_est) and np.isfinite(tau_err)),
        "input": {
            "input_file": input_file,
            "out_dir": out_dir,
            "therm": int(therm),
            "N_full": int(y_full.size),
            "N_used": int(y.size),
        },
        "method": {
            "name": "berg_binning",
            "c_plateau": 4.0,
            "min_nbs": 4,
            "reported_factor": 0.5,  # reported = Berg/2
        },
        "estimate": {
            "tau_int": None if not np.isfinite(tau_est) else float(tau_est),
            "err": None if not np.isfinite(tau_err) else float(tau_err),
            "Nb": None if Nb_est is None else int(Nb_est),
            "Nbs": None if Nbs_est is None else int(Nbs_est),
            "found_plateau": bool(found),
            "used_index": None if used_index is None else int(used_index),
            "used_reason": used_reason,
        },
        "binning_table": binning_table,
        "n_therm_scan": scan_payload,
        "plots": {
            "tau_vs_nb_pdf": plot_nb_path,
            "tau_vs_n_therm_pdf": (None if scan_payload.get("skipped", True) else plot_scan_path),
        },
        "outputs": {
            "results_json": json_path,
        },
    }

    # Write ONLY the JSON file for "text output"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    # Also print JSON to stdout (handy for pipelines)
    print(json.dumps(results, indent=2, sort_keys=True))

    # Keep return signature
    return tau_est, tau_err, Nb_est, Nbs_est, found


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _main_cli():
    ap = argparse.ArgumentParser(description="Generic tau_int (Berg binning; reported = Berg/2). JSON-only text output.")
    ap.add_argument("input_file")
    ap.add_argument("out_dir")
    ap.add_argument("therm", type=int)
    ap.add_argument("--plot_styles", default=None)
    ap.add_argument("--base_name", default="tau_int")
    args = ap.parse_args()

    compute_tau_from_file(
        input_file=args.input_file,
        out_dir=args.out_dir,
        therm=args.therm,
        plot_styles=args.plot_styles,
        base_name=args.base_name,
    )


if __name__ == "__main__":
    _main_cli()