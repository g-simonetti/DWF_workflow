#!/usr/bin/env python3
"""
Extract HMC/MD statistics from Grid log-*.zst files and write ONE JSON file containing:
  - hmc_extract: statistics (including tau_int_plaq + errors)
  - plaq_history: full plaquette history (forward-filled), including pre-therm
  - metadata/keys: parsed parameters + ensemble CSV lookup

Also runs autocorr_time/tau_int.py (via compute_tau_from_file) on the plaquette history,
using therm from ensembles.csv as the main estimate.

This replaces the previous two-file outputs (log_hmc_extract.txt + plaq_history.txt).
"""

import argparse
import glob
import io
import json
import os
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import zstandard as zstd
import matplotlib.pyplot as plt

# Ensure src/ is importable so we can import autocorr_time.tau_int
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from autocorr_time.tau_int import compute_tau_from_file  # noqa: E402


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
# Utility
# -----------------------------------------------------------------------------
def normpath_posix(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def read_zst_lines(path: str):
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                yield line.rstrip("\n")


def bootstrap_mean_err(x, n_boot: int = 1000, rng=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(means.std(ddof=1))


def slice_therm_delta(x, therm: int, delta: int, n_conf: int):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or n_conf <= 0:
        return x[:0]
    start = min(int(therm), x.size)
    idx = start + int(delta) * np.arange(int(n_conf), dtype=int)
    idx = idx[idx < x.size]
    return x[idx]


# -----------------------------------------------------------------------------
# Parse log_dir into metadata keys
# -----------------------------------------------------------------------------
def parse_keys_from_log_dir(log_dir: str) -> dict:
    p = normpath_posix(log_dir)

    m = re.search(
        r"(?:^|/)raw_data/NF(?P<NF>\d+)/Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/(?P<subdir>.+)/log$",
        p,
    )
    if not m:
        raise ValueError(
            f"Could not parse log_dir:\n  {log_dir}\n"
            "Expected: .../raw_data/NF*/Nt*/Ns*/<subdir>/log"
        )

    NF = int(m.group("NF"))
    Nt = int(m.group("Nt"))
    Ns = int(m.group("Ns"))
    subdir = m.group("subdir")

    if NF == 0:
        m0 = re.match(r"^B(?P<beta>[^/]+)$", subdir)
        if not m0:
            raise ValueError(
                f"NF=0 but subdir does not look like 'B{{beta}}'.\n"
                f"subdir={subdir}"
            )
        return {
            "NF": 0,
            "Nt": Nt,
            "Ns": Ns,
            "beta": m0.group("beta"),
        }

    m1 = re.match(
        r"^Ls(?P<Ls>\d+)/"
        r"B(?P<beta>[^/]+)/"
        r"M(?P<mass>[^/]+)/"
        r"mpv(?P<mpv>[^/]+)/"
        r"alpha(?P<alpha>[^/]+)/"
        r"a5(?P<a5>[^/]+)/"
        r"M5(?P<M5>[^/]+)"
        r"(?:/.*)?$",
        subdir,
    )
    if not m1:
        raise ValueError(
            f"NF>0 but subdir does not start with expected dynamical pattern.\n"
            f"subdir={subdir}\n"
            "Expected start like:\n"
            "  Ls8/B7.4/M0.1/mpv1.0/alpha1.75/a51.0/M51.8\n"
            "Optionally followed by /<run>/..."
        )

    d = m1.groupdict()
    return {
        "NF": NF,
        "Nt": Nt,
        "Ns": Ns,
        "Ls": int(d["Ls"]),
        "beta": d["beta"],
        "mass": d["mass"],
        "mpv": d["mpv"],
        "alpha": d["alpha"],
        "a5": d["a5"],
        "M5": d["M5"],
    }


def lookup_metadata_from_csv(ensembles_csv: str, keys: dict) -> tuple[str, int, int]:
    df = pd.read_csv(ensembles_csv, sep=r"\t|,", engine="python")

    core_required = ("name", "therm", "delta_traj", "NF", "Nt", "Ns", "beta")
    for col in core_required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {ensembles_csv}")

    for col in ("NF", "Nt", "Ns", "beta"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def close(series, x):
        return np.isclose(series.to_numpy(dtype=float), float(x), rtol=0.0, atol=1e-12)

    NF = int(keys["NF"])

    if NF == 0:
        sel = (
            close(df["NF"], 0)
            & close(df["Nt"], keys["Nt"])
            & close(df["Ns"], keys["Ns"])
            & close(df["beta"], keys["beta"])
        )
        dfq = df[sel]
    else:
        dyn_required = ("Ls", "mass", "mpv", "alpha", "a5", "M5")
        for col in dyn_required:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in {ensembles_csv} "
                    f"(required for dynamical NF>0 ensembles)"
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        sel = (
            close(df["NF"], keys["NF"])
            & close(df["Nt"], keys["Nt"])
            & close(df["Ns"], keys["Ns"])
            & close(df["Ls"], keys["Ls"])
            & close(df["beta"], keys["beta"])
            & close(df["mass"], keys["mass"])
            & close(df["mpv"], keys["mpv"])
            & close(df["alpha"], keys["alpha"])
            & close(df["a5"], keys["a5"])
            & close(df["M5"], keys["M5"])
        )
        dfq = df[sel]

    if len(dfq) != 1:
        cols_show = [
            c
            for c in (
                "NF",
                "Nt",
                "Ns",
                "Ls",
                "beta",
                "mass",
                "mpv",
                "alpha",
                "a5",
                "M5",
                "name",
                "therm",
                "delta_traj",
            )
            if c in df.columns
        ]
        preview = dfq[cols_show].head(20).to_string(index=False) if len(dfq) else "(no rows)"
        raise ValueError(
            f"Metadata lookup expected 1 row, got {len(dfq)}.\n"
            f"Parsed keys: {keys}\n"
            f"Matching rows preview:\n{preview}\n"
        )

    row = dfq.iloc[0]
    return str(row["name"]), int(row["therm"]), int(row["delta_traj"])


# -----------------------------------------------------------------------------
# Log parsing
# -----------------------------------------------------------------------------
def extract_from_logs(log_dir: str):
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


def build_full_series_for_plaquette(data: dict) -> tuple[np.ndarray, np.ndarray]:
    traj_numbers = data["traj_numbers"]
    if traj_numbers.size > 0:
        t_min = int(traj_numbers.min())
        t_max = int(traj_numbers.max())
        n_traj_total = t_max - t_min + 1
    else:
        lengths = [data["fullbcs_raw"].size, data["traj_times"].size]
        n_traj_total = int(max(lengths)) if lengths else 0
        t_min = 0

    plaq_pairs = data["plaq_pairs"]
    if plaq_pairs.size == 0 or n_traj_total <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    traj_idx = plaq_pairs[:, 0].astype(int)
    plaq_vals = plaq_pairs[:, 1].astype(float)

    full_series = np.full(n_traj_total, np.nan, dtype=float)
    for ti, pv in zip(traj_idx, plaq_vals):
        j = ti - t_min
        if 0 <= j < n_traj_total:
            full_series[j] = pv

    # Forward fill
    for i in range(1, n_traj_total):
        if np.isnan(full_series[i]):
            full_series[i] = full_series[i - 1]

    # Fill early segment
    if n_traj_total > 0 and np.isnan(full_series[0]):
        first_valid = np.where(~np.isnan(full_series))[0]
        if first_valid.size > 0:
            full_series[: first_valid[0]] = full_series[first_valid[0]]

    mc_times = np.arange(t_min, t_min + n_traj_total, dtype=int)
    return mc_times, full_series


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract HMC statistics from log-*.zst files into ONE JSON.")
    parser.add_argument("log_dir")
    parser.add_argument("--ensembles_csv", required=True, help="Path to metadata/ensembles.csv")
    parser.add_argument("--plot_styles", default=None, help="Matplotlib style(s): comma-separated ok")
    parser.add_argument("--hmc_out", required=True, help="Output JSON file")
    args = parser.parse_args()

    apply_plot_styles(args.plot_styles)

    keys = parse_keys_from_log_dir(args.log_dir)
    name, therm, delta_traj = lookup_metadata_from_csv(args.ensembles_csv, keys)
    print(f"[meta] name={name} therm={therm} delta_traj={delta_traj}")

    data = extract_from_logs(args.log_dir)

    # FULL plaquette series (includes pre-therm), forward-filled
    mc_times, plaq_full = build_full_series_for_plaquette(data)

    out_dir = os.path.dirname(os.path.abspath(args.hmc_out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Stats (excluding tau_int until we compute it)
    n_traj_total = int(plaq_full.size)
    usable = max(0, n_traj_total - therm)
    n_conf = usable // int(delta_traj) if int(delta_traj) > 0 else 0

    fullbcs_incr_s = slice_therm_delta(data["fullbcs_incr"], therm, delta_traj, n_conf)
    traj_times_s = slice_therm_delta(data["traj_times"], therm, delta_traj, n_conf)
    plaq_s = slice_therm_delta(plaq_full, therm, delta_traj, n_conf)

    fullbcs_mean, fullbcs_err = bootstrap_mean_err(fullbcs_incr_s)
    bcs_mean, bcs_err = bootstrap_mean_err(fullbcs_incr_s)
    ttraj_mean, ttraj_err = bootstrap_mean_err(traj_times_s)
    plaq_mean, plaq_err = bootstrap_mean_err(plaq_s)

    length_traj = data["traj_length"] if data["traj_length"] is not None else np.nan
    n_steps = data["md_steps"] if data["md_steps"] is not None else np.nan
    accept_ratio = float(data["accept_ratio"]) if np.isfinite(data["accept_ratio"]) else np.nan

    # ---- Compute tau_int on plaquette full series via a TEMP series file ----
    # compute_tau_from_file expects a text series file; we write one temp file here.
    tmp_plaq_path = os.path.join(out_dir, "plaq_history_tmp_for_tau_int.txt")
    with open(tmp_plaq_path, "w") as fpl:
        for t, pv in zip(mc_times, plaq_full):
            if np.isfinite(pv):
                fpl.write(f"{int(t)} {float(pv):.16e}\n")

    tau_int_plaq, tau_int_plaq_err, Nb_est, Nbs_est, found = compute_tau_from_file(
        input_file=tmp_plaq_path,
        out_dir=out_dir,
        therm=therm,
        plot_styles=args.plot_styles,
        base_name="tau_int",
    )

    # Remove temp file (comment this out if you want to keep it for debugging)
    try:
        os.remove(tmp_plaq_path)
    except OSError:
        pass

    payload: dict[str, Any] = {
        "keys_from_path": keys,
        "ensemble": {
            "name": name,
            "therm": int(therm),
            "delta_traj": int(delta_traj),
        },
        "hmc_extract": {
            "accept": int(data["accept"]),
            "reject": int(data["reject"]),
            "accept_ratio": float(accept_ratio),
            "n_traj_total": int(n_traj_total),
            "n_conf": int(n_conf),
            "fullbcs": float(fullbcs_mean),
            "fullbcs_err": float(fullbcs_err),
            "bcs": float(bcs_mean),
            "bcs_err": float(bcs_err),
            "t_traj": float(ttraj_mean),
            "t_traj_err": float(ttraj_err),
            "plaq": float(plaq_mean),
            "plaq_err": float(plaq_err),
            "tau_int_plaq": float(tau_int_plaq),
            "tau_int_plaq_err": float(tau_int_plaq_err),
            "Nb_est": int(Nb_est),
            "Nbs_est": int(Nbs_est),
            "found_window": bool(found),
            "length_traj": float(length_traj),
            "n_steps": int(n_steps) if np.isfinite(n_steps) else None,
        },
        "plaq_history": {
            "t": [int(x) for x in mc_times.tolist()],
            "plaq": [float(x) for x in plaq_full.tolist()],
            "forward_filled": True,
            "includes_pre_therm": True,
        },
        # optional: helps debugging/repro
        "tau_int_outputs_dir": out_dir,
    }

    with open(args.hmc_out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"[log_ensembles_extract] wrote JSON → {args.hmc_out}")


if __name__ == "__main__":
    main()