#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter

plt.style.use("tableau-colorblind10")


# ============================================================
# Generic helpers
# ============================================================
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def safe_get(dct, *keys, default=np.nan):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def merge_prefer_finite(primary, fallback):
    out = dict(primary)
    for k, v in fallback.items():
        if not np.isfinite(to_float(out.get(k, np.nan))):
            out[k] = v
    return out


def normalize_flag_series(series):
    s = series.astype("string")
    return s.str.strip().str.lower().isin(["true", "1", "yes", "y"])


# ============================================================
# Path parsing helpers
# ============================================================
def extract_from_path(path):
    vals = {
        "beta": np.nan,
        "mass": np.nan,
        "Ns": np.nan,
    }

    parts = Path(path).parts

    for p in parts:
        if p.startswith("B"):
            vals["beta"] = to_float(p[1:], vals["beta"])
        elif p.startswith("Ns"):
            vals["Ns"] = to_float(p[2:], vals["Ns"])
        elif p.startswith("M5"):
            continue
        elif p.startswith("M") and not p.startswith("M5") and not np.isfinite(vals["mass"]):
            vals["mass"] = to_float(p[1:], vals["mass"])

    return vals


def read_common_parameters(data, path):
    params = safe_get(data, "parameters", default={})

    from_json = {
        "beta": to_float(safe_get(params, "beta", default=np.nan)),
        "mass": to_float(safe_get(params, "mass", default=np.nan)),
        "Ns": to_float(safe_get(params, "Ns", default=np.nan)),
    }

    from_path = extract_from_path(path)
    return merge_prefer_finite(from_json, from_path)


# ============================================================
# Readers
# ============================================================
def read_spectrum_json(path):
    data = read_json(path)
    rec = read_common_parameters(data, path)

    rec["mps"] = to_float(
        safe_get(data, "results", "standard_fit", "PP", "am_ps", "mean", default=np.nan)
    )
    rec["mps_err"] = to_float(
        safe_get(data, "results", "standard_fit", "PP", "am_ps", "sdev", default=np.nan)
    )

    if not np.isfinite(rec["mps"]) or not np.isfinite(rec["mps_err"]):
        raise ValueError(
            f"Missing or invalid PS mass data in spectrum JSON '{path}'.\n"
            "Expected numeric values at:\n"
            "  results -> standard_fit -> PP -> am_ps -> mean\n"
            "  results -> bootstrap_fit -> PP -> am_ps -> sdev"
        )

    if not np.isfinite(rec["beta"]) or not np.isfinite(rec["mass"]) or not np.isfinite(rec["Ns"]):
        raise ValueError(
            f"Could not determine beta/mass/Ns for '{path}' from JSON parameters or path."
        )

    rec["_source_file_mps"] = str(path)
    return rec


# ============================================================
# Metadata
# ============================================================
def read_metadata(metadata_csv, use_name):
    meta = pd.read_csv(metadata_csv, sep=r"\t|,", engine="python")

    flagcol = f"use_in_{use_name}"
    if flagcol not in meta.columns:
        raise ValueError(f"Column '{flagcol}' not found in {metadata_csv}")

    meta = meta[normalize_flag_series(meta[flagcol])].copy()
    if meta.empty:
        raise ValueError(f"No rows selected by column '{flagcol}'")

    for c in ["beta", "mass", "Ns"]:
        if c not in meta.columns:
            raise ValueError(f"Column '{c}' not found in {metadata_csv}")
        meta[c] = pd.to_numeric(meta[c], errors="coerce")

    meta = meta.dropna(subset=["beta", "mass", "Ns"]).copy()
    return meta


def choose_allowed_rows(meta, beta, mass):
    mask = np.isclose(meta["beta"], beta) & np.isclose(meta["mass"], mass)
    selected = meta[mask].copy()

    if selected.empty:
        raise ValueError(
            f"No metadata rows selected for finite-volume points with beta={beta}, mass={mass}"
        )

    return selected


def record_allowed(rec, allowed_rows):
    rb = to_float(rec.get("beta", np.nan))
    rm = to_float(rec.get("mass", np.nan))
    rns = to_float(rec.get("Ns", np.nan))

    if not (np.isfinite(rb) and np.isfinite(rm) and np.isfinite(rns)):
        return False

    mask = (
        np.isclose(allowed_rows["beta"], rb)
        & np.isclose(allowed_rows["mass"], rm)
        & np.isclose(allowed_rows["Ns"], rns)
    )
    return bool(mask.any())


# ============================================================
# Label placement
# ============================================================
def place_label(ax, x, y, text, pos):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    dx = 0.08 * np.cos(np.pi * pos) * width
    dy = 0.08 * np.sin(np.pi / 2 * pos) * height

    ax.text(x + dx, y + dy, text, fontsize=7)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Finite-volume plot from spectrum.json files. "
            "Selection is made using metadata rows identified by "
            "(use flag, beta, mass, Ns)."
        )
    )
    parser.add_argument("--plot_styles", default="")
    parser.add_argument("--m_ps", nargs="+", required=True, help="Input spectrum.json files")
    parser.add_argument("--metadata_csv", required=True, help="Path to ensembles.csv")
    parser.add_argument(
        "--use",
        default="finite_volume",
        help="Selection name; metadata rows are filtered using use_in_<use>.",
    )
    parser.add_argument("--fv_out", required=True, help="Output JSON summary")
    parser.add_argument("--fv_plot", required=True, help="Output plot file")
    parser.add_argument("--beta", required=True, type=float)
    parser.add_argument("--mass", required=True, type=float)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    meta = read_metadata(args.metadata_csv, args.use)
    allowed_rows = choose_allowed_rows(meta, args.beta, args.mass)

    selected = []
    skipped = []

    for spec_path in args.m_ps:
        try:
            rec = read_spectrum_json(spec_path)
        except Exception as e:
            print(f"WARNING: could not read spectrum JSON {spec_path}: {e}")
            continue

        if record_allowed(rec, allowed_rows):
            selected.append(rec)
        else:
            skipped.append(rec["_source_file_mps"])

    if not selected:
        raise ValueError(
            "No input files matched the metadata-selected finite-volume points "
            f"for beta={args.beta}, mass={args.mass}."
        )

    selected.sort(key=lambda r: r["Ns"])

    seen_ns = {}
    for rec in selected:
        ns_key = round(float(rec["Ns"]), 12)
        src = rec["_source_file_mps"]
        if ns_key in seen_ns:
            raise ValueError(
                f"More than one file matched the selected metadata rows for Ns={rec['Ns']}.\n"
                f"  First:  {seen_ns[ns_key]}\n"
                f"  Second: {src}"
            )
        seen_ns[ns_key] = src

    print("\nSelected finite-volume points:")
    for rec in selected:
        print(
            f"  Ns={rec['Ns']:>5g}  "
            f"mps={rec['mps']:.12g}  "
            f"err={rec['mps_err']:.12g}  "
            f"file={rec['_source_file_mps']}"
        )

    if skipped:
        print("\nSkipped candidate files (not selected by metadata):")
        for path in skipped:
            print(f"  {path}")

    Ns = np.array([float(r["Ns"]) for r in selected], dtype=float)
    Y = np.array([float(r["mps"]) for r in selected], dtype=float)
    Yerr = np.array([float(r["mps_err"]) for r in selected], dtype=float)
    selected_files = [r["_source_file_mps"] for r in selected]

    # take the largest-Ns point as infinite-volume proxy
    m_ps_inf = Y[-1]
    X_plot = Ns * m_ps_inf

    fit_result = None

    def fit_func(L, A):
        return m_ps_inf * (
            1.0 + A * np.exp(-m_ps_inf * L) / (m_ps_inf * L) ** 1.5
        )

    if len(Ns) >= 2:
        try:
            popt, pcov = curve_fit(
                fit_func,
                Ns,
                Y,
                sigma=Yerr,
                absolute_sigma=True,
                maxfev=10000,
            )
            fit_result = {
                "A_mean": float(popt[0]),
                "A_sdev": float(np.sqrt(pcov[0, 0])) if pcov.size else None,
            }
        except Exception as e:
            print(f"WARNING: fit failed: {e}")
            popt = None
    else:
        popt = None
        print("WARNING: fewer than 2 points; skipping fit.")

    out_data = {
        "label": args.label,
        "use": args.use,
        "metadata_csv": args.metadata_csv,
        "beta": float(args.beta),
        "mass": float(args.mass),
        "selected_metadata_rows": [
            {
                "beta": float(row.beta),
                "mass": float(row.mass),
                "Ns": float(row.Ns),
            }
            for row in allowed_rows.itertuples(index=False)
        ],
        "m_ps_inf": float(m_ps_inf),
        "Ns": [float(v) for v in Ns],
        "m_ps": [float(v) for v in Y],
        "m_ps_err": [float(v) for v in Yerr],
        "m_ps_inf_L": [float(v) for v in X_plot],
        "input_files": selected_files,
        "fit": fit_result,
    }

    out_dir = os.path.dirname(args.fv_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.fv_out, "w") as f:
        json.dump(out_data, f, indent=2)

    fig, ax = plt.subplots(figsize=(4.5, 2.5), layout="constrained")

    legend_label = rf"$\beta={args.beta},\, am_0={args.mass}$"

    ax.errorbar(
        X_plot,
        Y,
        yerr=Yerr,
        ls="none",
        marker="o",
        label=legend_label,
    )

    if popt is not None:
        xx = np.linspace(np.min(X_plot), np.max(X_plot), 200)
        ns_fit = xx / m_ps_inf
        ax.plot(xx, fit_func(ns_fit, *popt), "k--")

    ax.set_ylabel(r"$am_{\rm PS}$")
    ax.set_xlabel(r"$m_{\rm PS}^{\rm inf} N_s$")

    ax.axhline(
        y=m_ps_inf,
        color="gray",
        linestyle="-",
        label=r"$am_{\rm PS}^{\rm inf}$",
    )

    if len(X_plot) == 1:
        place_label(ax, X_plot[0], Y[0], rf"$N_s={int(Ns[0])}$", 0.5)
    else:
        for i in range(len(X_plot)):
            pos = i / (len(X_plot) - 1)
            place_label(ax, X_plot[i], Y[i], rf"$N_s={int(Ns[i])}$", pos)

    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.legend()

    plot_dir = os.path.dirname(args.fv_plot)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(args.fv_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Saved JSON  → {args.fv_out}")
    print(f"✓ Saved plot  → {args.fv_plot}")


if __name__ == "__main__":
    main()