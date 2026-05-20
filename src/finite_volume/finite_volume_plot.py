#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit

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
            "  results -> standard_fit -> PP -> am_ps -> sdev"
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

    dx = 0.04 * np.cos(np.pi * pos) * width
    dy = 0.06 * np.sin(np.pi / 2 * pos) * height

    ax.text(x + dx, y + dy, text, fontsize=7)


# ============================================================
# Fit helpers
# ============================================================
def weighted_linear_fit(x, y, sigma):
    """
    Fit y = a + b x using weights 1/sigma^2.
    Returns coeffs and covariance.
    """
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(1.0 / sigma**2)
    XtW = X.T @ W
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ XtW @ y
    return beta, cov


def fv_ansatz(Ns, m_inf, A):
    """
    Actual finite-volume ansatz:
      m(L) = m_inf + A * m_inf / (m_inf L)^(3/2) * exp(-m_inf L)
    with L identified here as Ns.
    """
    x = m_inf * Ns
    return m_inf + A * m_inf * np.exp(-x) / (x ** 1.5)


def guess_initial_parameters(Ns, Y):
    """
    Build reasonable starting values for the nonlinear fit.
    """
    m_inf0 = float(np.min(Y))
    if not np.isfinite(m_inf0) or m_inf0 <= 0:
        m_inf0 = float(Y[-1])

    delta = Y - m_inf0
    positive = delta > 0

    if np.count_nonzero(positive) > 0:
        i = np.where(positive)[0][0]
        x0 = m_inf0 * Ns[i]
        A0 = delta[i] * (x0 ** 1.5) / m_inf0 * np.exp(x0)
        if not np.isfinite(A0) or A0 <= 0:
            A0 = 1.0
    else:
        A0 = 1.0

    return m_inf0, A0


def fit_actual_fv_model(Ns, Y, Yerr):
    """
    Nonlinear fit of the actual FV ansatz to the measured masses.
    """
    m_inf0, A0 = guess_initial_parameters(Ns, Y)

    popt, pcov = curve_fit(
        fv_ansatz,
        Ns,
        Y,
        p0=[m_inf0, A0],
        sigma=Yerr,
        absolute_sigma=True,
        maxfev=20000,
        bounds=([1e-12, 0.0], [np.inf, np.inf]),
    )

    m_inf_fit, A_fit = popt
    m_inf_err, A_err = np.sqrt(np.diag(pcov))
    return popt, pcov, m_inf_fit, m_inf_err, A_fit, A_err


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Finite-volume plot from spectrum.json files. "
            "Selection is made using metadata rows identified by "
            "(use flag, beta, mass, Ns). "
            "The plot is the linearized form of the asymptotic FV ansatz."
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

    # ------------------------------------------------------------
    # 1) Fit the actual finite-volume ansatz to determine m_inf
    # ------------------------------------------------------------
    popt_nl, pcov_nl, m_ps_inf, m_ps_inf_err, A_nl, A_nl_err = fit_actual_fv_model(Ns, Y, Yerr)

    # x = m_inf * L
    X_plot = Ns * m_ps_inf

    # delta = m(L) - m_inf
    delta = Y - m_ps_inf

    # Linearized observable:
    # log( delta * (m_inf L)^(3/2) / m_inf ) = log(A) - m_inf L
    # Keep only points with positive delta so the log is defined.
    linear_mask = delta > 0.0

    if np.count_nonzero(linear_mask) < 2:
        raise ValueError(
            "Need at least two points with m_ps(L) > fitted m_ps_inf to make the logarithmic linearized plot."
        )

    X_lin = X_plot[linear_mask]
    Y_lin_raw = Y[linear_mask]
    Yerr_lin_raw = Yerr[linear_mask]
    Ns_lin = Ns[linear_mask]
    delta_lin = delta[linear_mask]

    Y_lin = np.log(delta_lin * X_lin**1.5 / m_ps_inf)

    # Error propagation with m_inf treated as fixed in the linearized step
    Y_lin_err = Yerr_lin_raw / delta_lin

    # Weighted linear fit: y = a + b x
    coeffs, cov = weighted_linear_fit(X_lin, Y_lin, Y_lin_err)
    a_fit, b_fit = coeffs
    a_err = np.sqrt(cov[0, 0])
    b_err = np.sqrt(cov[1, 1])

    fit_result = {
        "nonlinear_actual_ansatz": {
            "m_ps_inf_mean": float(m_ps_inf),
            "m_ps_inf_sdev": float(m_ps_inf_err),
            "A_mean": float(A_nl),
            "A_sdev": float(A_nl_err),
            "covariance": pcov_nl.tolist(),
        },
        "linearized_fit": {
            "intercept_mean": float(a_fit),
            "intercept_sdev": float(a_err),
            "slope_mean": float(b_fit),
            "slope_sdev": float(b_err),
            "A_from_intercept_mean": float(np.exp(a_fit)),
            "A_from_intercept_sdev": float(np.exp(a_fit) * a_err),
        },
    }

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
        "m_ps_inf_err": float(m_ps_inf_err),
        "Ns": [float(v) for v in Ns],
        "m_ps": [float(v) for v in Y],
        "m_ps_err": [float(v) for v in Yerr],
        "m_ps_inf_L": [float(v) for v in X_plot],
        "delta_m_ps": [float(v) for v in delta],
        "linearized_points": {
            "Ns": [float(v) for v in Ns_lin],
            "x": [float(v) for v in X_lin],
            "y": [float(v) for v in Y_lin],
            "y_err": [float(v) for v in Y_lin_err],
        },
        "input_files": selected_files,
        "fit": fit_result,
    }

    out_dir = os.path.dirname(args.fv_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.fv_out, "w") as f:
        json.dump(out_data, f, indent=2)

    fig, ax = plt.subplots(figsize=(4.8, 2.8), layout="constrained")

    data_label = (
        rf"$\beta={args.beta},\, am_0={args.mass}$"
        "\n"
        rf"$m_{{\rm PS}}^\infty={m_ps_inf:.4g}\pm{m_ps_inf_err:.2g}$"
    )

    fit_label = (
        rf"linear fit: $a=({a_fit:.3g}\pm{a_err:.2g})$, "
        rf"$b=({b_fit:.3g}\pm{b_err:.2g})$"
    )

    ax.errorbar(
        X_lin,
        Y_lin,
        yerr=Y_lin_err,
        ls="none",
        marker="o",
        label=data_label,
    )

    xx = np.linspace(np.min(X_lin), np.max(X_lin), 200)
    yy = a_fit + b_fit * xx
    ax.plot(xx, yy, "k--", label=fit_label)

    if len(X_lin) == 1:
        place_label(ax, X_lin[0], Y_lin[0], rf"$N_s={int(Ns_lin[0])}$", 0.5)
    else:
        for i in range(len(X_lin)):
            pos = i / (len(X_lin) - 1)
            place_label(ax, X_lin[i], Y_lin[i], rf"$N_s={int(Ns_lin[i])}$", pos)

    ax.set_xlabel(r"$m_{\rm PS}^{\inf} N_s$")
    ax.set_ylabel(
        r"$\log\!\left[(m_{\rm PS}-m_{\rm PS}^{\inf})(m_{\rm PS}^{\inf}N_s)^{3/2}/m_{\rm PS}^{\inf}\right]$"
    )

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.legend()

    plot_dir = os.path.dirname(args.fv_plot)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(args.fv_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nNonlinear fit to actual FV ansatz:")
    print(f"  m_ps_inf = {m_ps_inf:.12g} ± {m_ps_inf_err:.12g}")
    print(f"  A        = {A_nl:.12g} ± {A_nl_err:.12g}")

    print("\nLinearized fit results:")
    print(f"  intercept = {a_fit:.12g} ± {a_err:.12g}")
    print(f"  slope     = {b_fit:.12g} ± {b_err:.12g}")
    print(f"  A         = {np.exp(a_fit):.12g} ± {np.exp(a_fit)*a_err:.12g}")

    print(f"\n✓ Saved JSON  → {args.fv_out}")
    print(f"✓ Saved plot  → {args.fv_plot}")


if __name__ == "__main__":
    main()