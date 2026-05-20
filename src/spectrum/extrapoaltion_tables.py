#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        value = float(x)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def format_phys_err(value, error, force_decimals=None):
    value = to_float(value)
    error = abs(to_float(error))

    if not np.isfinite(value):
        return "—"
    if not np.isfinite(error) or error == 0:
        if force_decimals is not None:
            return f"{value:.{force_decimals}f}"
        return f"{value:g}"

    exp = int(np.floor(np.log10(error)))
    norm = error / 10**exp
    sig = 2 if norm < 3 else 1
    decimals = max(0, -exp + sig - 1)

    if force_decimals is not None:
        decimals = force_decimals

    value_r = round(value, decimals)
    error_r = round(error, decimals)

    value_str = f"{value_r:.{decimals}f}"
    if error_r < 1:
        err_digits = int(round(error_r * 10**decimals))
        return f"{value_str}({err_digits})"

    error_str = f"{error_r:.{decimals}f}"
    return f"{value_str}({error_str})"


def format_chi2_over_dof(chi2, dof):
    chi2 = to_float(chi2)
    dof = to_float(dof)
    if not np.isfinite(chi2) or not np.isfinite(dof) or dof <= 0:
        return "—"
    return f"{chi2 / dof:.2f}"


def build_longtable_scaffold(f, header_line, longtable_spec):
    f.write("%%%\\color{red}\n")
    f.write(f"%%%\\begin{{longtable}}{{{longtable_spec}}}\n")
    f.write("%%%\\caption\n")
    f.write("%%%\\label \\\\\n\n")

    f.write("% ================= FIRST PAGE HEADER =================\n")
    f.write(header_line)
    f.write("\\hline\n")
    f.write("\\endfirsthead\n\n")

    f.write("% ================ HEADER FOR PAGE 2+ =================\n")
    f.write("\\hline\n")
    f.write(header_line)
    f.write("\\hline\n")
    f.write("\\endhead\n\n")

    f.write("% ================= FOOTER FOR INTERMEDIATE PAGES =================\n")
    f.write("\\hline\n")
    f.write("\\endfoot\n\n")

    f.write("% ================= FINAL FOOTER =================\n")
    f.write("\\hline\\hline\n")
    f.write("\\endlastfoot\n\n")

    f.write("% ===================== TABLE BODY =====================\n")


def extract_fit_row(label, discretization, fit, include_linear_a_terms):
    row = {
        "label": label,
        "discretization": discretization,
        "m_M_chi_sq": format_phys_err(
            fit.get("m_M_chi_sq"), fit.get("m_M_chi_sq_err"), force_decimals=3
        ),
        "L_m_M": format_phys_err(fit.get("L_m_M"), fit.get("L_m_M_err"), force_decimals=2),
        "Q_m_M": format_phys_err(fit.get("Q_m_M"), fit.get("Q_m_M_err"), force_decimals=2),
        "W_m_M": "—",
        "R_m_M": format_phys_err(fit.get("R_m_M"), fit.get("R_m_M_err"), force_decimals=2),
        "C_m_M": "—",
        "chi2_over_dof": format_chi2_over_dof(fit.get("chi2"), fit.get("dof")),
    }

    if include_linear_a_terms:
        row["W_m_M"] = format_phys_err(
            fit.get("W_m_M"), fit.get("W_m_M_err"), force_decimals=2
        )
        row["C_m_M"] = format_phys_err(
            fit.get("C_m_M"), fit.get("C_m_M_err"), force_decimals=2
        )

    return row


def collect_rows(mv_fit_data, mv_fit_bootstrap_data):
    rows = []

    dw_bootstrap = mv_fit_bootstrap_data.get("fits", {}).get("dw2", {})
    dw_central = dw_bootstrap.get("central_nonlinear")
    if isinstance(dw_central, dict):
        rows.append(
            extract_fit_row(
                label="Central fit",
                discretization="MDWF",
                fit=dw_central,
                include_linear_a_terms=False,
            )
        )

    dw_summary = dw_bootstrap.get("bootstrap_summary")
    if isinstance(dw_summary, dict):
        rows.append(
            extract_fit_row(
                label="Bootstrap fit",
                discretization="MDWF",
                fit=dw_summary,
                include_linear_a_terms=False,
            )
        )

    wilson_fits_central = mv_fit_data.get("fits", {}).get("wilson_physical", {})
    wilson_central = wilson_fits_central.get("central_nonlinear")
    if not isinstance(wilson_central, dict):
        wilson_central = wilson_fits_central.get("nonlinear")
    if isinstance(wilson_central, dict):
        rows.append(
            extract_fit_row(
                label="Central fit",
                discretization="Wilson",
                fit=wilson_central,
                include_linear_a_terms=True,
            )
        )

    wilson_fits_bootstrap = mv_fit_bootstrap_data.get("fits", {}).get("wilson_physical", {})
    wilson_bootstrap = wilson_fits_bootstrap.get("bootstrap_summary")
    if not isinstance(wilson_bootstrap, dict):
        wilson_bootstrap = wilson_fits_bootstrap.get("nonlinear")
    if isinstance(wilson_bootstrap, dict):
        rows.append(
            extract_fit_row(
                label="Bootstrap fit",
                discretization="Wilson",
                fit=wilson_bootstrap,
                include_linear_a_terms=True,
            )
        )

    return rows


def build_table(rows, output_file):
    header_line = (
        "Fit & Discretization & $m_{M,\\chi}^2$ & $L_{m_M}$ & $Q_{m_M}$ & "
        "$W_{m_M}$ & $R_{m_M}$ & $C_{m_M}$ & $\\chi^2/\\mathrm{d.o.f.}$ \\\\\n"
    )
    longtable_spec = "|c|c|c|c|c|c|c|c|c|"

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        build_longtable_scaffold(f, header_line, longtable_spec)

        for i, row in enumerate(rows):
            line = (
                f"{row['label']} & "
                f"{row['discretization']} & "
                f"{row['m_M_chi_sq']} & "
                f"{row['L_m_M']} & "
                f"{row['Q_m_M']} & "
                f"{row['W_m_M']} & "
                f"{row['R_m_M']} & "
                f"{row['C_m_M']} & "
                f"{row['chi2_over_dof']}"
            )
            if i < len(rows) - 1:
                line += r" \\"
            f.write(line + "\n")

    print(f"[extrapolation_tables] wrote {output_file} with {len(rows)} rows")


def collect_wilson_point_rows(mv_fit_bootstrap_data):
    rows = []
    for point in mv_fit_bootstrap_data.get("points", {}).get("wilson", []):
        rows.append(
            {
                "ensemble": point.get("Ensemble", "—"),
                "beta": f"{to_float(point.get('beta')):.2f}" if np.isfinite(to_float(point.get("beta"))) else "—",
                "m0": f"{to_float(point.get('m0')):.3f}" if np.isfinite(to_float(point.get("m0"))) else "—",
                "x": format_phys_err(point.get("x"), point.get("xerr"), force_decimals=5),
                "y": format_phys_err(point.get("y"), point.get("yerr"), force_decimals=5),
                "a_over_w0": format_phys_err(
                    point.get("a_over_w0"), point.get("a_over_w0_err"), force_decimals=5
                ),
            }
        )
    return rows


def build_wilson_points_table(rows, output_file):
    header_line = (
        "Ensemble & $\\beta$ & $am_0$ & $(m_{\\rm PS} w_0)^2$ & "
        "$(m_{\\rm V} w_0)^2$ & $a/w_0$ \\\\\n"
    )
    longtable_spec = "|c|c|c|c|c|c|"

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        build_longtable_scaffold(f, header_line, longtable_spec)

        for i, row in enumerate(rows):
            line = (
                f"{row['ensemble']} & "
                f"{row['beta']} & "
                f"{row['m0']} & "
                f"{row['x']} & "
                f"{row['y']} & "
                f"{row['a_over_w0']}"
            )
            if i < len(rows) - 1:
                line += r" \\"
            f.write(line + "\n")

    print(f"[extrapolation_tables] wrote {output_file} with {len(rows)} Wilson points")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX longtable for the mV-mPS extrapolation results, "
            "including central-fit and bootstrap-fit parameters and chi2/d.o.f."
        )
    )
    parser.add_argument(
        "--mv_fit",
        default="intermediary_data/NF2/spectrum/mv_mps_fit.json",
        help="JSON file with central mV-mPS fit results",
    )
    parser.add_argument(
        "--mv_fit_bootstrap",
        default="intermediary_data/NF2/spectrum/mv_mps_fit_bootstrap.json",
        help="JSON file with bootstrap mV-mPS fit results",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output LaTeX table file",
    )
    parser.add_argument(
        "--output_wilson_points",
        default="",
        help="Optional output LaTeX table file for Wilson points",
    )
    args = parser.parse_args()

    mv_fit_data = read_json(args.mv_fit)
    mv_fit_bootstrap_data = read_json(args.mv_fit_bootstrap)
    rows = collect_rows(mv_fit_data, mv_fit_bootstrap_data)
    build_table(rows, args.output_file)

    if args.output_wilson_points:
        wilson_point_rows = collect_wilson_point_rows(mv_fit_bootstrap_data)
        build_wilson_points_table(wilson_point_rows, args.output_wilson_points)


if __name__ == "__main__":
    main()
