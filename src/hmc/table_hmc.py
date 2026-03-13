#!/usr/bin/env python3
"""
Build one or more LaTeX longtables summarizing HMC results.

Features:
- HMC quantities are read from the files passed on the command line.
- beta and mass are read from keys_from_path in each HMC JSON.
- Missing beta/mass are printed as "-".
- Output is a longtable with multipage headers.
- Metadata rows are filtered using --use, i.e. column use_in_<use>.
- The last table row does not end with '\\'.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

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


def first_finite(*vals):
    for v in vals:
        v = to_float(v)
        if np.isfinite(v):
            return v
    return np.nan


def latex_escape(x):
    if x is None:
        return "-"
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return "-"
    return s.replace("_", r"\_")


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------

def format_phys_err(value, error, force_decimals=None):
    value = float(value)
    error = abs(float(error))

    if not np.isfinite(value) or not np.isfinite(error):
        return "—"
    if error == 0:
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
    else:
        err_str = f"{error_r:.{decimals}f}"
        return f"{value_str}({err_str})"


def format_intish(x):
    x = to_float(x)
    if np.isfinite(x):
        return str(int(round(x)))
    return "—"


def format_floatish(x, fmt=".3g"):
    x = to_float(x)
    if np.isfinite(x):
        return format(x, fmt)
    return "—"


def format_stringish(x):
    if x is None:
        return "-"
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return "-"
    return latex_escape(s)


# ---------------------------------------------------------------------
# Reading HMC data
# ---------------------------------------------------------------------

def read_hmc_file(path):
    """
    Read one HMC JSON file.
    beta and mass are read from keys_from_path.
    Missing beta/mass are allowed and later formatted as '-'.
    """
    data = read_json(path)

    rec = {
        "name": str(safe_get(data, "ensemble", "name", default="")).strip(),

        "beta": safe_get(data, "keys_from_path", "beta", default=np.nan),
        "mass": safe_get(data, "keys_from_path", "mass", default=np.nan),

        "therm": to_float(safe_get(data, "ensemble", "therm")),
        "delta_traj": to_float(safe_get(data, "ensemble", "delta_traj")),

        "length_traj": to_float(safe_get(data, "hmc_extract", "length_traj")),
        "n_steps": to_float(safe_get(data, "hmc_extract", "n_steps")),
        "n_conf": to_float(safe_get(data, "hmc_extract", "n_conf")),

        "fullbcs": to_float(safe_get(data, "hmc_extract", "fullbcs")),
        "fullbcs_err": to_float(safe_get(data, "hmc_extract", "fullbcs_err")),

        "t_traj": to_float(safe_get(data, "hmc_extract", "t_traj")),
        "t_traj_err": to_float(safe_get(data, "hmc_extract", "t_traj_err")),

        "accept_ratio": to_float(safe_get(data, "hmc_extract", "accept_ratio")),

        "plaq": to_float(safe_get(data, "hmc_extract", "plaq")),
        "plaq_err": to_float(safe_get(data, "hmc_extract", "plaq_err")),

        "tau_int_plaq": to_float(safe_get(data, "hmc_extract", "tau_int_plaq")),
        "tau_int_plaq_err": to_float(safe_get(data, "hmc_extract", "tau_int_plaq_err")),

        "source_hmc_file": str(path),
    }

    return rec


# ---------------------------------------------------------------------
# Metadata grouping / ordering
# ---------------------------------------------------------------------

def split_metadata_by_blank_rows(meta):
    groups = []
    current = []

    for _, row in meta.iterrows():
        if row.isna().all():
            if current:
                groups.append(pd.DataFrame(current))
                current = []
        else:
            current.append(row)

    if current:
        groups.append(pd.DataFrame(current))

    return groups


def normalize_flag_series(series):
    """
    Robust conversion of a metadata flag column to booleans.
    Accepts bools, 1/0, yes/no, true/false.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean").fillna(False).astype(bool)

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


# ---------------------------------------------------------------------
# LaTeX writer
# ---------------------------------------------------------------------

def build_table(df, output_table):
    header_line = (
        "Ensemble & $\\beta$ & $am_0$ & "
        "$l_{\\mathrm{traj}}$ & $n_{\\mathrm{steps}}$ & "
        "$n_{\\mathrm{therm}}$ & $\\delta_{\\mathrm{traj}}^{\\mathrm{plaq}}$ & "
        "$n_{\\mathrm{conf}}$ & $n_{\\mathrm{CG}}$ & "
        "$t_{\\mathrm{traj}}[\\mathrm{s}]$ & "
        "$\\mathrm{Acc.\\ [\\%]}$ & "
        "$\\langle P \\rangle$ & "
        "$\\tau_{\\mathrm{int}}^{\\mathrm{plaq}}$ \\\\\n"
    )

    outdir = os.path.dirname(output_table)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(output_table, "w") as f:
        f.write("%%%\\begin{longtable}{|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n")

        f.write("% ===================== FIRST PAGE HEADER =====================\n")

        f.write(header_line)
        f.write("\\hline\n")
        f.write("\\endfirsthead\n\n")

        f.write("% ===================== HEADER FOR PAGE 2+ =====================\n")
        f.write("\\hline\n")
        f.write(header_line)
        f.write("\\hline\n")
        f.write("\\endhead\n\n")

        f.write("% ===================== FOOTER FOR INTERMEDIATE PAGES =====================\n")
        f.write("\\hline\n")
        f.write("\\endfoot\n\n")

        f.write("% ===================== FINAL FOOTER =====================\n")
        f.write("\\hline\\hline\n")
        f.write("\\endlastfoot\n\n")

        nrows = len(df)

        for i, (_, r) in enumerate(df.iterrows()):
            end = " \\\\\n" if i < nrows - 1 else "\n"

            row = (
                f"{latex_escape(r['name'])} & "
                f"{r['beta_fmt']} & "
                f"{r['mass_fmt']} & "
                f"{format_floatish(r['length_traj'], '.3g')} & "
                f"{format_intish(r['n_steps'])} & "
                f"{format_intish(r['therm'])} & "
                f"{format_intish(r['delta_traj'])} & "
                f"{format_intish(r['n_conf'])} & "
                f"{r['fullbcs_fmt']} & "
                f"{r['t_traj_fmt']} & "
                f"{r['accept_fmt']} & "
                f"{r['plaq_fmt']} & "
                f"{r['tau_fmt']}"
                f"{end}"
            )

            f.write(row)

        f.write("%%%\\end{longtable}\n")

    print(f"[table_hmc] wrote {output_table} with {len(df)} ensembles")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build LaTeX longtables from HMC JSON files, including beta and mass from keys_from_path."
    )
    parser.add_argument(
        "--hmc",
        nargs="+",
        required=True,
        help="List of HMC JSON files, e.g. .../hmc/log_hmc_extract.json",
    )
    parser.add_argument(
        "--metadata_csv",
        required=True,
        help="Metadata CSV used to define ordering and blank-row splits.",
    )
    parser.add_argument(
        "--output_table",
        required=True,
        help="Base output .tex path.",
    )
    parser.add_argument(
        "--use",
        required=True,
        help="Selection name; metadata rows are filtered using column use_in_<use>.",
    )
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata_csv, sep=r"\t|,", engine="python")

    flagcol = f"use_in_{args.use}"
    if flagcol not in meta.columns:
        raise ValueError(f"Column '{flagcol}' not found in {args.metadata_csv}")

    mask = normalize_flag_series(meta[flagcol])
    meta_sel = meta[mask].copy()

    if meta_sel.empty:
        raise ValueError(f"No rows selected by column '{flagcol}'")

    if "name" not in meta_sel.columns:
        raise ValueError("metadata_csv must contain a 'name' column")

    meta_sel["name"] = meta_sel["name"].astype(str).str.strip()

    groups = split_metadata_by_blank_rows(meta_sel)
    print(f"[table_hmc] Found {len(groups)} metadata group(s) for {flagcol}")

    records = [read_hmc_file(path) for path in args.hmc]
    df_all = pd.DataFrame(records)

    if df_all.empty:
        raise RuntimeError("No HMC records were loaded.")

    df_all["name"] = df_all["name"].astype(str).str.strip()

    df_all["beta_fmt"] = df_all["beta"].apply(format_stringish)
    df_all["mass_fmt"] = df_all["mass"].apply(format_stringish)

    df_all["fullbcs_fmt"] = df_all.apply(
        lambda x: "-" if to_float(x["fullbcs"]) == 0 else format_phys_err(x["fullbcs"], x["fullbcs_err"]),
        axis=1
    )
    df_all["t_traj_fmt"] = df_all.apply(
        lambda x: format_phys_err(x["t_traj"], x["t_traj_err"]), axis=1
    )
    df_all["plaq_fmt"] = df_all.apply(
        lambda x: format_phys_err(x["plaq"], x["plaq_err"]), axis=1
    )
    df_all["tau_fmt"] = df_all.apply(
        lambda x: format_phys_err(x["tau_int_plaq"], x["tau_int_plaq_err"]), axis=1
    )
    df_all["accept_fmt"] = df_all["accept_ratio"].apply(
        lambda x: f"{int(round(100 * x))}" if np.isfinite(to_float(x)) else "—"
    )

    for i, group in enumerate(groups, start=1):
        order = pd.unique(group["name"].dropna().astype(str).str.strip()).tolist()
        df = df_all[df_all["name"].isin(order)].copy()

        if df.empty:
            print(f"[table_hmc] warning: no data found for group {i}")
            continue

        df = df.drop_duplicates(subset="name", keep="first")

        df["name"] = pd.Categorical(df["name"], categories=order, ordered=True)
        df = df.sort_values("name").reset_index(drop=True)

        if len(groups) == 1:
            output_path = args.output_table
        else:
            base, ext = os.path.splitext(args.output_table)
            output_path = f"{base}_{i}{ext}"

        build_table(df, output_path)


if __name__ == "__main__":
    main()