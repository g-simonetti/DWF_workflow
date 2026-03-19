#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Generic helpers
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


def normalize_flag_series(series):
    s = series.astype("string")
    return s.str.strip().str.lower().isin(["true", "1", "yes", "y"])


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


def format_floatish(x, fmt=".3f"):
    x = to_float(x)
    if np.isfinite(x):
        return format(x, fmt)
    return "—"


# ---------------------------------------------------------------------
# Path parsing helpers
# ---------------------------------------------------------------------

def extract_from_path(path):
    vals = {
        "beta": np.nan,
        "mass": np.nan,
        "Nt": np.nan,
        "Ns": np.nan,
        "Ls": np.nan,
        "alpha": np.nan,
        "a5": np.nan,
        "m5": np.nan,
        "mpv": np.nan,
    }

    parts = Path(path).parts

    for p in parts:
        if p.startswith("B"):
            vals["beta"] = to_float(p[1:], vals["beta"])
        elif p.startswith("Nt"):
            vals["Nt"] = to_float(p[2:], vals["Nt"])
        elif p.startswith("Ns"):
            vals["Ns"] = to_float(p[2:], vals["Ns"])
        elif p.startswith("Ls"):
            vals["Ls"] = to_float(p[2:], vals["Ls"])
        elif p.startswith("alpha"):
            vals["alpha"] = to_float(p[5:], vals["alpha"])
        elif p.startswith("a5"):
            vals["a5"] = to_float(p[2:], vals["a5"])
        elif p.startswith("mpv"):
            vals["mpv"] = to_float(p[3:], vals["mpv"])
        elif p.startswith("M5"):
            vals["m5"] = to_float(p[2:], vals["m5"])
        elif p.startswith("M") and not p.startswith("M5") and not np.isfinite(vals["mass"]):
            vals["mass"] = to_float(p[1:], vals["mass"])

    return vals


def merge_prefer_finite(primary, fallback):
    out = dict(primary)
    for k, v in fallback.items():
        if not np.isfinite(to_float(out.get(k, np.nan))):
            out[k] = v
    return out


# ---------------------------------------------------------------------
# Shared parameter reader
# ---------------------------------------------------------------------

def read_common_parameters(data, path):
    params = safe_get(data, "parameters", default={})

    from_json = {
        "beta": to_float(safe_get(params, "beta", default=np.nan)),
        "mass": to_float(safe_get(params, "mass", default=np.nan)),
        "Nt": to_float(safe_get(params, "Nt", default=np.nan)),
        "Ns": to_float(safe_get(params, "Ns", default=np.nan)),
        "Ls": to_float(safe_get(params, "Ls", default=np.nan)),
        "alpha": to_float(safe_get(params, "alpha", default=np.nan)),
        "a5": to_float(safe_get(params, "a5", default=np.nan)),
        "m5": to_float(
            safe_get(params, "m5", default=safe_get(params, "M5", default=np.nan))
        ),
        "mpv": to_float(safe_get(params, "mpv", default=np.nan)),
    }

    from_path = extract_from_path(path)
    return merge_prefer_finite(from_json, from_path)


# ---------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------

def read_spectrum_json(path):
    data = read_json(path)

    rec = read_common_parameters(data, path)

    rec["n_cfg"] = to_float(
        safe_get(data, "data_shape", "Ncfg", default=np.nan)
    )
    rec["mps"] = to_float(
        safe_get(data, "results", "standard_fit", "PP", "am_ps", "mean", default=np.nan)
    )
    rec["mps_err"] = to_float(
        safe_get(data, "results", "bootstrap_fit", "PP", "am_ps", "sdev", default=np.nan)
    )
    rec["chi2_red"] = to_float(
        safe_get(
            data,
            "results", "standard_fit", "PP", "fit_stats", "chi2_over_dof",
            default=np.nan,
        )
    )
    rec["plateau_start"] = to_float(
        safe_get(data, "windows", "ps", "t0", default=np.nan)
    )
    rec["plateau_end"] = to_float(
        safe_get(data, "windows", "ps", "t1", default=np.nan)
    )

    if not np.isfinite(rec["mps"]) or not np.isfinite(rec["mps_err"]):
        raise ValueError(
            f"Missing or invalid PS mass data in spectrum JSON '{path}'.\n"
            "Expected numeric values at:\n"
            "  results -> standard_fit -> PP -> am_ps -> mean\n"
            "  results -> bootstrap_fit -> PP -> am_ps -> sdev"
        )

    rec["_source_file_mps"] = str(path)
    return rec


def read_mres_json(path):
    data = read_json(path)

    rec = read_common_parameters(data, path)

    rec["tau_int_ptll"] = to_float(
        safe_get(data, "mres_extract", "ptll_tau_int", "tau_int", default=np.nan)
    )
    rec["tau_int_ptll_err"] = to_float(
        safe_get(data, "mres_extract", "ptll_tau_int", "tau_int_err", default=np.nan)
    )

    rec["_source_file_mres"] = str(path)
    return rec


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def read_metadata(metadata_csv, use_name):
    meta = pd.read_csv(metadata_csv, sep=r"\t|,", engine="python")

    flagcol = f"use_in_{use_name}"
    if flagcol not in meta.columns:
        raise ValueError(f"Column '{flagcol}' not found in {metadata_csv}")

    meta = meta[normalize_flag_series(meta[flagcol])].copy()
    if meta.empty:
        raise ValueError(f"No rows selected by column '{flagcol}'")

    if "name" not in meta.columns:
        raise ValueError(f"Column 'name' not found in {metadata_csv}")

    meta["name"] = meta["name"].astype(str).str.strip()

    numeric_cols = ["beta", "mass", "Nt", "Ns", "Ls", "alpha", "a5", "mpv", "delta_traj_ps"]
    for c in numeric_cols:
        if c in meta.columns:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")

    if "M5" in meta.columns:
        meta["m5"] = pd.to_numeric(meta["M5"], errors="coerce")
    elif "m5" in meta.columns:
        meta["m5"] = pd.to_numeric(meta["m5"], errors="coerce")
    else:
        meta["m5"] = np.nan

    meta["_meta_order"] = np.arange(len(meta))
    return meta


# ---------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------

MATCH_COLS = ["beta", "mass", "Nt", "Ns", "Ls", "alpha", "a5", "m5", "mpv"]


def match_record(meta, rec):
    missing = [c for c in MATCH_COLS if c not in meta.columns]
    if missing:
        raise ValueError(f"Metadata is missing required matching columns: {missing}")

    mask = np.ones(len(meta), dtype=bool)
    for c in MATCH_COLS:
        rv = to_float(rec.get(c, np.nan))
        if not np.isfinite(rv):
            return None
        mask &= np.isclose(meta[c], rv, equal_nan=False)

    matches = meta[mask]

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        src = rec.get("_source_file_mps", rec.get("_source_file_mres", "<unknown>"))
        raise ValueError(
            f"Ambiguous metadata match for {src}: matched {len(matches)} rows"
        )

    return matches.iloc[0]


def record_key(rec):
    vals = []
    for c in MATCH_COLS:
        x = to_float(rec.get(c, np.nan))
        if np.isfinite(x):
            vals.append(round(x, 12))
        else:
            vals.append(np.nan)
    return tuple(vals)


# ---------------------------------------------------------------------
# Build dataframe
# ---------------------------------------------------------------------

def build_dataframe(mps_files, mres_files, metadata_csv, use_name):
    meta = read_metadata(metadata_csv, use_name)

    mps_map = {}
    for path in mps_files:
        try:
            rec = read_spectrum_json(path)
        except Exception as e:
            print(f"Warning: could not read spectrum JSON {path}: {e}")
            continue
        mps_map[record_key(rec)] = rec

    mres_map = {}
    for path in mres_files:
        try:
            rec = read_mres_json(path)
        except Exception as e:
            print(f"Warning: could not read m_res JSON {path}: {e}")
            continue
        mres_map[record_key(rec)] = rec

    all_keys = list(dict.fromkeys(list(mps_map.keys()) + list(mres_map.keys())))
    rows = []

    for key in all_keys:
        rec = {}

        if key in mps_map:
            rec.update(mps_map[key])
        if key in mres_map:
            rec.update(mres_map[key])

        match = match_record(meta, rec)

        if match is None:
            fallback = rec.get("_source_file_mps", rec.get("_source_file_mres", "unknown"))
            rec["name"] = Path(fallback).parent.parent.name
            rec["_meta_order"] = 10**9
        else:
            rec["name"] = match["name"]
            rec["_meta_order"] = match["_meta_order"]

            if "delta_traj_ps" in match.index:
                rec["delta_traj_ps"] = to_float(match["delta_traj_ps"])

            for c in ["Nt", "Ns", "Ls"]:
                if not np.isfinite(to_float(rec.get(c, np.nan))) and c in match.index:
                    rec[c] = to_float(match[c])

        rows.append(rec)

    if not rows:
        raise RuntimeError("No finite-volume records were loaded.")

    df = pd.DataFrame(rows)

    for col in [
        "beta", "mass", "Nt", "Ns", "Ls", "alpha", "a5", "m5",
        "n_cfg", "delta_traj_ps",
        "tau_int_ptll", "tau_int_ptll_err",
        "mps", "mps_err",
        "plateau_start", "plateau_end",
        "chi2_red",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["am_ps_fmt"] = df.apply(
        lambda r: format_phys_err(r["mps"], r["mps_err"]),
        axis=1
    )
    df["tau_int_ptll_fmt"] = df.apply(
        lambda r: format_phys_err(r["tau_int_ptll"], r["tau_int_ptll_err"]),
        axis=1
    )

    df = (
        df.sort_values(["_meta_order", "name"])
          .drop_duplicates(subset=["name"], keep="first")
          .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------
# LaTeX writer
# ---------------------------------------------------------------------

def build_table(df, output_table):
    header_line = (
        "Ensemble & $\\beta$ & $am_0$ & $N_t$ & $N_s$ & $L_s$ & "
        "$\\alpha$ & $a_5/a$ & $am_5$ & "
        "$n_{\\rm cfg}$ & $\\delta_{\\rm traj}^{\\rm PS}$ & "
        "$\\tau_{\\rm int}^{\\rm PS}$ & "
        "$am_{\\rm PS}$ & "
        "$\\tilde{t}^{am_{\\rm PS}}_{\\rm start}$ & "
        "$\\tilde{t}^{am_{\\rm PS}}_{\\rm end}$ & "
        "$\\chi^2_{\\rm red}$ \\\\\n"
    )
    longtable_spec = "|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|"

    out_dir = os.path.dirname(output_table)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_table, "w") as f:
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

        nrows = len(df)
        for i, (_, r) in enumerate(df.iterrows()):
            line = (
                f"{r['name']} & "
                f"{format_floatish(r['beta'], '.1f')} & "
                f"{format_floatish(r['mass'], '.3g')} & "
                f"{format_intish(r['Nt'])} & "
                f"{format_intish(r['Ns'])} & "
                f"{format_intish(r['Ls'])} & "
                f"{format_floatish(r['alpha'], '.3g')} & "
                f"{format_floatish(r['a5'], '.3g')} & "
                f"{format_floatish(r['m5'], '.3g')} & "
                f"{format_intish(r['n_cfg'])} & "
                f"{format_intish(r['delta_traj_ps'])} & "
                f"{r['tau_int_ptll_fmt']} & "
                f"{r['am_ps_fmt']} & "
                f"{format_intish(r['plateau_start'])} & "
                f"{format_intish(r['plateau_end'])} & "
                f"{format_floatish(r['chi2_red'], '.3f')}"
            )

            if i < nrows - 1:
                line += r" \\"

            f.write(line + "\n")

    print(f"[table_finite_volume] wrote {output_table} with {len(df)} ensembles")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX longtable from spectrum.json and m_res.json files, "
            "using metadata for ensemble selection, names, ordering, and delta_traj_ps."
        )
    )
    parser.add_argument(
        "--mps",
        nargs="+",
        required=True,
        help="List of spectrum.json files",
    )
    parser.add_argument(
        "--mres",
        nargs="+",
        required=True,
        help="List of m_res.json files",
    )
    parser.add_argument(
        "--metadata_csv",
        required=True,
        help="Path to ensembles.csv",
    )
    parser.add_argument(
        "--output_table",
        required=True,
        help="Output LaTeX file",
    )
    parser.add_argument(
        "--use",
        default="finite_volume",
        help="Selection name; metadata rows are filtered using column use_in_<use>.",
    )

    args = parser.parse_args()

    df = build_dataframe(args.mps, args.mres, args.metadata_csv, args.use)
    build_table(df, args.output_table)


if __name__ == "__main__":
    main()