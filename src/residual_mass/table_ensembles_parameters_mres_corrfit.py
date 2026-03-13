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
# Reader for m_res.json only
# ---------------------------------------------------------------------

def extract_mres_at_nt_half(data, Nt):
    Nt = to_float(Nt)
    if not np.isfinite(Nt):
        return np.nan, np.nan

    Nt_int = int(round(Nt))
    if Nt_int <= 0:
        return np.nan, np.nan

    t_half = Nt_int // 2

    mres_series = safe_get(data, "mres_series", "mres", default=[])
    mres_err_series = safe_get(data, "mres_series", "mres_err", default=[])

    if not isinstance(mres_series, list) or not isinstance(mres_err_series, list):
        return np.nan, np.nan

    if t_half >= len(mres_series) or t_half >= len(mres_err_series):
        return np.nan, np.nan

    return to_float(mres_series[t_half]), to_float(mres_err_series[t_half])


def read_mres_json(path):
    data = read_json(path)

    Nt = to_float(safe_get(data, "parameters", "Nt", default=np.nan))
    mres_nt_half, mres_nt_half_err = extract_mres_at_nt_half(data, Nt)

    return {
        # parameters
        "beta": to_float(safe_get(data, "parameters", "beta", default=np.nan)),
        "mass": to_float(safe_get(data, "parameters", "mass", default=np.nan)),
        "Nt": Nt,
        "Ns": to_float(safe_get(data, "parameters", "Ns", default=np.nan)),
        "Ls": to_float(safe_get(data, "parameters", "Ls", default=np.nan)),
        "alpha": to_float(safe_get(data, "parameters", "alpha", default=np.nan)),
        "a5": to_float(safe_get(data, "parameters", "a5", default=np.nan)),
        "m5": to_float(
            safe_get(
                data, "parameters", "m5",
                default=safe_get(data, "parameters", "M5", default=np.nan)
            )
        ),
        "mpv": to_float(safe_get(data, "parameters", "mpv", default=np.nan)),

        # analysis settings / ensemble info
        "n_cfg": to_float(safe_get(data, "ensembles", "meas", "n_cfg", default=np.nan)),
        "delta_traj_ps": to_float(
            safe_get(data, "analysis_settings", "delta_traj_ps", default=np.nan)
        ),

        # fitted residual mass results
        "mres": to_float(safe_get(data, "mres_extract", "value", default=np.nan)),
        "mres_err": to_float(safe_get(data, "mres_extract", "error", default=np.nan)),
        "plateau_start": to_float(
            safe_get(
                data, "mres_extract", "plateau_start",
                default=safe_get(data, "analysis_settings", "plateau_start", default=np.nan)
            )
        ),
        "plateau_end": to_float(
            safe_get(
                data, "mres_extract", "plateau_end",
                default=safe_get(data, "analysis_settings", "plateau_end", default=np.nan)
            )
        ),
        "chi2_red": to_float(safe_get(data, "mres_extract", "reduced_chi2", default=np.nan)),

        # mres at Nt/2 from the time series
        "mres_nt_half": mres_nt_half,
        "mres_nt_half_err": mres_nt_half_err,

        # tau_int(pt_ll) embedded in m_res.json
        "tau_int_ptll": to_float(
            safe_get(data, "mres_extract", "ptll_tau_int", "tau_int", default=np.nan)
        ),
        "tau_int_ptll_err": to_float(
            safe_get(data, "mres_extract", "ptll_tau_int", "tau_int_err", default=np.nan)
        ),

        "_source_file": str(path),
    }


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

    numeric_cols = ["beta", "mass", "Nt", "Ns", "Ls", "alpha", "a5", "mpv"]
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
# Matching JSON rows to metadata rows
# ---------------------------------------------------------------------

def match_json_to_metadata(meta, rec):
    required = ["beta", "mass", "Nt", "Ns", "Ls", "alpha", "a5", "m5", "mpv"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Metadata is missing required matching columns: {missing}")

    mask = (
        np.isclose(meta["beta"], rec["beta"], equal_nan=False) &
        np.isclose(meta["mass"], rec["mass"], equal_nan=False) &
        np.isclose(meta["Nt"], rec["Nt"], equal_nan=False) &
        np.isclose(meta["Ns"], rec["Ns"], equal_nan=False) &
        np.isclose(meta["Ls"], rec["Ls"], equal_nan=False) &
        np.isclose(meta["alpha"], rec["alpha"], equal_nan=False) &
        np.isclose(meta["a5"], rec["a5"], equal_nan=False) &
        np.isclose(meta["m5"], rec["m5"], equal_nan=False) &
        np.isclose(meta["mpv"], rec["mpv"], equal_nan=False)
    )

    matches = meta[mask]

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous metadata match for {rec['_source_file']}: "
            f"matched {len(matches)} rows"
        )

    return matches.iloc[0]


def build_dataframe(mres_files, metadata_csv, use_name):
    meta = read_metadata(metadata_csv, use_name)

    rows = []
    for mres_file in mres_files:
        try:
            rec = read_mres_json(mres_file)
        except Exception as e:
            print(f"Warning: could not read m_res JSON {mres_file}: {e}")
            continue

        match = match_json_to_metadata(meta, rec)

        if match is None:
            rec["name"] = Path(mres_file).parent.parent.name
            rec["_meta_order"] = 10**9
        else:
            rec["name"] = match["name"]
            rec["_meta_order"] = match["_meta_order"]

            for c in ["Nt", "Ns", "Ls"]:
                if not np.isfinite(to_float(rec.get(c, np.nan))) and c in match.index:
                    rec[c] = to_float(match[c])

        rows.append(rec)

    if not rows:
        raise RuntimeError("No residual-mass records were loaded.")

    df = pd.DataFrame(rows)

    df["am_res"] = df.apply(
        lambda r: format_phys_err(r["mres"], r["mres_err"]),
        axis=1
    )
    df["am_res_nt_half"] = df.apply(
        lambda r: format_phys_err(r["mres_nt_half"], r["mres_nt_half_err"]),
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

def build_table(df, output_table, use_name):
    is_scan_beta = (use_name == "scan_beta")

    if is_scan_beta:
        header_line = (
            "Ensemble & $\\beta$ & $am_0$ & $N_t$ & $N_s$ & $L_s$ & "
            "$\\alpha$ & $a_5/a$ & $am_5$ & $am_{\\rm PV}$ & "
            "$n_{\\rm cfg}$ & $\\delta_{\\rm traj}^{\\rm PS}$ & "
            "$\\tau_{\\rm int}^{\\rm PS}$ & "
            "$am_{\\rm res}(aN_t/2)$ \\\\\n"
        )
        longtable_spec = "|c|c|c|c|c|c|c|c|c|c|c|c|c|c|"
    else:
        header_line = (
            "Ensemble & $\\beta$ & $am_0$ & $N_t$ & $N_s$ & $L_s$ & "
            "$\\alpha$ & $a_5/a$ & $am_5$ & $am_{\\rm PV}$ & "
            "$n_{\\rm cfg}$ & $\\delta_{\\rm traj}^{\\rm PS}$ & "
            "$\\tau_{\\rm int}^{\\rm PS}$ & "
            "$am_{\\rm res}$ & $\\tilde{t}^{am_{\\rm res}}_{\\rm start}$ & "
            "$\\tilde{t}^{am_{\\rm res}}_{\\rm end}$ & "
            "$\\chi^2_{\\rm red}$ \\\\\n"
        )
        longtable_spec = "|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|"

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
            if is_scan_beta:
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
                    f"{format_floatish(r['mpv'], '.3g')} & "
                    f"{format_intish(r['n_cfg'])} & "
                    f"{format_intish(r['delta_traj_ps'])} & "
                    f"{r['tau_int_ptll_fmt']} & "
                    f"{r['am_res_nt_half']}"
                )
            else:
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
                    f"{format_floatish(r['mpv'], '.3g')} & "
                    f"{format_intish(r['n_cfg'])} & "
                    f"{format_intish(r['delta_traj_ps'])} & "
                    f"{r['tau_int_ptll_fmt']} & "
                    f"{r['am_res']} & "
                    f"{format_intish(r['plateau_start'])} & "
                    f"{format_intish(r['plateau_end'])} & "
                    f"{format_floatish(r['chi2_red'], '.3f')}"
                )

            if i < nrows - 1:
                line += r" \\"

            f.write(line + "\n")

    print(f"[table_mres] wrote {output_table} with {len(df)} ensembles")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX longtable from m_res.json files, using metadata "
            "only for ensemble selection, names, and ordering."
        )
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
        required=True,
        help="Selection name; metadata rows are filtered using column use_in_<use>.",
    )

    args = parser.parse_args()

    df = build_dataframe(args.mres, args.metadata_csv, args.use)

    build_table(df, args.output_table, args.use)


if __name__ == "__main__":
    main()