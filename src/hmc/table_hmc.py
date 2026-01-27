#!/usr/bin/env python3
"""
Build one or more LaTeX longtables summarizing HMC results from multiple log_hmc_extract.txt files.

Now formatted to match the requested red longtable style with caption, label, and double lines.
"""

import argparse
import numpy as np
import pandas as pd
import os


def read_extract_file(path):
    """Read a single log_hmc_extract.txt into a dict."""
    with open(path) as f:
        header = f.readline().split()
        values = f.readline().split()
    d = {}
    for h, v in zip(header, values):
        if h == "name":
            d[h] = v
        else:
            try:
                d[h] = float(v)
            except ValueError:
                d[h] = np.nan
    return d


def format_phys_err(value, error, ndigits_value=3):
    """Format as value(error) where error is printed as a float, e.g. 7.8(9.8)."""
    if not np.isfinite(value) or not np.isfinite(error):
        return "—"

    error = abs(float(error))

    if error == 0:
        return f"{value:.{ndigits_value}g}"

    # Choose how many decimals to show based on the *error* scale (like your current code)
    exp = int(np.floor(np.log10(error)))  # error ~ 10^exp
    digits = max(0, -exp + 1)             # keep ~2 significant digits in error

    # Print both with the same number of decimals
    value_str = f"{value:.{digits}f}"
    err_str = f"{error:.{digits}f}"

    # Trim trailing zeros in the error (keeps "9.8", turns "9.80"->"9.8")
    err_str = err_str.rstrip("0").rstrip(".") if "." in err_str else err_str

    return f"{value_str}({err_str})"


def build_table(df, output_file):
    """Generate and save a single LaTeX longtable with the requested multi-page styling."""

    # -------- HEADER LINE (used in firsthead and head) --------
    header_line = (
        "Ensemble & $l_{\\mathrm{traj}}$ & $n_{\\mathrm{steps}}$ & "
        "$n_{\\mathrm{therm}}$ & $\\delta_{\\mathrm{traj}}$ & "
        "$n_{\\mathrm{conf}}$ & $n_{\\mathrm{CG}}$ & "
        "$t_{\\mathrm{traj}}[\\rm s]$ & "
        "$\\mathrm{Acceptance\\ [\\%]}$ & "
        "$\\tau_{\\mathrm{int}}^{\\mathrm{plaq}}$ \\\\\n"
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:

        # ============================================================
        #   BEGIN LONGTABLE + CAPTION
        # ============================================================
        f.write("\\color{red}\n")
        f.write("\\begin{longtable}{|c|c|c|c|c|c|c|c|c|c|}\n")

        f.write(
            "\\caption{Characterisation of the ensembles generated to study the parameter scan of "
            "$Sp(4)$ with $N_{\\rm f} = 2$ Dirac fermions transforming in the fundamental "
            "representation, realised with the MDWF formalism. For each ensemble, we tabulate here "
            "the length of the trajectory and the number of molecular dynamics steps, "
            "$l_{\\rm traj}$ and $n_{\\rm steps}$, the number of thermalisation trajectories, "
            "$n_{\\rm therm}$, the trajectory separation and the total number of the analysed "
            "configurations, $\\delta_{\\rm traj}$ and $n_{\\rm conf}$, the number of the conjugate "
            "gradient (CG) applications and the time expressed in seconds for a single trajectory, "
            "$n_{\\rm CG}$ and $t_{\\rm traj}$, the acceptance of the HMC and the integrated "
            "autocorrelation time for the average plaquette, $\\tau_{\\rm int}^{\\rm plaq}$.}\n"
        )
        f.write("\\label{tab:hmc_summary} \\\\\n\n")

        # ============================================================
        #   FIRST PAGE HEADER
        # ============================================================
        f.write("% ===================== FIRST PAGE HEADER =====================\n")
        f.write("\\hline\\hline\n")
        f.write(header_line)
        f.write("\\hline\n")
        f.write("\\endfirsthead\n\n")

        # ============================================================
        #   HEADER FOR PAGE 2+
        # ============================================================
        f.write("% ===================== HEADER FOR PAGE 2+ =====================\n")
        f.write("\\hline\n")
        f.write(header_line)
        f.write("\\hline\n")
        f.write("\\endhead\n\n")

        # ============================================================
        #   FOOTER FOR INTERMEDIATE PAGES
        # ============================================================
        f.write("% ===================== FOOTER FOR INTERMEDIATE PAGES =====================\n")
        f.write("\\hline\n")
        f.write("\\endfoot\n\n")

        # ============================================================
        #   FINAL FOOTER
        # ============================================================
        f.write("% ===================== FINAL FOOTER =====================\n")
        f.write("\\hline\\hline\n")
        f.write("\\endlastfoot\n\n")

        # ============================================================
        #   TABLE BODY
        # ============================================================
        f.write("% ===================== TABLE BODY =====================\n")

        for _, r in df.iterrows():

            length_traj_str = f"{r['length_traj']:.3g}" if np.isfinite(r['length_traj']) else "—"
            n_steps_str     = str(int(r['n_steps'])) if np.isfinite(r['n_steps']) else "—"
            therm_str       = str(int(r['therm'])) if np.isfinite(r['therm']) else "—"
            delta_str       = str(int(r['delta_traj'])) if np.isfinite(r['delta_traj']) else "—"
            nconf_str       = str(int(r['n_conf'])) if np.isfinite(r['n_conf']) else "—"

            row = (
                f"{r['name']} & "
                f"{length_traj_str} & "
                f"{n_steps_str} & "
                f"{therm_str} & "
                f"{delta_str} & "
                f"{nconf_str} & "
                f"{r['fullbcs_fmt']} & "
                f"{r['t_traj_fmt']} & "
                f"{r['accept_fmt']} & "
                f"{r['tau_fmt']} \\\\\n"
            )

            f.write(row)

        f.write("\\end{longtable}\n")

    print(f"[table_hmc] wrote {output_file} with {len(df)} ensembles (multi-page red longtable)")




def main():
    parser = argparse.ArgumentParser(description="Build one or more LaTeX longtables from HMC extract files.")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="List of log_hmc_extract.txt files.")
    parser.add_argument("--metadata_csv", required=True,
                        help="Path to the metadata CSV to define ensemble order and splits (blank rows).")
    parser.add_argument("--output_file", required=True, help="Base output .tex file path.")
    args = parser.parse_args()

    # Read metadata
    meta = pd.read_csv(args.metadata_csv, sep=r"\t|,", engine="python")

    # Split metadata into groups by blank rows
    blank_mask = meta.isnull().all(axis=1)
    groups, current = [], []
    for _, row in meta.iterrows():
        if all(pd.isna(row)):
            if current:
                groups.append(pd.DataFrame(current))
                current = []
        else:
            current.append(row)
    if current:
        groups.append(pd.DataFrame(current))

    print(f"[table_hmc] Found {len(groups)} group(s) of ensembles separated by blank rows")

    # Read all input files
    records = [read_extract_file(f) for f in args.inputs]
    df_all = pd.DataFrame(records)

    # Process each group separately
    for i, group in enumerate(groups, start=1):
        order = group["name"].dropna().tolist()
        df = df_all[df_all["name"].isin(order)].copy()

        if df.empty:
            print(f"[table_hmc] ⚠️ No data found for group {i}")
            continue

        # Reorder
        df["name"] = pd.Categorical(df["name"], categories=order, ordered=True)
        df = df.sort_values("name").reset_index(drop=True)

        # Formatting
        df["fullbcs_fmt"] = df.apply(lambda x: format_phys_err(x["fullbcs"], x["fullbcs_err"]), axis=1)
        df["t_traj_fmt"]  = df.apply(lambda x: format_phys_err(x["t_traj"], x["t_traj_err"]), axis=1)
        df["tau_fmt"]     = df.apply(lambda x: format_phys_err(x["tau_int_plaq"], x["tau_int_plaq_err"]), axis=1)
        df["accept_fmt"]  = df["accept_ratio"].apply(lambda x: f"{int(round(x * 100))}" if np.isfinite(x) else "—")

        # Output file name
        if len(groups) == 1:
            output_path = args.output_file
        else:
            base, ext = os.path.splitext(args.output_file)
            output_path = f"{base}_{i}{ext}"

        build_table(df, output_path)


if __name__ == "__main__":
    main()
