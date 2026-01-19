import argparse
import pandas as pd
import numpy as np
import os


def read_mres_file(path):
    """Read the residual mass file and return mres, mres_err, plateau_start, plateau_end."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Residual mass file not found: {path}")
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = line.split()
            if len(vals) < 4:
                raise ValueError(f"Unexpected format in {path}")
            mres, mres_err, pstart, pend = vals[:4]
            return float(mres), float(mres_err), int(pstart), int(pend)
    raise ValueError(f"No valid data lines in {path}")


def format_with_err(value, err):
    """Format value ± err as 0.00234 (32), matching last digits of the value."""
    if value is None or err is None or np.isnan(value) or np.isnan(err):
        return "--"

    if err == 0:
        return f"{value:.5f}"

    # number of digits in error
    exp = int(np.floor(np.log10(err)))
    digits = max(0, -exp + 1)

    value_fmt = f"{{:.{digits+1}f}}"
    value_str = value_fmt.format(value)
    err_rounded = int(round(err * 10**(digits+1)))
    err_str = str(err_rounded).rjust(2, "0")[-2:]  # ensure at least two digits
    return f"{value_str} ({err_str})"


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table of ensemble parameters.")
    parser.add_argument("--ensembles_csv", required=True, help="Path to ensembles.csv")
    parser.add_argument("--output_file", required=True, help="Output LaTeX file")
    args = parser.parse_args()

    df = pd.read_csv(args.ensembles_csv, sep="\t|,", engine="python")

    # Extract subdir from path: metadata/<subdir>/ensembles.csv
    parts = args.ensembles_csv.split(os.sep)
    subdir = parts[1] if len(parts) > 2 else ""

    rows = []
    for _, row in df.iterrows():
        try:
            path = (
                f"intermediary_data/{subdir}/NF2/"
                f"Nt{int(row['Nt'])}/Ns{int(row['Ns'])}/Ls{int(row['Ls'])}/"
                f"B{row['beta']}/M{row['mass']}/"
                f"mpv{row['mpv']}/alpha{row['alpha']}/a5{row['a5']}/M5{row['M5']}/"
                "residual_mass/m_res_fit.txt"
            )
            mres, mres_err, pstart, pend = read_mres_file(path)
        except Exception as e:
            print(f"Warning: could not read m_res file for {row['name']}: {e}")
            mres = mres_err = pstart = pend = np.nan

        rows.append({
            "name": row["name"],
            "beta": row["beta"],
            "mass": row["mass"],
            "Nt": int(row["Nt"]),
            "Ns": int(row["Ns"]),
            "Ls": int(row["Ls"]),
            "alpha": row["alpha"],
            "a5": row["a5"],
            "m5": row["M5"],
            "mpv": row["mpv"],
            "mres": mres,
            "mres_err": mres_err,
            "mres_p_start": pstart,
            "mres_p_end": pend
        })

    df_out = pd.DataFrame(rows)

    # Apply custom formatting for mres ± err
    df_out["am_res"] = df_out.apply(lambda x: format_with_err(x["mres"], x["mres_err"]), axis=1)

    # Keep only the desired columns
    df_out = df_out[
        ["name", "beta", "mass", "Nt", "Ns", "Ls", "alpha", "a5", "m5", "mpv", "am_res", "mres_p_start", "mres_p_end"]
    ]

    # Column labels in LaTeX
    header = (
        "name & $\\beta$ & $am_0$ & $N_t$ & $N_s$ & $L_s$ & "
        "$\\alpha$ & $a_5$ & $m_5$ & $m_{\\rm PV}$ & "
        "$am_{\\rm res}$ & $am_{\\rm res}^{\\rm plateau\\ start}$ & "
        "$am_{\\rm res}^{\\rm plateau\\ end}$ \\\\"
    )

    # Write LaTeX table
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\begin{tabular}{lcccccccccccc}\n")
        f.write("\\hline\n")
        f.write(header + "\n")
        f.write("\\hline\n")
        for _, r in df_out.iterrows():
            f.write(
                f"{r['name']} & {r['beta']} & {r['mass']} & {r['Nt']} & {r['Ns']} & {r['Ls']} & "
                f"{r['alpha']} & {r['a5']} & {r['m5']} & {r['mpv']} & "
                f"{r['am_res']} & {r['mres_p_start']} & {r['mres_p_end']} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Ensemble parameters with residual mass values.}\n")
        f.write("\\label{tab:ensemble_parameters}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table written to {args.output_file}")


if __name__ == "__main__":
    main()
