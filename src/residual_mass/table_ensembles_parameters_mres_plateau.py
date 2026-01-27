import argparse
import pandas as pd
import numpy as np
import os


def read_mres_file(path):
    """Read the residual mass file and return mres, mres_err, reduced_chi2, plateau_start, plateau_end."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Residual mass file not found: {path}")
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = line.split()
            if len(vals) < 5:
                raise ValueError(f"Unexpected format in {path}")
            mres, mres_err, chi2_red, pstart, pend = vals[:5]
            return float(mres), float(mres_err), float(chi2_red), int(pstart), int(pend)
    raise ValueError(f"No valid data lines in {path}")


def format_with_err(value, err):
    """Format value ± err as 0.00234 (32), matching last digits of the value."""
    if value is None or err is None or np.isnan(value) or np.isnan(err):
        return "--"
    if err == 0:
        return f"{value:.5f}"

    exp = int(np.floor(np.log10(err)))
    digits = max(0, -exp + 1)
    value_fmt = f"{{:.{digits+1}f}}"
    value_str = value_fmt.format(value)
    err_rounded = int(round(err * 10**(digits+1)))
    err_str = str(err_rounded).rjust(2, "0")[-2:]
    return f"{value_str}({err_str})"


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX longtable of ensemble parameters with residual masses (includes reduced χ²)."
    )
    parser.add_argument("--ensembles_csv", required=True, help="Path to ensembles.csv")
    parser.add_argument("--output_file", required=True, help="Output LaTeX file")
    args = parser.parse_args()

    df = pd.read_csv(args.ensembles_csv, sep="\t|,", engine="python")

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
            mres, mres_err, chi2_red, pstart, pend = read_mres_file(path)
        except Exception as e:
            print(f"Warning: could not read m_res file for {row['name']}: {e}")
            mres = mres_err = chi2_red = pstart = pend = np.nan

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
            "mres_chi2_red": chi2_red,
            "mres_p_start": pstart,
            "mres_p_end": pend
        })

    df_out = pd.DataFrame(rows)
    df_out["am_res"] = df_out.apply(lambda x: format_with_err(x["mres"], x["mres_err"]), axis=1)

    # ====== WRITE NEW LONGTABLE STRUCTURE ======
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    header_line = (
        "Ensemble & $\\beta$ & $am_0$ & $N_t$ & $N_s$ & $L_s$ & "
        "$\\alpha$ & $a_5/a$ & $am_5$ & $am_{\\rm PV}$ & "
        "$am_{\\rm res}$ & $\\tilde{t}^{am_{\\rm res}}_{\\rm start}$ & "
        "$\\tilde{t}^{am_{\\rm res}}_{\\rm end}$ & $\\chi^2_\\text{red}$ \\\\\n"
    )

    with open(args.output_file, "w") as f:

        f.write("\\color{red}\n")
        f.write("\\begin{longtable}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n")

        # Caption + label
        f.write(
            "\\caption{Characterisation of the ensembles generated to study the dynamics of Sp(4) with "
            "$N_{\\rm f} = 2$ Dirac fermions transforming in the fundamental representation, realised with "
            "the MDWF formalism. For each ensemble, we tabulate here the lattice coupling, $\\beta$, the number of "
            "sites in the lattice directions, $L_s$, $N_t$, and $N_s$, the bare mass $am_0$, the M\"obius action "
            "parameters $\\alpha$, $a_5/a$, $am_5$ and $am_{\\rm PV}$, the residual mass $am_{\\rm res}$ extracted "
            "from a fit over the plateau region identified with $\\tilde{t}^{am_{\\rm res}}_{\\rm start}$ and "
            "$\\tilde{t}^{am_{\\rm res}}_{\\rm end}$, together with the reduced chi-square of the fit.}\n"
        )
        f.write("\\label{tab:scan_summary} \\\\\n\n")

        # FIRST PAGE HEADER
        f.write("% ================= FIRST PAGE HEADER =================\n")
        f.write("\\hline\n\\hline\n")
        f.write(header_line)
        f.write("\\hline\n")
        f.write("\\endfirsthead\n\n")

        # PAGE 2+ HEADER
        f.write("% ================ HEADER FOR PAGE 2+ =================\n")
        f.write("\\hline\n")
        f.write(header_line)
        f.write("\\hline\n")
        f.write("\\endhead\n\n")

        # FOOTERS
        f.write("% ================= FOOTER FOR INTERMEDIATE PAGES =================\n")
        f.write("\\hline\n")
        f.write("\\endfoot\n\n")

        f.write("% ================= FINAL FOOTER =================\n")
        f.write("\\hline\\hline\n")
        f.write("\\endlastfoot\n\n")

        # BODY
        f.write("% ===================== TABLE BODY =====================\n")

        for _, r in df_out.iterrows():

            chi2_str = f"{r['mres_chi2_red']:.3f}" if not np.isnan(r['mres_chi2_red']) else "--"
            pstart_str = f"{int(r['mres_p_start'])}" if np.isfinite(r['mres_p_start']) else "--"
            pend_str   = f"{int(r['mres_p_end'])}"   if np.isfinite(r['mres_p_end'])   else "--"

            f.write(
                f"{r['name']} & {r['beta']} & {r['mass']} & {r['Nt']} & {r['Ns']} & {r['Ls']} & "
                f"{r['alpha']} & {r['a5']} & {r['m5']} & {r['mpv']} & "
                f"{r['am_res']} & {pstart_str} & {pend_str} & {chi2_str} \\\\\n"
            )

        f.write("\\end{longtable}\n")

if __name__ == "__main__":
    main()
