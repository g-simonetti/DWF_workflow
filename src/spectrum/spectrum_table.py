#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------
# Format value ± error -> "1.234(56)"
# Generic formatter (used for all columns except tau_Q)
# -----------------------------------------------------------
def fmt_err(val, err):
    if err == 0 or np.isnan(err):
        return f"{val:.4e}"

    e = abs(err)

    # Order of magnitude of the error
    digits = int(np.floor(np.log10(e)))
    nd = max(0, -digits + 1)   # number of decimal places

    fmt = f"{{:.{nd}f}}"
    val_s = fmt.format(val)
    err_s = fmt.format(e)

    # Take fractional part of the error and strip *leading* zeros:
    #   0.0049 -> "0049" -> "49"
    frac = err_s.split(".")[-1]
    sig = frac.lstrip("0")
    if sig == "":
        sig = "0"

    return f"{val_s}({sig})"


# -----------------------------------------------------------
# Special formatter for tau_int^Q:
# - central value always with 2 decimals
# - error in units of last 2 decimals
#   Example: val=2.21, err=0.05  -> "2.21(5)"  (±0.05)
#            val=5.12, err=0.46  -> "5.12(46)" (±0.46)
# -----------------------------------------------------------
def fmt_err_fixed2(val, err, n_decimals=2):
    if err == 0 or np.isnan(err):
        return f"{val:.{n_decimals}f}"

    scale = 10 ** n_decimals

    val_s = f"{val:.{n_decimals}f}"
    err_scaled = int(round(abs(err) * scale))  # error in last-digit units
    err_digits = str(err_scaled).lstrip("0")
    if err_digits == "":
        err_digits = "0"

    return f"{val_s}({err_digits})"


# -----------------------------------------------------------
# Read spectrum results
# -----------------------------------------------------------
def read_spectrum_file(filename):
    df = pd.read_csv(filename, sep=r"\s+", comment="#", header=None)
    df.columns = [
        "am_ps", "am_ps_err",
        "am_v", "am_v_err",
        "chi2_ps", "chi2_v",
        "t1_ps", "t2_ps",
        "t1_v", "t2_v",
    ]
    return df.iloc[0]

# -----------------------------------------------------------
# Read Wilson flow summary
# -----------------------------------------------------------
def read_wflow_summary(filename):
    df = pd.read_csv(filename, sep=r"\s+", comment="#", header=None)
    df.columns = ["w0_sq", "w0_sq_err", "Q", "Q_err", "tau_q", "tau_q_err"]
    return df.iloc[0]

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum", nargs="+", required=True)
    parser.add_argument("--wflow", nargs="+", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata_csv)
    rows = []

    for idx, meta_row in meta.iterrows():
        spec = read_spectrum_file(args.spectrum[idx])
        wf   = read_wflow_summary(args.wflow[idx])

        tau_q_str = fmt_err_fixed2(wf["tau_q"], wf["tau_q_err"], n_decimals=2)

        rows.append([
            meta_row.get("name", f"ens{idx}"),
            int(spec["t1_ps"]), int(spec["t2_ps"]),
            fmt_err(spec["am_ps"], spec["am_ps_err"]),
            f"{spec['chi2_ps']:.2f}",
            int(spec["t1_v"]), int(spec["t2_v"]),
            fmt_err(spec["am_v"], spec["am_v_err"]),
            f"{spec['chi2_v']:.2f}",
            fmt_err(wf["w0_sq"], wf["w0_sq_err"]),
            fmt_err(wf["Q"], wf["Q_err"]),
            tau_q_str,
        ])

    # -------------------------------------------------------
    # Manual LaTeX table construction (SAFE!)
    # -------------------------------------------------------
    headers = [
        "name",
        "$t_{1}^{\\pi}$", "$t_{2}^{\\pi}$", "$am_\\pi$", "$\\chi^2_\\pi$",
        "$t_{1}^{\\mathrm{V}}$", "$t_{2}^{\\mathrm{V}}$", "$am_{\\mathrm{V}}$", "$\\chi^2_{\\mathrm{V}}$",
        "$w_0^2$", "$Q$", "$\\tau_Q$",
    ]

    tex = []
    tex.append("\\begin{tabular}{c c c c c c c c c c c c}")
    tex.append("\\hline")
    tex.append(" & ".join(headers) + " \\\\")
    tex.append("\\hline")

    for r in rows:
        tex.append(" & ".join(map(str, r)) + " \\\\")

    tex.append("\\hline")
    tex.append("\\end{tabular}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write("\n".join(tex))

    print(f"✓ Wrote LaTeX table → {args.output_file}")

if __name__ == "__main__":
    main()
