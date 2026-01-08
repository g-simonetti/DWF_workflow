#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------
# Format value ± error -> "1.234(56)"
# -----------------------------------------------------------
def fmt_err(val, err):
    if err == 0 or np.isnan(err):
        return f"{val:.4e}"

    e = abs(err)
    digits = int(np.floor(np.log10(e)))
    nd = max(0, -digits + 1)

    fmt = f"{{:.{nd}f}}"
    val_s = fmt.format(val)
    err_s = fmt.format(e)

    frac = err_s.split(".")[-1]
    sig = frac.lstrip("0")
    if sig == "":
        sig = "0"

    return f"{val_s}({sig})"


# -----------------------------------------------------------
# tau_Q formatter
# -----------------------------------------------------------
def fmt_err_fixed2(val, err, n_decimals=2):
    if err == 0 or np.isnan(err):
        return f"{val:.{n_decimals}f}"

    scale = 10 ** n_decimals
    val_s = f"{val:.{n_decimals}f}"
    err_scaled = int(round(abs(err) * scale))

    err_digits = str(err_scaled).lstrip("0")
    if err_digits == "":
        err_digits = "0"

    return f"{val_s}({err_digits})"


# -----------------------------------------------------------
# sqrt error propagation
# -----------------------------------------------------------
def sqrt_with_error(w0_sq, w0_sq_err):
    if w0_sq <= 0 or np.isnan(w0_sq):
        return np.nan, np.nan
    w0 = np.sqrt(w0_sq)
    if np.isnan(w0_sq_err):
        return w0, np.nan
    err = abs(w0_sq_err) / (2 * w0)
    return w0, err


# -----------------------------------------------------------
# Read spectrum results (updated format)
# -----------------------------------------------------------
def read_spectrum_file(filename):
    cols = [
        "am_ps", "am_ps_err",
        "am_v", "am_v_err",
        "chi2_ps", "chi2_v",
        "af_ps", "af_ps_err",
        "chi2_fps",     # still read but NOT printed
        "Z_A", "Z_A_err",
    ]
    df = pd.read_csv(filename, sep=r"\s+", comment="#", header=None)
    df.columns = cols
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

        w0, w0_err = sqrt_with_error(wf["w0_sq"], wf["w0_sq_err"])
        w0_str = fmt_err(w0, w0_err)

        # f_PS and ZA (NEW)
        f_ps_str = fmt_err(spec["af_ps"], spec["af_ps_err"])
        Z_A_str  = fmt_err(spec["Z_A"],  spec["Z_A_err"])

        rows.append([
            meta_row.get("name", f"ens{idx}"),

            fmt_err(spec["am_ps"], spec["am_ps_err"]),
            f"{spec['chi2_ps']:.2f}",

            fmt_err(spec["am_v"], spec["am_v_err"]),
            f"{spec['chi2_v']:.2f}",

            f_ps_str,        # printed
            Z_A_str,         # printed

            w0_str,
            fmt_err(wf["Q"], wf["Q_err"]),
            tau_q_str,
        ])

    # -------------------------------------------------------
    # Updated LaTeX headers 
    # -------------------------------------------------------
    headers = [
        "name",
        "$am_\\pi$", "$\\chi^2_\\pi$",
        "$am_{\\mathrm{V}}$", "$\\chi^2_{\\mathrm{V}}$",
        "$af_{\\pi}$",
        "$Z_A$",
        "$w_0/a$",
        "$Q$", "$\\tau_Q$",
    ]

    tex = []
    tex.append("\\begin{tabular}{c c c c c c c c c c}")
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
