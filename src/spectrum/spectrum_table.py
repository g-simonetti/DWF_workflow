#!/usr/bin/env python3
import argparse
import json
import os
import re

import numpy as np
import pandas as pd


# -----------------------------------------------------------
# Format value ± error -> "1.234(56)"
# -----------------------------------------------------------
def fmt_err(val, err):
    if pd.isna(val):
        return "nan"
    if err == 0 or pd.isna(err):
        return f"{val:.4e}"

    e = abs(err)
    if e == 0:
        return f"{val:.4e}"

    digits = int(np.floor(np.log10(e)))
    nd = max(0, -digits + 1)

    fmt = f"{{:.{nd}f}}"
    val_s = fmt.format(val)
    err_s = fmt.format(e)

    frac = err_s.split(".")[-1] if "." in err_s else ""
    sig = frac.lstrip("0") or "0"
    return f"{val_s}({sig})"


# -----------------------------------------------------------
# tau_Q formatter with fixed decimals -> "12.34(56)"
# -----------------------------------------------------------
def fmt_err_fixed2(val, err, n_decimals=2):
    if pd.isna(val):
        return "nan"
    if err == 0 or pd.isna(err):
        return f"{val:.{n_decimals}f}"

    scale = 10 ** n_decimals
    val_s = f"{val:.{n_decimals}f}"
    err_scaled = int(round(abs(err) * scale))
    err_digits = str(err_scaled).lstrip("0") or "0"
    return f"{val_s}({err_digits})"


# -----------------------------------------------------------
# sqrt error propagation
# -----------------------------------------------------------
def sqrt_with_error(w0_sq, w0_sq_err):
    if pd.isna(w0_sq) or w0_sq <= 0:
        return np.nan, np.nan
    w0 = np.sqrt(w0_sq)
    if pd.isna(w0_sq_err):
        return w0, np.nan
    err = abs(w0_sq_err) / (2.0 * w0)
    return w0, err


# -----------------------------------------------------------
# JSON helpers (IGNORE any "ok" fields)
# -----------------------------------------------------------
def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _num(x, default=np.nan):
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _extract_value_err(obj, key: str):
    """
    Supports:
      - {"key": {"mean":..., "sdev":...}}
      - {"key": {"value":..., "err":...}}
      - {"key": ..., "key_err": ...}
    """
    if not isinstance(obj, dict):
        return (np.nan, np.nan)

    if key in obj and isinstance(obj[key], dict):
        d = obj[key]
        if ("mean" in d) or ("sdev" in d):
            return (_num(d.get("mean")), _num(d.get("sdev")))
        if ("value" in d) or ("err" in d):
            return (_num(d.get("value")), _num(d.get("err")))

    if key in obj:
        return (_num(obj.get(key)), _num(obj.get(f"{key}_err")))

    return (np.nan, np.nan)


# -----------------------------------------------------------
# Read spectrum.json (IGNORE "ok")
# -----------------------------------------------------------
def read_spectrum_json(filename: str) -> pd.Series:
    j = _load_json(filename)
    base = j.get("results", j)

    am_ps, am_ps_err = _extract_value_err(base, "am_ps")
    am_v, am_v_err   = _extract_value_err(base, "am_v")
    af_ps, af_ps_err = _extract_value_err(base, "af_ps")
    Z_A, Z_A_err     = _extract_value_err(base, "Z_A")

    # support either explicit chi2_* keys or chi2_over_dof dict
    chi2_map = base.get("chi2_over_dof", {}) if isinstance(base.get("chi2_over_dof", {}), dict) else {}
    chi2_ps  = _num(base.get("chi2_ps",  chi2_map.get("ps")))
    chi2_v   = _num(base.get("chi2_v",   chi2_map.get("v")))
    chi2_fps = _num(base.get("chi2_fps", chi2_map.get("fps")))
    chi2_Z   = _num(base.get("chi2_Z",   chi2_map.get("Z")))

    return pd.Series(
        {
            "am_ps": am_ps,
            "am_ps_err": am_ps_err,
            "am_v": am_v,
            "am_v_err": am_v_err,
            "chi2_ps": chi2_ps,
            "chi2_v": chi2_v,
            "af_ps": af_ps,
            "af_ps_err": af_ps_err,
            "chi2_fps": chi2_fps,
            "Z_A": Z_A,
            "Z_A_err": Z_A_err,
            "chi2_Z": chi2_Z,
        }
    )


# -----------------------------------------------------------
# Read wflow_extract.json (IGNORE "ok")
# -----------------------------------------------------------
def read_wflow_json(filename: str) -> pd.Series:
    j = _load_json(filename)
    base = j.get("results", j)

    w0_sq, w0_sq_err = _extract_value_err(base, "w0_sq")
    Q, Q_err         = _extract_value_err(base, "Q")

    tau_q, tau_q_err = _extract_value_err(base, "tau_q")
    if pd.isna(tau_q):
        tau_q, tau_q_err = _extract_value_err(base, "tau_Q")

    return pd.Series(
        {
            "w0_sq": w0_sq,
            "w0_sq_err": w0_sq_err,
            "Q": Q,
            "Q_err": Q_err,
            "tau_q": tau_q,
            "tau_q_err": tau_q_err,
        }
    )


# -----------------------------------------------------------
# Parse parameters from the path (matches Snakefile wildcards)
# Example:
# intermediary_data/NF2/Nt32/Ns16/Ls8/B7.4/M0.06/mpv1.0/alpha1.75/a51.0/M51.8/spectrum/spectrum.json
# -----------------------------------------------------------
def parse_params_from_path(path: str) -> dict:
    p = path.replace("\\", "/")

    def grab(pattern, cast=str):
        m = re.search(pattern, p)
        return cast(m.group(1)) if m else None

    return {
        "NF":   grab(r"/NF(\d+)/", int),
        "Nt":   grab(r"/Nt(\d+)/", int),
        "Ns":   grab(r"/Ns(\d+)/", int),
        "Ls":   grab(r"/Ls(\d+)/", int),
        "beta": grab(r"/B([0-9.]+)/", float),
        "mass": grab(r"/M([0-9.]+)/mpv", float),
        "mpv":  grab(r"/mpv([0-9.]+)/", float),
        "alpha":grab(r"/alpha([0-9.]+)/", float),
        "a5":   grab(r"/a5([0-9.]+)/", float),
        "M5":   grab(r"/M5([0-9.]+)/", float),
    }


def lookup_metadata_row(meta: pd.DataFrame, pars: dict):
    df = meta.copy()

    # int columns
    for c in ["NF", "Nt", "Ns", "Ls"]:
        if c in df.columns and pars.get(c) is not None:
            df = df[df[c].astype(int) == int(pars[c])]

    # float-ish columns (tolerant matching)
    for c, tol in [
        ("beta", 1e-6),
        ("mass", 1e-12),
        ("mpv", 1e-6),
        ("alpha", 1e-6),
        ("a5", 1e-6),
        ("M5", 1e-6),
    ]:
        if c in df.columns and pars.get(c) is not None:
            df = df[np.isclose(df[c].astype(float), float(pars[c]), rtol=0, atol=tol)]

    if len(df) == 0:
        return None

    # prefer use_in_spectrum True if present
    if "use_in_spectrum" in df.columns:
        df2 = df[df["use_in_spectrum"] == True]
        if len(df2) > 0:
            df = df2

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

    if len(args.spectrum) != len(args.wflow):
        raise ValueError("Number of --spectrum and --wflow files must match.")

    meta = pd.read_csv(args.metadata_csv)

    rows = []
    for spec_path, wf_path in zip(args.spectrum, args.wflow):
        spec = read_spectrum_json(spec_path)
        wf   = read_wflow_json(wf_path)

        pars = parse_params_from_path(spec_path)
        meta_row = lookup_metadata_row(meta, pars)

        if meta_row is not None and "name" in meta_row.index:
            name = str(meta_row["name"])
        else:
            # fallback if not found
            name = os.path.basename(os.path.dirname(os.path.dirname(spec_path)))

        tau_q_str = fmt_err_fixed2(wf["tau_q"], wf["tau_q_err"], n_decimals=2)

        w0, w0_err = sqrt_with_error(wf["w0_sq"], wf["w0_sq_err"])
        w0_str = fmt_err(w0, w0_err)

        f_ps_str = fmt_err(spec["af_ps"], spec["af_ps_err"])
        Z_A_str  = fmt_err(spec["Z_A"],  spec["Z_A_err"])

        chi2_Z_str  = "nan" if pd.isna(spec["chi2_Z"])  else f"{spec['chi2_Z']:.2f}"
        chi2_ps_str = "nan" if pd.isna(spec["chi2_ps"]) else f"{spec['chi2_ps']:.2f}"
        chi2_v_str  = "nan" if pd.isna(spec["chi2_v"])  else f"{spec['chi2_v']:.2f}"

        rows.append([
            name,
            fmt_err(spec["am_ps"], spec["am_ps_err"]),
            chi2_ps_str,
            fmt_err(spec["am_v"], spec["am_v_err"]),
            chi2_v_str,
            f_ps_str,
            Z_A_str,
            chi2_Z_str,
            w0_str,
            fmt_err(wf["Q"], wf["Q_err"]),
            tau_q_str,
        ])

    headers = [
        "name",
        "$am_\\pi$", "$\\chi^2_\\pi$",
        "$am_{\\mathrm{V}}$", "$\\chi^2_{\\mathrm{V}}$",
        "$af_{\\pi}$",
        "$Z_A$",
        "$\\chi^2_{Z_A}$",
        "$w_0/a$",
        "$Q$", "$\\tau_Q$",
    ]

    tex = []
    tex.append("\\begin{tabular}{c c c c c c c c c c c}")
    tex.append("\\hline")
    tex.append(" & ".join(headers) + " \\\\")
    tex.append("\\hline")
    for r in rows:
        tex.append(" & ".join(map(str, r)) + " \\\\")
    tex.append("\\hline")
    tex.append("\\end{tabular}")

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_file, "w") as f:
        f.write("\n".join(tex))

    print(f"✓ Wrote LaTeX table → {args.output_file}")


if __name__ == "__main__":
    main()