#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from shared_continuum_models import solve_weighted_least_squares

plt.style.use("tableau-colorblind10")


# ============================================================
# MANUAL PLOT SELECTION
# ------------------------------------------------------------
# Keep ALL fits computed and saved to JSON.
# Only the fits listed here will be shown on the plot.
#
# Available keys:
#   "dw"
#   "dw2"
#   "wilson_1"
#   "wilson_2"
#   "wilson_3"
#   "wilson_4"
#   "wilson_5"
# ============================================================
PLOT_FITS = [
    # "wilson_1",
    # "wilson_2",
    # "wilson_3",
     "wilson_4",
    # "wilson_5",
    # "dw",
     "dw2",
]


# ============================================================
# Generic readers
# ============================================================
def read_json_file(filename):
    """Read JSON file and return parsed object."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Could not read JSON file: {filename}\n{e}") from e


def parse_pair(obj, key, filename):
    """
    Parse a [value, error] pair from obj[key].

    Accepted forms:
      - [value, error]
      - {"mean": ..., "sdev": ...}
    """
    if key not in obj:
        raise ValueError(f"Missing key '{key}' in '{filename}'")

    val = obj[key]

    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(val[0]), float(val[1])

    if isinstance(val, dict) and "mean" in val and "sdev" in val:
        return float(val["mean"]), float(val["sdev"])

    raise ValueError(
        f"Key '{key}' in '{filename}' must be either [value, error] "
        f"or {{'mean': ..., 'sdev': ...}}, got: {val}"
    )


# ============================================================
# Domain-wall readers
# ============================================================
def read_spectrum_json(filename):
    """
    Read DW spectrum JSON and extract:
      - results.standard_fit.PP.am_ps.mean/sdev
      - results.standard_fit.Z_A.Z_A.mean/sdev
      - results.standard_fit.simultaneous_PP_A0P.af_ps.mean/sdev

    The decay-constant quantity used for plotting is:
      fps = Z_A * af_ps
    """
    data = read_json_file(filename)

    try:
        standard_fit = data["results"]["standard_fit"]

        pp = standard_fit["PP"]["am_ps"]
        za = standard_fit["Z_A"]["Z_A"]
        afps = standard_fit["simultaneous_PP_A0P"]["af_ps"]

        am_ps = float(pp["mean"])
        am_ps_err = float(pp["sdev"])

        z_a = float(za["mean"])
        z_a_err = float(za["sdev"])

        af_ps = float(afps["mean"])
        af_ps_err = float(afps["sdev"])
    except KeyError as e:
        raise ValueError(
            f"Missing expected key in spectrum JSON '{filename}': {e}\n"
            "Expected results.standard_fit.PP.am_ps.mean/sdev, "
            "results.standard_fit.Z_A.Z_A.mean/sdev, and "
            "results.standard_fit.simultaneous_PP_A0P.af_ps.mean/sdev"
        ) from e
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid numeric content in spectrum JSON '{filename}': {e}"
        ) from e

    fps = z_a * af_ps
    fps_err = np.sqrt((af_ps * z_a_err) ** 2 + (z_a * af_ps_err) ** 2)

    return {
        "am_ps": am_ps,
        "am_ps_err": am_ps_err,
        "fps": fps,
        "fps_err": fps_err,
    }


def read_wflow_json(filename):
    """
    Read DW wflow JSON and extract:
      - summary.w0_sq / summary.w0_sq_err
        or
      - summary.w0 / summary.w0_err (converted to squared form)
    """
    data = read_json_file(filename)

    try:
        summary = data["summary"]

        if "w0_sq" in summary and "w0_sq_err" in summary:
            w0_sq = float(summary["w0_sq"])
            w0_sq_err = float(summary["w0_sq_err"])
            return w0_sq, w0_sq_err

        if "w0" in summary and "w0_err" in summary:
            w0 = float(summary["w0"])
            w0_sq = w0 * w0

            # Prefer the bootstrap samples when present so the squared error
            # matches the current wflow output format more closely.
            w0_samples = data.get("bootstrap", {}).get("w0", {}).get("samples")
            if isinstance(w0_samples, list) and w0_samples:
                parsed_samples = []
                for sample in w0_samples:
                    if isinstance(sample, dict):
                        value = sample.get("w0")
                    else:
                        value = sample

                    if value is None:
                        parsed_samples.append(np.nan)
                    else:
                        parsed_samples.append(float(value))

                w0_sq_boot = np.square(np.asarray(parsed_samples, dtype=float))
                w0_sq_err = float(np.nanstd(w0_sq_boot, ddof=1))
            else:
                w0_err = float(summary["w0_err"])
                w0_sq_err = 2.0 * abs(w0) * w0_err

            return w0_sq, w0_sq_err

        raise KeyError("w0_sq/w0_sq_err or w0/w0_err")
    except KeyError as e:
        raise ValueError(
            f"Missing expected key in wflow JSON '{filename}': {e}\n"
            "Expected either summary.w0_sq and summary.w0_sq_err, or "
            "summary.w0 and summary.w0_err"
        ) from e
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid numeric content in wflow JSON '{filename}': {e}"
        ) from e


# ============================================================
# Wilson readers
# ============================================================
def read_wilson_spectrum_json(filename):
    """
    Read Wilson spectrum table JSON.

    Expected format: a list of dicts, one per ensemble, with keys:
      - Ensemble
      - amps : [value, error] or {"mean": ..., "sdev": ...}
      - afps : [value, error] or {"mean": ..., "sdev": ...}
      - beta
      - m0
    """
    data = read_json_file(filename)

    if not isinstance(data, list):
        raise ValueError(
            f"Wilson spectrum JSON '{filename}' must contain a list of ensembles."
        )

    out = {}
    for row in data:
        if "Ensemble" not in row:
            raise ValueError(f"Missing 'Ensemble' key in '{filename}'")

        ens = row["Ensemble"]
        am_ps, am_ps_err = parse_pair(row, "amps", filename)
        af_ps, af_ps_err = parse_pair(row, "afps", filename)

        out[ens] = {
            "beta": float(row["beta"]),
            "m0": float(row["m0"]),
            "am_ps": am_ps,
            "am_ps_err": am_ps_err,
            "af_ps": af_ps,
            "af_ps_err": af_ps_err,
        }

    return out


def read_wilson_wflow_json(filename):
    """
    Read Wilson ensemble/wflow table JSON.

    Expected format: a list of dicts, one per ensemble, with keys:
      - Ensemble
      - w0a : [value, error] or {"mean": ..., "sdev": ...}
      - beta
      - m0
    """
    data = read_json_file(filename)

    if not isinstance(data, list):
        raise ValueError(
            f"Wilson wflow JSON '{filename}' must contain a list of ensembles."
        )

    out = {}
    for row in data:
        if "Ensemble" not in row:
            raise ValueError(f"Missing 'Ensemble' key in '{filename}'")

        ens = row["Ensemble"]
        w0a, w0a_err = parse_pair(row, "w0a", filename)

        out[ens] = {
            "beta": float(row["beta"]),
            "m0": float(row["m0"]),
            "w0a": w0a,
            "w0a_err": w0a_err,
        }

    return out


# ============================================================
# Extract beta and mass from directory structure
# ============================================================
def extract_beta_mass_from_path(path):
    """
    Extract beta and mass from path components like:
      .../B7.6/M0.06/...
    """
    parts = Path(path).parts
    beta = None
    mass = None

    for p in parts:
        if p.startswith("B"):
            try:
                beta = float(p[1:])
            except ValueError:
                pass
        elif p.startswith("M") and mass is None:
            try:
                mass = float(p[1:])
            except ValueError:
                pass

    return beta, mass


# ============================================================
# Error propagation
# ============================================================
def mw0_sq_and_error_from_w0sq(am, am_err, w0_sq, w0_sq_err):
    """
    x = (am)^2 * w0_sq = (m w0)^2
    """
    x = (am**2) * w0_sq
    dx_dam = 2.0 * am * w0_sq
    dx_dw0sq = am**2
    var = (dx_dam**2) * (am_err**2) + (dx_dw0sq**2) * (w0_sq_err**2)
    return x, np.sqrt(var)


def fw0_sq_and_error_from_w0sq(af, af_err, w0_sq, w0_sq_err):
    """
    y = (af)^2 * w0_sq = (f w0)^2
    """
    y = (af**2) * w0_sq
    dy_daf = 2.0 * af * w0_sq
    dy_dw0sq = af**2
    var = (dy_daf**2) * (af_err**2) + (dy_dw0sq**2) * (w0_sq_err**2)
    return y, np.sqrt(var)


def mw0_sq_and_error_from_w0a(am, am_err, w0a, w0a_err):
    """
    x = (am * w0a)^2 = (m w0)^2
    """
    x = (am * w0a) ** 2
    dx_dam = 2.0 * am * (w0a**2)
    dx_dw0a = 2.0 * (am**2) * w0a
    var = (dx_dam**2) * (am_err**2) + (dx_dw0a**2) * (w0a_err**2)
    return x, np.sqrt(var)


def fw0_sq_and_error_from_w0a(af, af_err, w0a, w0a_err):
    """
    y = (af * w0a)^2 = (f w0)^2
    """
    y = (af * w0a) ** 2
    dy_daf = 2.0 * af * (w0a**2)
    dy_dw0a = 2.0 * (af**2) * w0a
    var = (dy_daf**2) * (af_err**2) + (dy_dw0a**2) * (w0a_err**2)
    return y, np.sqrt(var)


def a_over_w0_and_error(w0a, w0a_err):
    """
    z = a / w0 = 1 / (w0 / a)
    """
    z = 1.0 / w0a
    z_err = w0a_err / (w0a**2)
    return z, z_err


def a_over_w0_sq_and_error_from_w0sq(w0_sq, w0_sq_err):
    """
    z = (a / w0)^2 = 1 / w0_sq
    """
    z = 1.0 / w0_sq
    z_err = w0_sq_err / (w0_sq**2)
    return z, z_err


def square_with_error(z, z_err):
    """
    q = z^2
    """
    q = z**2
    q_err = 2.0 * abs(z) * z_err
    return q, q_err


# ============================================================
# Point filtering
# ============================================================
def exclude_wilson_endpoints(points, n_first=0, m_last=0, x_key="x"):
    """
    Exclude the first n points (smallest x) and the last m points (largest x).

    Returns
    -------
    kept_points, removed_first, removed_last
    """
    if n_first < 0 or m_last < 0:
        raise ValueError("n_first and m_last must be non-negative.")

    if n_first + m_last >= len(points) and len(points) > 0:
        raise ValueError(
            f"Cannot exclude first {n_first} and last {m_last} points: "
            f"only {len(points)} Wilson points available."
        )

    sorted_points = sorted(points, key=lambda p: p[x_key])

    removed_first = sorted_points[:n_first] if n_first > 0 else []
    removed_last = sorted_points[-m_last:] if m_last > 0 else []

    kept_start = n_first
    kept_stop = len(sorted_points) - m_last if m_last > 0 else len(sorted_points)
    kept = sorted_points[kept_start:kept_stop]

    return kept, removed_first, removed_last


def print_removed_wilson_points(removed_points, which):
    if not removed_points:
        return

    print(f"Excluded Wilson points from the {which} in x:")
    for p in removed_points:
        print(
            f"  Ensemble={p['Ensemble']}, beta={p['beta']}, m0={p['m0']}, "
            f"x=(m_PS w0)^2={p['x']:.8g}"
        )


# ============================================================
# DWF/MDWF continuum fits
# ============================================================
def fit_dw_continuum(points):
    """
    Fit DWF/MDWF points to:
        y = A + B*x + C*(a/w0)^2
    """
    n_params = 3

    if len(points) < n_params:
        raise ValueError(
            f"Need at least {n_params} DWF/MDWF points for continuum fit."
        )

    x = np.array([p["x"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    ye = np.array([p["yerr"] for p in points], dtype=float)
    z = np.array([p["a_over_w0_sq"] for p in points], dtype=float)

    M = np.column_stack([np.ones_like(x), x, z])

    coeffs, errs, cov, chi2, dof = solve_weighted_least_squares(
        M, y, ye, "DWF/MDWF"
    )

    A, B, C = coeffs
    A_err, B_err, C_err = errs

    if A <= 0:
        L = np.nan
        L_err = np.nan
    else:
        L = B / A
        dL_dA = -B / (A**2)
        dL_dB = 1.0 / A
        var_L = (
            dL_dA**2 * cov[0, 0]
            + dL_dB**2 * cov[1, 1]
            + 2.0 * dL_dA * dL_dB * cov[0, 1]
        )
        L_err = np.sqrt(max(var_L, 0.0))

    return {
        "A": A,
        "A_err": A_err,
        "B": B,
        "B_err": B_err,
        "C": C,
        "C_err": C_err,
        "cov": cov,
        "chi2": chi2,
        "dof": dof,
        "L": L,
        "L_err": L_err,
        "model_key": "dw",
        "label": r"MDWF: $A + Bx + C(a/w_0)^2$",
        "label_plain": "MDWF: A + Bx + C(a/w0)^2",
    }


def fit_dw2_continuum(points):
    """
    Fit DWF/MDWF points to:
        y = A + B*x + C*x^2 + D*(a/w0)^2
    """
    n_params = 4

    if len(points) < n_params:
        raise ValueError(
            f"Need at least {n_params} DWF/MDWF points for continuum fit with x^2 term."
        )

    x = np.array([p["x"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    ye = np.array([p["yerr"] for p in points], dtype=float)
    z = np.array([p["a_over_w0_sq"] for p in points], dtype=float)

    M = np.column_stack([np.ones_like(x), x, x**2, z])

    coeffs, errs, cov, chi2, dof = solve_weighted_least_squares(
        M, y, ye, "DWF/MDWF (with x^2)"
    )

    A, B, C, D = coeffs
    A_err, B_err, C_err, D_err = errs

    if A <= 0:
        L = np.nan
        L_err = np.nan
    else:
        L = B / A
        dL_dA = -B / (A**2)
        dL_dB = 1.0 / A
        var_L = (
            dL_dA**2 * cov[0, 0]
            + dL_dB**2 * cov[1, 1]
            + 2.0 * dL_dA * dL_dB * cov[0, 1]
        )
        L_err = np.sqrt(max(var_L, 0.0))

    return {
        "A": A,
        "A_err": A_err,
        "B": B,
        "B_err": B_err,
        "C": C,
        "C_err": C_err,
        "D": D,
        "D_err": D_err,
        "cov": cov,
        "chi2": chi2,
        "dof": dof,
        "L": L,
        "L_err": L_err,
        "model_key": "dw2",
        "label": r"MDWF: $A + Bx + Cx^2 + D(a/w_0)^2$",
        "label_plain": "MDWF: A + Bx + Cx^2 + D(a/w0)^2",
    }


def continuum_line_and_band_dw(x, fit):
    """
    Continuum line and 1-sigma band for:
        y_cont(x) = A + B x
    """
    A = fit["A"]
    B = fit["B"]
    cov = fit["cov"]

    y = A + B * x
    var = cov[0, 0] + x**2 * cov[1, 1] + 2.0 * x * cov[0, 1]
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)

    return y, err


def continuum_line_and_band_dw2(x, fit):
    """
    Continuum line and 1-sigma band for:
        y_cont(x) = A + B x + C x^2
    """
    A = fit["A"]
    B = fit["B"]
    C = fit["C"]
    cov = fit["cov"]

    y = A + B * x + C * x**2

    M_cont = np.column_stack([np.ones_like(x), x, x**2])
    cov_cont = cov[:3, :3]
    var = np.einsum("ij,jk,ik->i", M_cont, cov_cont, M_cont)
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)

    return y, err


# ============================================================
# Wilson generic model fits
# ============================================================
def wilson_term_vector(points, term):
    x = np.array([p["x"] for p in points], dtype=float)

    if term == "1":
        return np.ones_like(x)

    if term == "x":
        return x

    if term == "x2":
        return x**2

    if term == "a_over_w0":
        return np.array([p["a_over_w0"] for p in points], dtype=float)

    if term == "a_over_w0_sq":
        return np.array([p["a_over_w0_sq"] for p in points], dtype=float)

    if term == "x_a_over_w0":
        return x * np.array([p["a_over_w0"] for p in points], dtype=float)

    raise ValueError(f"Unsupported Wilson basis term: {term}")


def wilson_design_matrix(points, basis_terms):
    cols = [wilson_term_vector(points, term) for term in basis_terms]
    return np.column_stack(cols)


def fit_wilson_model(points, basis_terms, label, model_key, label_plain):
    """
    Generic weighted least-squares fit for Wilson data.

    Model:
        y = sum_i c_i * f_i(x, a/w0)

    Weighted only by y-errors.
    """
    n_params = len(basis_terms)

    if len(points) < n_params:
        raise ValueError(
            f"Need at least {n_params} Wilson points for fit '{label}', "
            f"but got only {len(points)}."
        )

    y = np.array([p["y"] for p in points], dtype=float)
    ye = np.array([p["yerr"] for p in points], dtype=float)
    M = wilson_design_matrix(points, basis_terms)

    coeffs, errs, cov, chi2, dof = solve_weighted_least_squares(
        M, y, ye, f"Wilson fit '{label_plain}'"
    )

    return {
        "basis_terms": basis_terms,
        "coeffs": coeffs,
        "coeff_errs": errs,
        "cov": cov,
        "chi2": chi2,
        "dof": dof,
        "label": label,
        "label_plain": label_plain,
        "model_key": model_key,
    }


def fit_wilson_x_plus_a_over_w0(points):
    return fit_wilson_model(
        points,
        basis_terms=["1", "x", "a_over_w0"],
        label=r"Wilson: $A + Bx + C(a/w_0)$",
        model_key="wilson_1",
        label_plain="Wilson: A + Bx + C(a/w0)",
    )


def fit_wilson_x_plus_a_over_w0_plus_sq(points):
    return fit_wilson_model(
        points,
        basis_terms=["1", "x", "a_over_w0", "a_over_w0_sq"],
        label=r"Wilson: $A + Bx + C(a/w_0) + D(a/w_0)^2$",
        model_key="wilson_2",
        label_plain="Wilson: A + Bx + C(a/w0) + D(a/w0)^2",
    )


def fit_wilson_x_plus_xaow0_plus_aow0_plus_sq(points):
    return fit_wilson_model(
        points,
        basis_terms=["1", "x", "x_a_over_w0", "a_over_w0", "a_over_w0_sq"],
        label=r"Wilson: $A + Bx + Cx(a/w_0) + D(a/w_0) + E(a/w_0)^2$",
        model_key="wilson_3",
        label_plain="Wilson: A + Bx + Cx(a/w0) + D(a/w0) + E(a/w0)^2",
    )


def fit_wilson_x_plus_x2_plus_xaow0_plus_aow0_plus_sq(points):
    return fit_wilson_model(
        points,
        basis_terms=["1", "x", "x2", "x_a_over_w0", "a_over_w0", "a_over_w0_sq"],
        label=r"Wilson: $A + Bx + Cx^2 + Dx(a/w_0) + E(a/w_0) + F(a/w_0)^2$",
        model_key="wilson_4",
        label_plain="Wilson: A + Bx + Cx^2 + Dx(a/w0) + E(a/w0) + F(a/w0)^2",
    )


def fit_wilson_x_plus_x2_plus_aow0_plus_sq(points):
    return fit_wilson_model(
        points,
        basis_terms=["1", "x", "x2", "a_over_w0", "a_over_w0_sq"],
        label=r"Wilson: $A + Bx + Cx^2 + D(a/w_0) + E(a/w_0)^2$",
        model_key="wilson_5",
        label_plain="Wilson: A + Bx + Cx^2 + D(a/w0) + E(a/w0)^2",
    )


def wilson_continuum_line_and_band(x_grid, fit):
    """
    Continuum line and 1-sigma band for Wilson models.

    In the continuum limit:
        a/w0 -> 0
    """
    basis_terms = fit["basis_terms"]
    coeffs = fit["coeffs"]
    cov = fit["cov"]

    survivors = []
    survivor_indices = []

    for i, term in enumerate(basis_terms):
        if term in {"1", "x", "x2"}:
            survivor_indices.append(i)

            if term == "1":
                survivors.append(np.ones_like(x_grid))
            elif term == "x":
                survivors.append(x_grid)
            elif term == "x2":
                survivors.append(x_grid**2)

    if not survivors:
        raise ValueError("Wilson continuum model has no surviving continuum terms.")

    M_cont = np.column_stack(survivors)
    c_cont = coeffs[survivor_indices]
    cov_cont = cov[np.ix_(survivor_indices, survivor_indices)]

    y = M_cont @ c_cont
    var = np.einsum("ij,jk,ik->i", M_cont, cov_cont, M_cont)
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)

    return y, err


# ============================================================
# Formatting helpers
# ============================================================
def print_dw_fit_summary(fit):
    print("DWF/MDWF continuum fit:")
    print(f"  A = {fit['A']:.8g} ± {fit['A_err']:.3g}")
    print(f"  B = {fit['B']:.8g} ± {fit['B_err']:.3g}")
    print(f"  C = {fit['C']:.8g} ± {fit['C_err']:.3g}")
    print(f"  L = {fit['L']:.8g} ± {fit['L_err']:.3g}")
    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


def print_dw2_fit_summary(fit):
    print("DWF/MDWF continuum fit [with x^2]:")
    print(f"  A = {fit['A']:.8g} ± {fit['A_err']:.3g}")
    print(f"  B = {fit['B']:.8g} ± {fit['B_err']:.3g}")
    print(f"  C = {fit['C']:.8g} ± {fit['C_err']:.3g}")
    print(f"  D = {fit['D']:.8g} ± {fit['D_err']:.3g}")
    print(f"  L = {fit['L']:.8g} ± {fit['L_err']:.3g}")
    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


def print_generic_wilson_fit_summary(title, fit):
    print(f"{title}:")
    for i, (term, coeff, err) in enumerate(
        zip(fit["basis_terms"], fit["coeffs"], fit["coeff_errs"])
    ):
        name = chr(ord("A") + i)
        print(f"  {name} = {coeff:.8g} ± {err:.3g}   [{term}]")
    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


# ============================================================
# Data collection
# ============================================================
def collect_dw_points(spectrum_files, wflow_files):
    if len(spectrum_files) != len(wflow_files):
        raise ValueError("Number of --spectrum files must equal number of --wflow files.")

    dw_points = []

    for spec_path, wflow_path in zip(spectrum_files, wflow_files):
        beta, mass = extract_beta_mass_from_path(spec_path)
        if beta is None or mass is None:
            raise ValueError(f"Could not extract beta/mass from path: {spec_path}")

        spec = read_spectrum_json(spec_path)
        w0_sq, w0_sq_err = read_wflow_json(wflow_path)

        x, xerr = mw0_sq_and_error_from_w0sq(
            spec["am_ps"], spec["am_ps_err"], w0_sq, w0_sq_err
        )
        y, yerr = fw0_sq_and_error_from_w0sq(
            spec["fps"], spec["fps_err"], w0_sq, w0_sq_err
        )
        z, zerr = a_over_w0_sq_and_error_from_w0sq(w0_sq, w0_sq_err)

        dw_points.append(
            {
                "beta": beta,
                "m0": mass,
                "x": x,
                "xerr": xerr,
                "y": y,
                "yerr": yerr,
                "a_over_w0_sq": z,
                "a_over_w0_sq_err": zerr,
            }
        )

    return dw_points


def collect_wilson_points(spectrum_file, wflow_file):
    wilson_spec = read_wilson_spectrum_json(spectrum_file)
    wilson_wflow = read_wilson_wflow_json(wflow_file)

    common_ensembles = sorted(set(wilson_spec) & set(wilson_wflow))
    if not common_ensembles:
        raise ValueError(
            "No common ensembles found between Wilson spectrum and wflow JSON files."
        )

    wilson_points = []
    for ens in common_ensembles:
        srow = wilson_spec[ens]
        wrow = wilson_wflow[ens]

        if abs(srow["beta"] - wrow["beta"]) > 1e-12:
            raise ValueError(
                f"Beta mismatch for Wilson ensemble '{ens}': "
                f"{srow['beta']} vs {wrow['beta']}"
            )
        if abs(srow["m0"] - wrow["m0"]) > 1e-12:
            raise ValueError(
                f"m0 mismatch for Wilson ensemble '{ens}': "
                f"{srow['m0']} vs {wrow['m0']}"
            )

        x, xerr = mw0_sq_and_error_from_w0a(
            srow["am_ps"], srow["am_ps_err"], wrow["w0a"], wrow["w0a_err"]
        )
        y, yerr = fw0_sq_and_error_from_w0a(
            srow["af_ps"], srow["af_ps_err"], wrow["w0a"], wrow["w0a_err"]
        )

        a_over_w0, a_over_w0_err = a_over_w0_and_error(
            wrow["w0a"], wrow["w0a_err"]
        )
        a_over_w0_sq, a_over_w0_sq_err = square_with_error(
            a_over_w0, a_over_w0_err
        )

        wilson_points.append(
            {
                "Ensemble": ens,
                "beta": srow["beta"],
                "m0": srow["m0"],
                "x": x,
                "xerr": xerr,
                "y": y,
                "yerr": yerr,
                "a_over_w0": a_over_w0,
                "a_over_w0_err": a_over_w0_err,
                "a_over_w0_sq": a_over_w0_sq,
                "a_over_w0_sq_err": a_over_w0_sq_err,
            }
        )

    return wilson_points


# ============================================================
# Plot styling registry
# ============================================================
def get_plot_registry():
    return {
        "wilson_1": {
            "color": "black",
            "linestyle": "--",
            "alpha_band": 0.10,
        },
        "wilson_2": {
            "color": "dimgray",
            "linestyle": "-",
            "alpha_band": 0.10,
        },
        "wilson_3": {
            "color": "tab:blue",
            "linestyle": "-.",
            "alpha_band": 0.10,
        },
        "wilson_4": {
            "color": "tab:red",
            "linestyle": ":",
            "alpha_band": 0.10,
        },
        "wilson_5": {
            "color": "tab:orange",
            "linestyle": (0, (3, 1, 1, 1)),
            "alpha_band": 0.10,
        },
        "dw": {
            "color": "tab:green",
            "linestyle": (0, (5, 2, 1, 2)),
            "alpha_band": 0.12,
        },
        "dw2": {
            "color": "tab:purple",
            "linestyle": "-",
            "alpha_band": 0.12,
        },
    }


def validate_plot_fit_keys(plot_fit_keys, available_fit_keys):
    invalid = sorted(set(plot_fit_keys) - set(available_fit_keys))
    if invalid:
        raise ValueError(
            f"Unknown fit key(s) in PLOT_FITS: {invalid}\n"
            f"Allowed keys: {sorted(available_fit_keys)}"
        )


# ============================================================
# Plotting
# ============================================================
def plot_points_and_fits(
    dw_points,
    wilson_points,
    all_fits,
    plot_fit_keys,
    output_plot,
):
    plot_registry = get_plot_registry()
    validate_plot_fit_keys(plot_fit_keys, all_fits.keys())

    all_betas = sorted(
        {p["beta"] for p in dw_points}.union({p["beta"] for p in wilson_points})
    )
    beta_colors = {b: f"C{i % 10}" for i, b in enumerate(all_betas)}

    fig, ax = plt.subplots(figsize=(9.2, 5.8), layout="constrained")

    dw_label_used = False
    for p in dw_points:
        label = "DWF/MDWF data" if not dw_label_used else None
        dw_label_used = True

        ax.errorbar(
            p["x"],
            p["y"],
            xerr=p["xerr"],
            yerr=p["yerr"],
            fmt="o",
            linestyle="none",
            color=beta_colors[p["beta"]],
            markerfacecolor=beta_colors[p["beta"]],
            markeredgecolor=beta_colors[p["beta"]],
            markersize=5,
            capsize=2,
            alpha=0.95,
            label=label,
        )

    wilson_label_used = False
    for p in wilson_points:
        label = "Wilson data" if not wilson_label_used else None
        wilson_label_used = True

        ax.errorbar(
            p["x"],
            p["y"],
            xerr=p["xerr"],
            yerr=p["yerr"],
            fmt="s",
            linestyle="none",
            color=beta_colors[p["beta"]],
            markerfacecolor="none",
            markeredgecolor=beta_colors[p["beta"]],
            markersize=5,
            capsize=2,
            alpha=0.95,
            label=label,
        )

    x_all = np.array([p["x"] for p in dw_points] + [p["x"] for p in wilson_points])
    x_max = 1.05 * np.max(x_all)
    x_grid = np.linspace(0.0, x_max, 500)

    for fit_key in plot_fit_keys:
        fit = all_fits[fit_key]
        style = plot_registry[fit_key]

        if fit_key == "dw":
            y_fit, y_err = continuum_line_and_band_dw(x_grid, fit)
        elif fit_key == "dw2":
            y_fit, y_err = continuum_line_and_band_dw2(x_grid, fit)
        else:
            y_fit, y_err = wilson_continuum_line_and_band(x_grid, fit)

        ax.plot(
            x_grid,
            y_fit,
            color=style["color"],
            linestyle=style["linestyle"],
            label=fit["label"],
        )
        ax.fill_between(
            x_grid,
            y_fit - y_err,
            y_fit + y_err,
            color=style["color"],
            alpha=style["alpha_band"],
            linewidth=0,
        )

    ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$")
    ax.set_ylabel(r"$(f_{\rm PS} w_0)^2$")
    ax.set_xlim(0.0,)
    ax.set_ylim(0.0, 0.0200)

    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    ax.legend(loc="best", fontsize=8.5, framealpha=0.92)

    output_dir = os.path.dirname(output_plot)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_plot, dpi=300)
    plt.close()


# ============================================================
# JSON serialization helpers
# ============================================================
def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def fit_to_json_dict(fit):
    out = {
        "model_key": fit["model_key"],
        "label": fit.get("label_plain", fit["label"]),
        "chi2": fit["chi2"],
        "dof": fit["dof"],
        "cov": fit["cov"],
    }

    if "basis_terms" in fit:
        out["basis_terms"] = fit["basis_terms"]
        out["coeffs"] = fit["coeffs"]
        out["coeff_errs"] = fit["coeff_errs"]

    for key in ["A", "A_err", "B", "B_err", "C", "C_err", "D", "D_err", "L", "L_err"]:
        if key in fit:
            out[key] = fit[key]

    return to_serializable(out)


def save_fit_results_json(
    output_data,
    plot_fit_keys,
    dw_points_all,
    wilson_points_used,
    removed_first,
    removed_last,
    all_fits,
):
    output_dir = os.path.dirname(output_data)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "plot_fits": plot_fit_keys,
        "n_dw_points": len(dw_points_all),
        "n_wilson_points_used": len(wilson_points_used),
        "excluded_wilson": {
            "n_first": len(removed_first),
            "n_last": len(removed_last),
            "removed_first": to_serializable(removed_first),
            "removed_last": to_serializable(removed_last),
        },
        "fits": {
            key: fit_to_json_dict(fit)
            for key, fit in all_fits.items()
        },
    }

    with open(output_data, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"✓ Saved fit data → {output_data}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot (m_PS w0)^2 vs (f_PS w0)^2 using DWF/MDWF files and Wilson tables, "
            "including multiple Wilson continuum ansaetze based on a/w0 and DWF/MDWF "
            "continuum ansaetze. Optionally exclude the first n Wilson points "
            "(lightest x) and/or the last m Wilson points (largest x) before fit/plot."
        )
    )
    parser.add_argument("--plot_styles", default="")
    parser.add_argument(
        "--spectrum",
        nargs="+",
        required=True,
        help="List of DWF/MDWF spectrum.json files",
    )
    parser.add_argument(
        "--wflow",
        nargs="+",
        required=True,
        help="List of DWF/MDWF wflow_extract.json files",
    )
    parser.add_argument(
        "--spectrum_w",
        required=True,
        help="Wilson spectrum table JSON",
    )
    parser.add_argument(
        "--wflow_w",
        required=True,
        help="Wilson ensemble/wflow table JSON",
    )
    parser.add_argument(
        "--exclude_first_wilson",
        type=int,
        default=4,
        help=(
            "Exclude this many Wilson points with the smallest x=(m_PS w0)^2 "
            "before fitting and plotting."
        ),
    )
    parser.add_argument(
        "--exclude_last_wilson",
        type=int,
        default=0,
        help=(
            "Exclude this many Wilson points with the largest x=(m_PS w0)^2 "
            "before fitting and plotting."
        ),
    )
    parser.add_argument(
        "--output_plot",
        required=True,
        help="Output plot file",
    )
    parser.add_argument(
        "--output_data",
        required=True,
        help="Output JSON file for fit results",
    )
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    dw_points = collect_dw_points(args.spectrum, args.wflow)
    wilson_points = collect_wilson_points(args.spectrum_w, args.wflow_w)

    wilson_points, removed_first, removed_last = exclude_wilson_endpoints(
        wilson_points,
        n_first=args.exclude_first_wilson,
        m_last=args.exclude_last_wilson,
        x_key="x",
    )

    print_removed_wilson_points(removed_first, which="beginning")
    print_removed_wilson_points(removed_last, which="end")

    wilson_fit_1 = fit_wilson_x_plus_a_over_w0(wilson_points)
    wilson_fit_2 = fit_wilson_x_plus_a_over_w0_plus_sq(wilson_points)
    wilson_fit_3 = fit_wilson_x_plus_xaow0_plus_aow0_plus_sq(wilson_points)
    wilson_fit_4 = fit_wilson_x_plus_x2_plus_xaow0_plus_aow0_plus_sq(wilson_points)
    wilson_fit_5 = fit_wilson_x_plus_x2_plus_aow0_plus_sq(wilson_points)
    dw_fit = fit_dw_continuum(dw_points)
    dw2_fit = fit_dw2_continuum(dw_points)

    all_fits = {
        "wilson_1": wilson_fit_1,
        "wilson_2": wilson_fit_2,
        "wilson_3": wilson_fit_3,
        "wilson_4": wilson_fit_4,
        "wilson_5": wilson_fit_5,
        "dw": dw_fit,
        "dw2": dw2_fit,
    }

    plot_points_and_fits(
        dw_points=dw_points,
        wilson_points=wilson_points,
        all_fits=all_fits,
        plot_fit_keys=PLOT_FITS,
        output_plot=args.output_plot,
    )

    save_fit_results_json(
        output_data=args.output_data,
        plot_fit_keys=PLOT_FITS,
        dw_points_all=dw_points,
        wilson_points_used=wilson_points,
        removed_first=removed_first,
        removed_last=removed_last,
        all_fits=all_fits,
    )

    print(f"✓ Saved plot → {args.output_plot}")
    print(f"Fits shown on plot: {PLOT_FITS}")

    print_generic_wilson_fit_summary("Wilson [x + a/w0]", wilson_fit_1)
    print_generic_wilson_fit_summary(
        "Wilson [x + a/w0 + (a/w0)^2]", wilson_fit_2
    )
    print_generic_wilson_fit_summary(
        "Wilson [x + x(a/w0) + a/w0 + (a/w0)^2]", wilson_fit_3
    )
    print_generic_wilson_fit_summary(
        "Wilson [x + x^2 + x(a/w0) + a/w0 + (a/w0)^2]", wilson_fit_4
    )
    print_generic_wilson_fit_summary(
        "Wilson [x + x^2 + a/w0 + (a/w0)^2]", wilson_fit_5
    )
    print_dw_fit_summary(dw_fit)
    print_dw2_fit_summary(dw2_fit)


if __name__ == "__main__":
    main()
