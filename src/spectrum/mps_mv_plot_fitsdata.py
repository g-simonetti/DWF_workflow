#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from shared_continuum_models import (
    derive_dw2_start_parameters as _shared_derive_dw2_start_parameters,
    derive_wilson_start_parameters as _shared_derive_wilson_start_parameters,
    fit_dw2_continuum_linear as _shared_fit_dw2_continuum_linear,
    fit_dw2_continuum_nonlinear as _shared_fit_dw2_continuum_nonlinear,
    fit_wilson_complete_model_linear as _shared_fit_wilson_complete_model_linear,
    fit_wilson_complete_model_nonlinear as _shared_fit_wilson_complete_model_nonlinear,
    wilson_physical_continuum_line_and_band as _shared_wilson_continuum_line_and_band,
)

plt.style.use("tableau-colorblind10")


# ============================================================
# Generic readers
# ============================================================
def read_json_file(filename):
    """Read JSON file and return parsed object."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
        return json.loads(text)
    except json.JSONDecodeError:
        # Some legacy Wilson tables are written as JSON arrays of objects but
        # miss commas between successive top-level entries. Repair only this
        # narrow pattern before failing.
        repaired = re.sub(r"(?<=\})\s*(?=\{)", ",\n", text)
        try:
            return json.loads(repaired)
        except Exception as e:
            raise ValueError(f"Could not read JSON file: {filename}\n{e}") from e
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
      - results.standard_fit.VV.am_v.mean/sdev
    """
    data = read_json_file(filename)

    try:
        standard_fit = data["results"]["standard_fit"]

        pp = standard_fit["PP"]["am_ps"]
        vv = standard_fit["VV"]["am_v"]

        am_ps = float(pp["mean"])
        am_ps_err = float(pp["sdev"])

        am_v = float(vv["mean"])
        am_v_err = float(vv["sdev"])
    except KeyError as e:
        raise ValueError(
            f"Missing expected key in spectrum JSON '{filename}': {e}\n"
            "Expected results.standard_fit.PP.am_ps.mean/sdev and "
            "results.standard_fit.VV.am_v.mean/sdev"
        ) from e
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid numeric content in spectrum JSON '{filename}': {e}"
        ) from e

    return {
        "am_ps": am_ps,
        "am_ps_err": am_ps_err,
        "am_v": am_v,
        "am_v_err": am_v_err,
    }


def read_wflow_json(filename):
    """
    Read DW wflow JSON and extract:
      - summary.w0_sq
      - summary.w0_sq_err
    """
    data = read_json_file(filename)

    try:
        summary = data["summary"]
        w0_sq = float(summary["w0_sq"])
        w0_sq_err = float(summary["w0_sq_err"])
    except KeyError as e:
        raise ValueError(
            f"Missing expected key in wflow JSON '{filename}': {e}\n"
            "Expected summary.w0_sq and summary.w0_sq_err"
        ) from e
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid numeric content in wflow JSON '{filename}': {e}"
        ) from e

    return w0_sq, w0_sq_err


# ============================================================
# Wilson readers
# ============================================================
def read_wilson_spectrum_json(filename):
    """
    Read Wilson spectrum table JSON.

    Expected format: a list of dicts, one per ensemble, with keys:
      - Ensemble
      - amps : [value, error] or {"mean": ..., "sdev": ...}
      - amv  : [value, error] or {"mean": ..., "sdev": ...}
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
        am_v, am_v_err = parse_pair(row, "amv", filename)

        out[ens] = {
            "beta": float(row["beta"]),
            "m0": float(row["m0"]),
            "am_ps": am_ps,
            "am_ps_err": am_ps_err,
            "am_v": am_v,
            "am_v_err": am_v_err,
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


def mw0_sq_and_error_from_w0a(am, am_err, w0a, w0a_err):
    """
    x = (am * w0a)^2 = (m w0)^2
    """
    x = (am * w0a) ** 2
    dx_dam = 2.0 * am * (w0a**2)
    dx_dw0a = 2.0 * (am**2) * w0a
    var = (dx_dam**2) * (am_err**2) + (dx_dw0a**2) * (w0a_err**2)
    return x, np.sqrt(var)


def a_over_w0_and_error(w0a, w0a_err):
    """
    z = a / w0 = 1 / (w0 / a)
    """
    z = 1.0 / w0a
    z_err = w0a_err / (w0a**2)
    return z, z_err


def a_over_w0_sq_and_error_from_w0sq(w0_sq, w0_sq_err):
    """
    z = (a / w0)^2 = 1 / (w0 / a)^2 = 1 / w0_sq
    """
    z = 1.0 / w0_sq
    z_err = w0_sq_err / (w0_sq**2)
    return z, z_err


def a_over_w0_from_w0sq(w0_sq, w0_sq_err):
    """
    z = a / w0 = 1 / sqrt(w0_sq)
    """
    z = 1.0 / np.sqrt(w0_sq)
    z_err = 0.5 * w0_sq_err / (w0_sq ** 1.5)
    return z, z_err


def square_with_error(z, z_err):
    """
    q = z^2
    """
    q = z**2
    q_err = 2.0 * abs(z) * z_err
    return q, q_err


def compute_L_and_error(A, B, cov):
    if A <= 0:
        return np.nan, np.nan

    dL_dA = -B / (A**2)
    dL_dB = 1.0 / A
    var_L = (
        dL_dA**2 * cov[0, 0]
        + dL_dB**2 * cov[1, 1]
        + 2.0 * dL_dA * dL_dB * cov[0, 1]
    )
    return B / A, np.sqrt(max(var_L, 0.0))


def build_dw2_fit_result(coeffs, cov, chi2, dof, stage):
    errs = np.sqrt(np.diag(cov))
    A, B, C, D = coeffs
    A_err, B_err, C_err, D_err = errs
    L, L_err = compute_L_and_error(A, B, cov)

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
        "label": r"MDWF: $A + Bx + Cx^2 + D(a/w_0)^2$",
        "stage": stage,
        "model_key": "dw2",
    }


def build_wilson_fit_result(coeffs, cov, chi2, dof, stage, basis_terms):
    errs = np.sqrt(np.diag(cov))
    return {
        "basis_terms": basis_terms,
        "coeffs": coeffs,
        "coeff_errs": errs,
        "cov": cov,
        "chi2": chi2,
        "dof": dof,
        "label": (
            r"Wilson: $A + Bx + Cx^2 + Dx(a/w_0) + E(a/w_0) + F(a/w_0)^2$"
        ),
        "stage": stage,
        "model_key": "wilson_complete",
    }


def derive_dw2_start_parameters(linear_fit):
    return _shared_derive_dw2_start_parameters(linear_fit)


def derive_wilson_start_parameters(linear_fit):
    return _shared_derive_wilson_start_parameters(linear_fit)


# ============================================================
# DWF/MDWF continuum fit
# ============================================================
def fit_dw2_continuum_linear(points):
    return _shared_fit_dw2_continuum_linear(points)


def fit_dw2_continuum_nonlinear(points, initial_fit):
    return _shared_fit_dw2_continuum_nonlinear(points, initial_fit)


def continuum_line_and_band_dw2(m_PS_sq, fit):
    """
    Continuum line and 1-sigma band for:
        m_M^2 = m_M,chi^2 (1 + L_m_M m_PS^2 + Q_m_M m_PS^4)
    """
    m_M_chi_sq = fit["m_M_chi_sq"]
    L_m_M = fit["L_m_M"]
    Q_m_M = fit["Q_m_M"]
    cov = fit["cov"]

    y = m_M_chi_sq * (1.0 + L_m_M * m_PS_sq + Q_m_M * m_PS_sq**2)

    jac = np.column_stack(
        [
            1.0 + L_m_M * m_PS_sq + Q_m_M * m_PS_sq**2,
            m_M_chi_sq * m_PS_sq,
            m_M_chi_sq * m_PS_sq**2,
            np.zeros_like(m_PS_sq),
        ]
    )
    var = np.einsum("ij,jk,ik->i", jac, cov, jac)
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)

    return y, err


# ============================================================
# Wilson continuum fit
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


def fit_wilson_complete_model_linear(points):
    return _shared_fit_wilson_complete_model_linear(points)


def fit_wilson_complete_model_nonlinear(points, initial_fit):
    return _shared_fit_wilson_complete_model_nonlinear(
        points,
        initial_fit,
        p0=[
        0.320,   # m_M_chi_sq
        2.9,    # L_m_M
        -20,   # Q_m_M
        -0.183,   # W_m_M
        0.03,   # R_m_M
        -1,   # C_m_M
        ],
    )


def wilson_continuum_line_and_band(m_PS_sq, fit):
    return _shared_wilson_continuum_line_and_band(m_PS_sq, fit)


# ============================================================
# Formatting helpers
# ============================================================
def make_dw2_fit_text(fit):
    text = (
        r"$\mathrm{MDWF\ (nonlinear)}:$" "\n"
        r"$m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4) + R_{m_M} a^2$" "\n"
        rf"$m_{{M,\chi}}^2 = {fit['m_M_chi_sq']:.4f} \pm {fit['m_M_chi_sq_err']:.4f}$" "\n"
        rf"$L_{{m_M}} = {fit['L_m_M']:.4f} \pm {fit['L_m_M_err']:.4f}$" "\n"
        rf"$Q_{{m_M}} = {fit['Q_m_M']:.4f} \pm {fit['Q_m_M_err']:.4f}$" "\n"
        rf"$R_{{m_M}} = {fit['R_m_M']:.4f} \pm {fit['R_m_M_err']:.4f}$"
    )

    if fit["dof"] > 0:
        text += "\n" + rf"$\chi^2/\mathrm{{dof}} = {fit['chi2']:.2f}/{fit['dof']}$"

    return text


def make_wilson_fit_text(fit):
    text = (
        r"$\mathrm{Wilson\ (nonlinear)}:$" "\n"
        r"$m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$" "\n"
        r"$\qquad\qquad + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
    )

    text += "\n" + rf"$m_{{M,\chi}}^2 = {fit['m_M_chi_sq']:.4f} \pm {fit['m_M_chi_sq_err']:.4f}$"
    text += "\n" + rf"$L_{{m_M}} = {fit['L_m_M']:.4f} \pm {fit['L_m_M_err']:.4f}$"
    text += "\n" + rf"$Q_{{m_M}} = {fit['Q_m_M']:.4f} \pm {fit['Q_m_M_err']:.4f}$"
    text += "\n" + rf"$W_{{m_M}} = {fit['W_m_M']:.4f} \pm {fit['W_m_M_err']:.4f}$"
    text += "\n" + rf"$R_{{m_M}} = {fit['R_m_M']:.4f} \pm {fit['R_m_M_err']:.4f}$"
    text += "\n" + rf"$C_{{m_M}} = {fit['C_m_M']:.4f} \pm {fit['C_m_M_err']:.4f}$"

    if fit["dof"] > 0:
        text += "\n" + rf"$\chi^2/\mathrm{{dof}} = {fit['chi2']:.2f}/{fit['dof']}$"

    return text


def print_dw2_fit_summary(fit, title):
    print(f"{title}:")
    if "m_M_chi_sq" in fit:
        print(f"  m_M_chi_sq = {fit['m_M_chi_sq']:.8g} ± {fit['m_M_chi_sq_err']:.3g}")
        print(f"  L_m_M = {fit['L_m_M']:.8g} ± {fit['L_m_M_err']:.3g}")
        print(f"  Q_m_M = {fit['Q_m_M']:.8g} ± {fit['Q_m_M_err']:.3g}")
        print(f"  R_m_M = {fit['R_m_M']:.8g} ± {fit['R_m_M_err']:.3g}")
    else:
        for i, (term, coeff, err) in enumerate(
            zip(fit["basis_terms"], fit["coeffs"], fit["coeff_errs"])
        ):
            name = chr(ord("A") + i)
            print(f"  {name} = {coeff:.8g} ± {err:.3g}   [{term}]")

    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


def print_wilson_fit_summary(fit, title):
    print(f"{title}:")
    if "m_M_chi_sq" in fit:
        print(f"  m_M_chi_sq = {fit['m_M_chi_sq']:.8g} ± {fit['m_M_chi_sq_err']:.3g}")
        print(f"  L_m_M = {fit['L_m_M']:.8g} ± {fit['L_m_M_err']:.3g}")
        print(f"  Q_m_M = {fit['Q_m_M']:.8g} ± {fit['Q_m_M_err']:.3g}")
        print(f"  W_m_M = {fit['W_m_M']:.8g} ± {fit['W_m_M_err']:.3g}")
        print(f"  R_m_M = {fit['R_m_M']:.8g} ± {fit['R_m_M_err']:.3g}")
        print(f"  C_m_M = {fit['C_m_M']:.8g} ± {fit['C_m_M_err']:.3g}")
    else:
        for i, (term, coeff, err) in enumerate(
            zip(fit["basis_terms"], fit["coeffs"], fit["coeff_errs"])
        ):
            name = chr(ord("A") + i)
            print(f"  {name} = {coeff:.8g} ± {err:.3g}   [{term}]")

    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


def print_starting_parameters(title, params):
    print(f"{title}:")
    for key, value in params.items():
        print(f"  {key} = {value:.8g}")


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
        y, yerr = mw0_sq_and_error_from_w0sq(
            spec["am_v"], spec["am_v_err"], w0_sq, w0_sq_err
        )
        a_over_w0, a_over_w0_err = a_over_w0_from_w0sq(w0_sq, w0_sq_err)
        z, zerr = a_over_w0_sq_and_error_from_w0sq(w0_sq, w0_sq_err)

        dw_points.append(
            {
                "beta": beta,
                "m0": mass,
                "x": x,
                "xerr": xerr,
                "y": y,
                "yerr": yerr,
                "a_over_w0": a_over_w0,
                "a_over_w0_err": a_over_w0_err,
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
        y, yerr = mw0_sq_and_error_from_w0a(
            srow["am_v"], srow["am_v_err"], wrow["w0a"], wrow["w0a_err"]
        )

        z_lin, z_lin_err = a_over_w0_and_error(wrow["w0a"], wrow["w0a_err"])
        z_quad, z_quad_err = square_with_error(z_lin, z_lin_err)

        wilson_points.append(
            {
                "Ensemble": ens,
                "beta": srow["beta"],
                "m0": srow["m0"],
                "x": x,
                "xerr": xerr,
                "y": y,
                "yerr": yerr,
                "a_over_w0": z_lin,
                "a_over_w0_err": z_lin_err,
                "a_over_w0_sq": z_quad,
                "a_over_w0_sq_err": z_quad_err,
            }
        )

    return wilson_points


# ============================================================
# Plotting
# ============================================================
def plot_points_and_fits(
    dw_points,
    wilson_points,
    dw2_fit,
    wilson_fit,
    output_file,
):
    all_betas = sorted(
        {p["beta"] for p in dw_points}.union({p["beta"] for p in wilson_points})
    )
    beta_colors = {b: f"C{i % 10}" for i, b in enumerate(all_betas)}

    fig, ax = plt.subplots(figsize=(8.4, 5.0), layout="constrained")

    # DWF/MDWF points
    dw_label_used = False
    for p in dw_points:
        label = "DWF/MDWF" if not dw_label_used else None
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
            label=label,
        )

    # Wilson points
    wilson_label_used = False
    for p in wilson_points:
        label = "Wilson" if not wilson_label_used else None
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
            label=label,
        )

    # Continuum lines and bands
    x_all = np.array([p["x"] for p in dw_points] + [p["x"] for p in wilson_points])
    x_max = 1.05 * np.max(x_all)
    x_grid = np.linspace(0.0, x_max, 400)

    # Wilson complete model
    y_wilson, y_wilson_err = wilson_continuum_line_and_band(x_grid, wilson_fit)
    ax.plot(
        x_grid,
        y_wilson,
        linestyle=":",
        color="tab:red",
        label=wilson_fit["label"],
    )
    ax.fill_between(
        x_grid,
        y_wilson - y_wilson_err,
        y_wilson + y_wilson_err,
        color="tab:red",
        alpha=0.10,
        linewidth=0,
    )

    # MDWF with x^2
    y_dw2, y_dw2_err = continuum_line_and_band_dw2(x_grid, dw2_fit)
    ax.plot(
        x_grid,
        y_dw2,
        linestyle="-",
        color="tab:purple",
        label=dw2_fit["label"],
    )
    ax.fill_between(
        x_grid,
        y_dw2 - y_dw2_err,
        y_dw2 + y_dw2_err,
        color="tab:purple",
        alpha=0.12,
        linewidth=0,
    )

    ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$")
    ax.set_ylabel(r"$(m_{\rm V} w_0)^2$")
    ax.set_xlim(0.0,)

    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    ax.legend(loc="best", fontsize=9)

    wilson_text = make_wilson_fit_text(wilson_fit)
    dw_text = make_dw2_fit_text(dw2_fit)

    ax.text(
        0.98,
        0.02,
        dw_text,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    ax.text(
        0.02,
        0.78,
        wilson_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_file, dpi=300)
    plt.close()


# ============================================================
# JSON helpers
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


def linear_fit_to_json_dict(fit):
    out = {
        "model_key": fit["model_key"],
        "stage": fit["stage"],
        "label": fit["label"],
        "chi2": fit["chi2"],
        "dof": fit["dof"],
        "cov": fit["cov"],
    }

    for key in ["A", "A_err", "B", "B_err", "C", "C_err", "D", "D_err", "L", "L_err"]:
        if key in fit:
            out[key] = fit[key]

    if "basis_terms" in fit:
        out["basis_terms"] = fit["basis_terms"]
        out["coeffs"] = fit["coeffs"]
        out["coeff_errs"] = fit["coeff_errs"]

    return to_serializable(out)


def physical_fit_to_json_dict(fit):
    out = {
        "model_key": fit["model_key"],
        "stage": fit["stage"],
        "label": fit["label"],
        "chi2": fit["chi2"],
        "dof": fit["dof"],
        "cov": fit["cov"],
    }

    for key in [
        "m_M_chi_sq",
        "m_M_chi_sq_err",
        "L_m_M",
        "L_m_M_err",
        "Q_m_M",
        "Q_m_M_err",
        "W_m_M",
        "W_m_M_err",
        "R_m_M",
        "R_m_M_err",
        "C_m_M",
        "C_m_M_err",
    ]:
        if key in fit:
            out[key] = fit[key]

    return to_serializable(out)


def save_fit_results_json(
    output_data,
    dw_points,
    wilson_points,
    dw2_fit_linear,
    dw2_fit_nonlinear,
    wilson_fit_linear,
    wilson_fit_nonlinear,
):
    output_dir = os.path.dirname(output_data)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "n_dw_points_used": len(dw_points),
        "n_wilson_points_used": len(wilson_points),
        "fits": {
            "dw2": {
                "linearized": linear_fit_to_json_dict(dw2_fit_linear),
                "starting_parameters": to_serializable(
                    derive_dw2_start_parameters(dw2_fit_linear)
                ),
                "nonlinear": physical_fit_to_json_dict(dw2_fit_nonlinear),
            },
            "wilson_physical": {
                "linearized": linear_fit_to_json_dict(wilson_fit_linear),
                "starting_parameters": to_serializable(
                    derive_wilson_start_parameters(wilson_fit_linear)
                ),
                "nonlinear": physical_fit_to_json_dict(wilson_fit_nonlinear),
            },
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
            "Plot (m_PS w0)^2 vs (m_V w0)^2 using DWF/MDWF files and Wilson tables, "
            "using two continuum extrapolations with 1-sigma bands. "
            "The linearized fit is used only to estimate starting parameters. "
            "The final plotted fit is then performed in terms of the physical "
            "parameters m_M_chi_sq, L_m_M, Q_m_M, W_m_M, R_m_M, and C_m_M "
            "for Wilson, and m_M_chi_sq, L_m_M, Q_m_M, and R_m_M for MDWF."
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
        "--output_plot",
        "--output_file",
        dest="output_plot",
        required=True,
        help="Output plot file",
    )
    parser.add_argument(
        "--output_data",
        required=True,
        help="Output JSON file storing both linearized and nonlinear fit results",
    )
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    # ------------------------------------------------------------
    # Collect data
    # ------------------------------------------------------------
    dw_points = collect_dw_points(args.spectrum, args.wflow)
    wilson_points = collect_wilson_points(args.spectrum_w, args.wflow_w)

    # ------------------------------------------------------------
    # Fits
    # ------------------------------------------------------------
    wilson_fit_linear = fit_wilson_complete_model_linear(wilson_points)
    dw2_fit_linear = fit_dw2_continuum_linear(dw_points)

    wilson_fit = fit_wilson_complete_model_nonlinear(
        wilson_points, wilson_fit_linear
    )
    dw2_fit = fit_dw2_continuum_nonlinear(dw_points, dw2_fit_linear)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plot_points_and_fits(
        dw_points=dw_points,
        wilson_points=wilson_points,
        dw2_fit=dw2_fit,
        wilson_fit=wilson_fit,
        output_file=args.output_plot,
    )

    save_fit_results_json(
        output_data=args.output_data,
        dw_points=dw_points,
        wilson_points=wilson_points,
        dw2_fit_linear=dw2_fit_linear,
        dw2_fit_nonlinear=dw2_fit,
        wilson_fit_linear=wilson_fit_linear,
        wilson_fit_nonlinear=wilson_fit,
    )

    print(f"✓ Saved plot → {args.output_plot}")
    print_starting_parameters(
        "Wilson starting parameters from linearized fit",
        derive_wilson_start_parameters(wilson_fit_linear),
    )
    print_wilson_fit_summary(wilson_fit, "Wilson complete model [nonlinear]")
    print_starting_parameters(
        "DWF/MDWF starting parameters from linearized fit",
        derive_dw2_start_parameters(dw2_fit_linear),
    )
    print_dw2_fit_summary(dw2_fit, "DWF/MDWF dw2 [nonlinear]")


if __name__ == "__main__":
    main()
