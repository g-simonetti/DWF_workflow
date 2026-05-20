import numpy as np
from scipy.optimize import curve_fit


DEFAULT_DW2_LINEAR_LABEL = r"MDWF linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a^2$"
DEFAULT_DW2_PHYSICAL_LABEL = (
    r"MDWF: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)"
    r" + R_{m_M} a^2$"
)
DEFAULT_WILSON_LINEAR_LABEL = (
    r"Wilson linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a m_{PS}^2 + E a + F a^2$"
)
DEFAULT_WILSON_PHYSICAL_LABEL = (
    r"Wilson: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
    r" + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
)


def solve_weighted_least_squares(design_matrix, y, ye, label):
    if np.any(ye <= 0):
        raise ValueError(f"All {label} y-errors must be positive.")

    w_sqrt = np.diag(1.0 / ye)
    mw = w_sqrt @ design_matrix
    yw = w_sqrt @ y

    coeffs, _, _, _ = np.linalg.lstsq(mw, yw, rcond=None)

    normal_matrix = mw.T @ mw
    cov = np.linalg.inv(normal_matrix)
    errs = np.sqrt(np.diag(cov))

    y_fit = design_matrix @ coeffs
    chi2 = np.sum(((y - y_fit) / ye) ** 2)
    dof = len(y) - design_matrix.shape[1]

    return coeffs, errs, cov, chi2, dof


def ratio_or_nan(num, den):
    if den == 0:
        return np.nan
    return num / den


def derive_dw2_start_parameters(linear_fit):
    coeffs = linear_fit["coeffs"]
    return {
        "m_M_chi_sq": coeffs[0],
        "L_m_M": ratio_or_nan(coeffs[1], coeffs[0]),
        "Q_m_M": ratio_or_nan(coeffs[2], coeffs[0]),
        "R_m_M": coeffs[3],
    }


def _build_dw2_physical_fit_result(params, cov, chi2, dof, stage, label):
    errs = np.sqrt(np.diag(cov))
    m_M_chi_sq, L_m_M, Q_m_M, R_m_M = params
    m_M_chi_sq_err, L_m_M_err, Q_m_M_err, R_m_M_err = errs

    return {
        "model_key": "dw2_physical",
        "stage": stage,
        "label": label,
        "m_M_chi_sq": float(m_M_chi_sq),
        "m_M_chi_sq_err": float(m_M_chi_sq_err),
        "L_m_M": float(L_m_M),
        "L_m_M_err": float(L_m_M_err),
        "Q_m_M": float(Q_m_M),
        "Q_m_M_err": float(Q_m_M_err),
        "R_m_M": float(R_m_M),
        "R_m_M_err": float(R_m_M_err),
        "cov": cov,
        "chi2": float(chi2),
        "dof": int(dof),
    }


def fit_dw2_continuum_linear(
    points,
    *,
    fit_label=DEFAULT_DW2_LINEAR_LABEL,
    error_label="DWF/MDWF linearized model",
):
    basis_terms = ["1", "x", "x2", "a_over_w0_sq"]
    n_params = len(basis_terms)

    if len(points) < n_params:
        raise ValueError(
            f"Need at least {n_params} DWF/MDWF points for linearized continuum fit."
        )

    x = np.array([p["x"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    ye = np.array([p["yerr"] for p in points], dtype=float)
    z = np.array([p["a_over_w0_sq"] for p in points], dtype=float)

    design = np.column_stack([np.ones_like(x), x, x**2, z])
    coeffs, errs, cov, chi2, dof = solve_weighted_least_squares(
        design, y, ye, error_label
    )

    return {
        "model_key": "dw2_linearized",
        "stage": "linearized",
        "label": fit_label,
        "basis_terms": basis_terms,
        "coeffs": coeffs,
        "coeff_errs": errs,
        "cov": cov,
        "chi2": chi2,
        "dof": dof,
    }


def dw2_physical_model(inputs, m_M_chi_sq, L_m_M, Q_m_M, R_m_M):
    m_ps_sq, a = inputs
    return m_M_chi_sq * (1.0 + L_m_M * m_ps_sq + Q_m_M * m_ps_sq**2) + R_m_M * a**2


def fit_dw2_continuum_nonlinear(
    points,
    initial_fit=None,
    *,
    p0=None,
    fit_label=DEFAULT_DW2_PHYSICAL_LABEL,
):
    m_ps_sq = np.array([p["x"] for p in points], dtype=float)
    a = np.array([p["a_over_w0"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    ye = np.array([p["yerr"] for p in points], dtype=float)

    if p0 is None:
        if initial_fit is None:
            raise ValueError("Either initial_fit or p0 must be provided.")
        p0_dict = derive_dw2_start_parameters(initial_fit)
        p0 = [
            p0_dict["m_M_chi_sq"],
            p0_dict["L_m_M"],
            p0_dict["Q_m_M"],
            p0_dict["R_m_M"],
        ]

    popt, pcov = curve_fit(
        dw2_physical_model,
        (m_ps_sq, a),
        y,
        sigma=ye,
        absolute_sigma=True,
        p0=p0,
        maxfev=20000,
    )

    residuals = y - dw2_physical_model((m_ps_sq, a), *popt)
    chi2 = float(np.sum((residuals / ye) ** 2))
    dof = int(len(y) - len(popt))
    return _build_dw2_physical_fit_result(
        popt, pcov, chi2, dof, stage="nonlinear", label=fit_label
    )


def dw2_physical_continuum_line_and_band(m_ps_sq, fit):
    """
    Continuum line and 1-sigma band for:
        m_M^2 = m_M,chi^2 (1 + L_m_M m_PS^2 + Q_m_M m_PS^4)
    """
    m_M_chi_sq = fit["m_M_chi_sq"]
    L_m_M = fit["L_m_M"]
    Q_m_M = fit["Q_m_M"]
    cov = np.asarray(fit["cov"], dtype=float)

    y = m_M_chi_sq * (1.0 + L_m_M * m_ps_sq + Q_m_M * m_ps_sq**2)

    jac = np.column_stack(
        [
            1.0 + L_m_M * m_ps_sq + Q_m_M * m_ps_sq**2,
            m_M_chi_sq * m_ps_sq,
            m_M_chi_sq * m_ps_sq**2,
            np.zeros_like(m_ps_sq),
        ]
    )
    var = np.einsum("ij,jk,ik->i", jac, cov, jac)
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)
    return y, err


def derive_wilson_start_parameters(linear_fit):
    coeffs = linear_fit["coeffs"]
    a, b, c, c_am, w_val, r_val = coeffs
    return {
        "m_M_chi_sq": float(a),
        "L_m_M": float(ratio_or_nan(b, a)),
        "Q_m_M": float(ratio_or_nan(c, a)),
        "W_m_M": float(w_val),
        "R_m_M": float(r_val),
        "C_m_M": float(c_am),
    }


def _build_wilson_physical_fit_result(params, cov, chi2, dof, stage, label):
    errs = np.sqrt(np.diag(cov))
    (
        m_M_chi_sq,
        L_m_M,
        Q_m_M,
        W_m_M,
        R_m_M,
        C_m_M,
    ) = params
    (
        m_M_chi_sq_err,
        L_m_M_err,
        Q_m_M_err,
        W_m_M_err,
        R_m_M_err,
        C_m_M_err,
    ) = errs

    return {
        "model_key": "wilson_physical",
        "stage": stage,
        "label": label,
        "m_M_chi_sq": float(m_M_chi_sq),
        "m_M_chi_sq_err": float(m_M_chi_sq_err),
        "L_m_M": float(L_m_M),
        "L_m_M_err": float(L_m_M_err),
        "Q_m_M": float(Q_m_M),
        "Q_m_M_err": float(Q_m_M_err),
        "W_m_M": float(W_m_M),
        "W_m_M_err": float(W_m_M_err),
        "R_m_M": float(R_m_M),
        "R_m_M_err": float(R_m_M_err),
        "C_m_M": float(C_m_M),
        "C_m_M_err": float(C_m_M_err),
        "cov": cov,
        "chi2": float(chi2),
        "dof": int(dof),
    }


def fit_wilson_complete_model_linear(
    points,
    *,
    fit_label=DEFAULT_WILSON_LINEAR_LABEL,
    error_label="Wilson complete model",
):
    basis_terms = ["1", "x", "x2", "x_a_over_w0", "a_over_w0", "a_over_w0_sq"]
    n_params = len(basis_terms)

    if len(points) < n_params:
        raise ValueError(
            f"Need at least {n_params} Wilson points for the complete Wilson fit."
        )

    x = np.array([p["x"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    ye = np.array([p["yerr"] for p in points], dtype=float)
    a = np.array([p["a_over_w0"] for p in points], dtype=float)
    a_sq = np.array([p["a_over_w0_sq"] for p in points], dtype=float)

    design = np.column_stack([np.ones_like(x), x, x**2, x * a, a, a_sq])
    coeffs, errs, cov, chi2, dof = solve_weighted_least_squares(
        design, y, ye, error_label
    )
    dof = len(y) - len(basis_terms) - 1

    return {
        "model_key": "wilson_complete",
        "stage": "linearized",
        "label": fit_label,
        "basis_terms": basis_terms,
        "coeffs": coeffs,
        "coeff_errs": errs,
        "cov": cov,
        "chi2": chi2,
        "dof": dof,
    }


def wilson_physical_model(inputs, m_M_chi_sq, L_m_M, Q_m_M, W_m_M, R_m_M, C_m_M):
    m_ps_sq, a = inputs
    return (
        m_M_chi_sq * (1.0 + L_m_M * m_ps_sq + Q_m_M * m_ps_sq**2)
        + W_m_M * a
        + R_m_M * a**2
        + C_m_M * a * m_ps_sq
    )


def fit_wilson_complete_model_nonlinear(
    points,
    initial_fit=None,
    *,
    p0=None,
    fit_label=DEFAULT_WILSON_PHYSICAL_LABEL,
):
    m_ps_sq = np.array([p["x"] for p in points], dtype=float)
    a = np.array([p["a_over_w0"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    ye = np.array([p["yerr"] for p in points], dtype=float)

    if p0 is None:
        if initial_fit is None:
            raise ValueError("Either initial_fit or p0 must be provided.")
        p0_dict = derive_wilson_start_parameters(initial_fit)
        p0 = [
            p0_dict["m_M_chi_sq"],
            p0_dict["L_m_M"],
            p0_dict["Q_m_M"],
            p0_dict["W_m_M"],
            p0_dict["R_m_M"],
            p0_dict["C_m_M"],
        ]

    popt, pcov = curve_fit(
        wilson_physical_model,
        (m_ps_sq, a),
        y,
        sigma=ye,
        absolute_sigma=True,
        p0=p0,
        maxfev=20000,
    )

    residuals = y - wilson_physical_model((m_ps_sq, a), *popt)
    chi2 = float(np.sum((residuals / ye) ** 2))
    dof = int(len(y) - len(popt) - 1)

    return _build_wilson_physical_fit_result(
        popt, pcov, chi2, dof, stage="nonlinear", label=fit_label
    )


def wilson_physical_continuum_line_and_band(m_ps_sq, fit):
    m_M_chi_sq = fit["m_M_chi_sq"]
    L_m_M = fit["L_m_M"]
    Q_m_M = fit["Q_m_M"]
    cov = np.asarray(fit["cov"], dtype=float)

    y = m_M_chi_sq * (1.0 + L_m_M * m_ps_sq + Q_m_M * m_ps_sq**2)
    jac = np.column_stack(
        [
            1.0 + L_m_M * m_ps_sq + Q_m_M * m_ps_sq**2,
            m_M_chi_sq * m_ps_sq,
            m_M_chi_sq * m_ps_sq**2,
            np.zeros_like(m_ps_sq),
            np.zeros_like(m_ps_sq),
            np.zeros_like(m_ps_sq),
        ]
    )
    var = np.einsum("ij,jk,ik->i", jac, cov, jac)
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)
    return y, err
