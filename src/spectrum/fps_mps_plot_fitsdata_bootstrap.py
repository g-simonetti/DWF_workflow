#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from fps_mps_plot import (
    PLOT_FITS,
    collect_wilson_points,
    continuum_line_and_band_dw,
    continuum_line_and_band_dw2,
    exclude_wilson_endpoints,
    extract_beta_mass_from_path,
    fit_dw_continuum,
    print_dw2_fit_summary,
    print_dw_fit_summary,
    print_removed_wilson_points,
    read_json_file,
    to_serializable,
    validate_plot_fit_keys,
    wilson_continuum_line_and_band,
)
from shared_continuum_models import (
    derive_dw2_start_parameters as _shared_derive_dw2_start_parameters,
    derive_wilson_start_parameters as _shared_derive_wilson_start_parameters,
    dw2_physical_model,
    fit_dw2_continuum_linear as _shared_fit_dw2_continuum_linear,
    fit_dw2_continuum_nonlinear as _shared_fit_dw2_continuum_nonlinear,
    fit_wilson_complete_model_linear as _shared_fit_wilson_complete_model_linear,
    fit_wilson_complete_model_nonlinear as _shared_fit_wilson_complete_model_nonlinear,
    wilson_physical_continuum_line_and_band as _shared_wilson_continuum_line_and_band,
)

plt.style.use("tableau-colorblind10")

FPS_WILSON_PHYSICAL_LABEL = (
    r"Wilson: $f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
    r" $+ W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
)


def summary_stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    if arr.size == 1:
        sdev = 0.0
    else:
        sdev = float(np.std(arr, ddof=1))
    return {
        "mean": float(np.mean(arr)),
        "sdev": sdev,
        "n": int(arr.size),
    }


def select_bootstrap_plot_fit_keys(has_wilson):
    plot_fit_keys = []
    for key in PLOT_FITS:
        mapped = "wilson_physical" if key.startswith("wilson_") else key
        if mapped == "wilson_physical" and not has_wilson:
            continue
        if mapped not in plot_fit_keys:
            plot_fit_keys.append(mapped)

    if "dw2" not in plot_fit_keys:
        plot_fit_keys.append("dw2")

    return plot_fit_keys


def _require_key(obj, keys, filename):
    cur = obj
    for key in keys:
        if key not in cur:
            joined = ".".join(keys)
            raise ValueError(f"Missing key '{joined}' in '{filename}'")
        cur = cur[key]
    return cur


def read_spectrum_bootstrap_json(filename):
    data = read_json_file(filename)
    return {
        "bootstrap": _require_key(data, ["bootstrap"], filename),
        "pp_samples": _require_key(
            data, ["results", "bootstrap_fit", "PP", "samples"], filename
        ),
        "sim_samples": _require_key(
            data,
            ["results", "bootstrap_fit", "simultaneous_PP_A0P", "samples"],
            filename,
        ),
        "za_samples": _require_key(
            data, ["results", "bootstrap_fit", "Z_A", "samples"], filename
        ),
    }


def read_wflow_bootstrap_json(filename):
    data = read_json_file(filename)
    bootstrap = _require_key(data, ["bootstrap", "w0"], filename)
    samples = _require_key(data, ["bootstrap", "w0", "samples"], filename)
    summary = _require_key(data, ["summary"], filename)
    return {
        "bootstrap": bootstrap,
        "w0_samples": samples,
        "summary": summary,
    }


def ensure_bootstrap_alignment(spec_bootstrap, flow_bootstrap, spec_path, flow_path):
    checks = [
        ("path_key", spec_bootstrap.get("path_key"), flow_bootstrap.get("path_key")),
        ("seed", spec_bootstrap.get("seed"), flow_bootstrap.get("seed")),
        ("n_boot", spec_bootstrap.get("n_boot"), flow_bootstrap.get("n_boot")),
        ("cfg_numbers", spec_bootstrap.get("cfg_numbers"), flow_bootstrap.get("cfg_numbers")),
        ("boot_idx", spec_bootstrap.get("boot_idx"), flow_bootstrap.get("boot_idx")),
    ]
    for name, lhs, rhs in checks:
        if lhs != rhs:
            raise ValueError(
                "Bootstrap mismatch between spectrum and wflow for\n"
                f"  spectrum: {spec_path}\n"
                f"  wflow:    {flow_path}\n"
                f"Field '{name}' differs. This usually means the selected "
                "configurations or bootstrap ensemble are not aligned."
            )


def build_dw_bootstrap_ensemble(spec_path, wflow_path):
    beta, mass = extract_beta_mass_from_path(spec_path)
    if beta is None or mass is None:
        raise ValueError(f"Could not extract beta/mass from path: {spec_path}")

    spec = read_spectrum_bootstrap_json(spec_path)
    wflow = read_wflow_bootstrap_json(wflow_path)
    ensure_bootstrap_alignment(spec["bootstrap"], wflow["bootstrap"], spec_path, wflow_path)

    pp_samples = spec["pp_samples"]
    sim_samples = spec["sim_samples"]
    za_samples = spec["za_samples"]
    w0_samples = wflow["w0_samples"]
    n_boot = int(spec["bootstrap"]["n_boot"])

    if (
        len(pp_samples) != n_boot
        or len(sim_samples) != n_boot
        or len(za_samples) != n_boot
        or len(w0_samples) != n_boot
    ):
        raise ValueError(
            f"Inconsistent bootstrap sample count for ensemble:\n"
            f"  spectrum: {spec_path}\n"
            f"  wflow:    {wflow_path}"
        )

    replica_points = []
    for b in range(n_boot):
        pp_b = pp_samples[b]
        sim_b = sim_samples[b]
        za_b = za_samples[b]
        w0_b = w0_samples[b]

        if pp_b is None or sim_b is None or za_b is None or w0_b is None:
            replica_points.append(None)
            continue

        m_ps = pp_b.get("m_ps")
        f_ps = sim_b.get("f_ps")
        z_a = za_b.get("Z_A")
        w0 = w0_b.get("w0")
        if m_ps is None or f_ps is None or z_a is None or w0 in (None, 0.0):
            replica_points.append(None)
            continue

        fps = float(z_a) * float(f_ps)
        x = float((float(m_ps) * float(w0)) ** 2)
        y = float((fps * float(w0)) ** 2)
        a_over_w0 = float(1.0 / float(w0))
        replica_points.append(
            {
                "index": int(b),
                "beta": beta,
                "m0": mass,
                "x": x,
                "y": y,
                "a_over_w0": a_over_w0,
                "w0": float(w0),
                "m_ps": float(m_ps),
                "f_ps": float(f_ps),
                "Z_A": float(z_a),
                "fps": fps,
            }
        )

    valid = [sample for sample in replica_points if sample is not None]
    if not valid:
        raise RuntimeError(f"No valid bootstrap replicas for ensemble: {spec_path}")

    x_stats = summary_stats([sample["x"] for sample in valid])
    y_stats = summary_stats([sample["y"] for sample in valid])
    a_stats = summary_stats([sample["a_over_w0"] for sample in valid])
    z_stats = summary_stats([sample["Z_A"] for sample in valid])
    fps_stats = summary_stats([sample["fps"] for sample in valid])

    point = {
        "beta": beta,
        "m0": mass,
        "x": x_stats["mean"],
        "xerr": x_stats["sdev"],
        "y": y_stats["mean"],
        "yerr": y_stats["sdev"],
        "Z_A": z_stats["mean"],
        "Z_A_err": z_stats["sdev"],
        "fps": fps_stats["mean"],
        "fps_err": fps_stats["sdev"],
        "a_over_w0": a_stats["mean"],
        "a_over_w0_err": a_stats["sdev"],
        "a_over_w0_sq": a_stats["mean"] ** 2,
        "a_over_w0_sq_err": 2.0 * abs(a_stats["mean"]) * a_stats["sdev"],
    }

    return {
        "point": point,
        "bootstrap_samples": replica_points,
        "bootstrap_meta": spec["bootstrap"],
        "paths": {"spectrum": spec_path, "wflow": wflow_path},
    }


def collect_dw_bootstrap_ensembles(spectrum_files, wflow_files):
    if len(spectrum_files) != len(wflow_files):
        raise ValueError("Number of --spectrum files must equal number of --wflow files.")

    ensembles = [
        build_dw_bootstrap_ensemble(spec_path, wflow_path)
        for spec_path, wflow_path in zip(spectrum_files, wflow_files)
    ]

    points = [entry["point"] for entry in ensembles]
    n_boot_values = {int(entry["bootstrap_meta"]["n_boot"]) for entry in ensembles}
    if len(n_boot_values) != 1:
        raise ValueError("All MDWF ensembles must have the same number of bootstrap replicas.")

    n_boot = n_boot_values.pop()
    bootstrap_point_sets = []
    failures = []

    for b in range(n_boot):
        point_set = []
        missing = []
        for entry in ensembles:
            sample = entry["bootstrap_samples"][b]
            if sample is None:
                missing.append(entry["paths"]["spectrum"])
                continue
            point_set.append(
                {
                    "beta": sample["beta"],
                    "m0": sample["m0"],
                    "x": sample["x"],
                    "y": sample["y"],
                    "a_over_w0": sample["a_over_w0"],
                    "a_over_w0_sq": sample["a_over_w0"] ** 2,
                    "yerr": entry["point"]["yerr"],
                }
            )

        if missing:
            failures.append(
                {
                    "index": int(b),
                    "reason": "missing ensemble sample",
                    "paths": missing,
                }
            )
            bootstrap_point_sets.append(None)
            continue

        bootstrap_point_sets.append(point_set)

    return points, bootstrap_point_sets, failures


def derive_dw2_start_parameters(linear_fit):
    return _shared_derive_dw2_start_parameters(linear_fit)


def fit_dw2_continuum_linear(points):
    return _shared_fit_dw2_continuum_linear(points)


def fit_dw2_continuum_nonlinear(points, initial_fit):
    return _shared_fit_dw2_continuum_nonlinear(points, initial_fit)


def fit_dw2_bootstrap_replica(points):
    linear_fit = fit_dw2_continuum_linear(points)
    nonlinear_fit = fit_dw2_continuum_nonlinear(points, linear_fit)
    params = np.array(
        [
            nonlinear_fit["m_M_chi_sq"],
            nonlinear_fit["L_m_M"],
            nonlinear_fit["Q_m_M"],
            nonlinear_fit["R_m_M"],
        ],
        dtype=float,
    )
    return linear_fit, nonlinear_fit, params, float(nonlinear_fit["chi2"])


def _robust_keep_mask(params):
    params = np.asarray(params, dtype=float)
    n_rows = params.shape[0]
    keep = np.all(np.isfinite(params), axis=1)

    if n_rows < 5 or not np.any(keep):
        return keep, []

    med = np.median(params[keep], axis=0)
    mad = np.median(np.abs(params[keep] - med), axis=0)
    robust_sigma = 1.4826 * mad

    rejected = []
    for i in range(n_rows):
        if not keep[i]:
            rejected.append(
                {
                    "row_index": int(i),
                    "reason": "non_finite_parameters",
                }
            )
            continue

        deviations = np.abs(params[i] - med)
        flagged_columns = []
        for j, sigma_j in enumerate(robust_sigma):
            if sigma_j <= 0.0:
                continue
            if deviations[j] > 10.0 * sigma_j:
                flagged_columns.append(int(j))

        if flagged_columns:
            keep[i] = False
            rejected.append(
                {
                    "row_index": int(i),
                    "reason": "robust_parameter_outlier",
                    "flagged_columns": flagged_columns,
                }
            )

    return keep, rejected


def fit_dw2_bootstrap_summary(bootstrap_point_sets, dw_points, central_fit, start_params):
    p0 = [
        start_params["m_M_chi_sq"],
        start_params["L_m_M"],
        start_params["Q_m_M"],
        start_params["R_m_M"],
    ]

    samples = []
    failures = []
    success_rows = []

    for b, point_set in enumerate(bootstrap_point_sets):
        if point_set is None:
            samples.append(None)
            continue
        try:
            replica_linear_fit = fit_dw2_continuum_linear(point_set)
            replica_start_params = derive_dw2_start_parameters(replica_linear_fit)
            replica_non_linear_fit = fit_dw2_continuum_nonlinear(point_set, replica_linear_fit)
            params = np.array(
                [
                    replica_non_linear_fit["m_M_chi_sq"],
                    replica_non_linear_fit["L_m_M"],
                    replica_non_linear_fit["Q_m_M"],
                    replica_non_linear_fit["R_m_M"],
                ],
                dtype=float,
            )
            chi2 = float(replica_non_linear_fit["chi2"])
            sample = {
                "index": int(b),
                "m_M_chi_sq": float(params[0]),
                "L_m_M": float(params[1]),
                "Q_m_M": float(params[2]),
                "R_m_M": float(params[3]),
                "chi2": chi2,
                "linearized": linear_fit_to_json_dict(replica_linear_fit),
                "starting_parameters": to_serializable(replica_start_params),
            }
            success_rows.append((int(b), params, chi2, sample))
            samples.append(sample)
        except Exception as exc:
            failures.append({"index": int(b), "error": str(exc)})
            samples.append(None)

    if not success_rows:
        raise RuntimeError("All bootstrap MDWF continuum fits failed.")

    raw_params = np.asarray([row[1] for row in success_rows], dtype=float)
    keep_mask, rejected = _robust_keep_mask(raw_params)

    param_rows = []
    chi2_values = []
    for row_idx, (boot_index, params, chi2, _sample) in enumerate(success_rows):
        if keep_mask[row_idx]:
            param_rows.append(params)
            chi2_values.append(chi2)
            continue

        samples[boot_index] = None
        reject_info = rejected.pop(0) if rejected else {"reason": "rejected_bootstrap_replica"}
        failures.append(
            {
                "index": int(boot_index),
                "error": reject_info["reason"],
                **{k: v for k, v in reject_info.items() if k != "row_index"},
            }
        )

    if not param_rows:
        raise RuntimeError("All bootstrap MDWF continuum fits were rejected by the outlier filter.")

    params = np.asarray(param_rows, dtype=float)
    mean_params = np.mean(params, axis=0)
    if params.shape[0] > 1:
        cov = np.cov(params, rowvar=False, ddof=1)
    else:
        cov = np.zeros((4, 4), dtype=float)

    x = np.array([p["x"] for p in dw_points], dtype=float)
    a = np.array([p["a_over_w0"] for p in dw_points], dtype=float)
    y = np.array([p["y"] for p in dw_points], dtype=float)
    ye = np.array([p["yerr"] for p in dw_points], dtype=float)
    residuals = y - dw2_physical_model((x, a), *mean_params)
    final_chi2 = float(np.sum((residuals / ye) ** 2))
    final_dof = int(len(y) - len(mean_params))

    errs = np.sqrt(np.diag(cov))
    fit = {
        "model_key": "dw2_physical_bootstrap",
        "stage": "bootstrap_summary",
        "label": (
            r"MDWF bootstrap: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)"
            r" + R_{m_M} a^2$"
        ),
        "m_M_chi_sq": float(mean_params[0]),
        "m_M_chi_sq_err": float(errs[0]),
        "L_m_M": float(mean_params[1]),
        "L_m_M_err": float(errs[1]),
        "Q_m_M": float(mean_params[2]),
        "Q_m_M_err": float(errs[2]),
        "R_m_M": float(mean_params[3]),
        "R_m_M_err": float(errs[3]),
        "cov": cov,
        "chi2": final_chi2,
        "dof": final_dof,
        "bootstrap_meta": {
            "n_requested": int(len(bootstrap_point_sets)),
            "n_success": int(params.shape[0]),
            "n_failed": int(len(bootstrap_point_sets) - params.shape[0]),
            "n_rejected_outliers": int(np.sum(~keep_mask)),
            "mean_chi2": float(np.mean(chi2_values)),
            "sdev_chi2": float(np.std(chi2_values, ddof=1)) if len(chi2_values) > 1 else 0.0,
        },
        "bootstrap_samples": samples,
        "bootstrap_failures": failures,
    }
    return fit


def derive_wilson_start_parameters(linear_fit):
    return _shared_derive_wilson_start_parameters(linear_fit)


def fit_wilson_complete_model_linear(points):
    return _shared_fit_wilson_complete_model_linear(points)


def fit_wilson_complete_model_nonlinear(points, initial_fit):
    return _shared_fit_wilson_complete_model_nonlinear(
        points,
        initial_fit,
        fit_label=FPS_WILSON_PHYSICAL_LABEL,
    )


def wilson_physical_continuum_line_and_band(m_ps_sq, fit):
    return _shared_wilson_continuum_line_and_band(m_ps_sq, fit)


def make_wilson_fit_text(fit):
    text = (
        r"$\mathrm{Wilson\ (nonlinear)}:$" "\n"
        r"$f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$" "\n"
        r"$\qquad\qquad + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
    )
    text += "\n" + rf"$f_{{{{\rm PS}},\chi}}^2 = {fit['m_M_chi_sq']:.4f} \pm {fit['m_M_chi_sq_err']:.4f}$"
    text += "\n" + rf"$L_{{m_M}} = {fit['L_m_M']:.4f} \pm {fit['L_m_M_err']:.4f}$"
    text += "\n" + rf"$Q_{{m_M}} = {fit['Q_m_M']:.4f} \pm {fit['Q_m_M_err']:.4f}$"
    text += "\n" + rf"$W_{{m_M}} = {fit['W_m_M']:.4f} \pm {fit['W_m_M_err']:.4f}$"
    text += "\n" + rf"$R_{{m_M}} = {fit['R_m_M']:.4f} \pm {fit['R_m_M_err']:.4f}$"
    text += "\n" + rf"$C_{{m_M}} = {fit['C_m_M']:.4f} \pm {fit['C_m_M_err']:.4f}$"
    if fit["dof"] > 0:
        text += "\n" + rf"$\chi^2/\mathrm{{dof}} = {fit['chi2']:.2f}/{fit['dof']}$"
    return text


def print_wilson_fit_summary(fit, title):
    print(f"{title}:")
    print(f"  m_M_chi_sq = {fit['m_M_chi_sq']:.8g} ± {fit['m_M_chi_sq_err']:.3g}")
    print(f"  L_m_M = {fit['L_m_M']:.8g} ± {fit['L_m_M_err']:.3g}")
    print(f"  Q_m_M = {fit['Q_m_M']:.8g} ± {fit['Q_m_M_err']:.3g}")
    print(f"  W_m_M = {fit['W_m_M']:.8g} ± {fit['W_m_M_err']:.3g}")
    print(f"  R_m_M = {fit['R_m_M']:.8g} ± {fit['R_m_M_err']:.3g}")
    print(f"  C_m_M = {fit['C_m_M']:.8g} ± {fit['C_m_M_err']:.3g}")
    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


def print_starting_parameters(title, params):
    print(f"{title}:")
    for key, value in params.items():
        print(f"  {key} = {value:.8g}")


def physical_dw2_to_plot_fit(fit):
    a = float(fit["m_M_chi_sq"])
    l_val = float(fit["L_m_M"])
    q_val = float(fit["Q_m_M"])
    r_val = float(fit["R_m_M"])
    cov_phys = np.asarray(fit["cov"], dtype=float)

    coeff_b = a * l_val
    coeff_c = a * q_val
    jac = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [l_val, a, 0.0, 0.0],
            [q_val, 0.0, a, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cov_coeff = jac @ cov_phys @ jac.T
    errs = np.sqrt(np.diag(cov_coeff))

    return {
        "A": a,
        "A_err": float(errs[0]),
        "B": coeff_b,
        "B_err": float(errs[1]),
        "C": coeff_c,
        "C_err": float(errs[2]),
        "D": r_val,
        "D_err": float(errs[3]),
        "cov": cov_coeff,
        "chi2": float(fit["chi2"]),
        "dof": int(fit["dof"]),
        "L": float(l_val),
        "L_err": float(fit["L_m_M_err"]),
        "model_key": "dw2",
        "label": fit["label"],
        "label_plain": fit.get("label", ""),
    }


def plot_points_and_fits_bootstrap(
    dw_points,
    wilson_points,
    all_fits,
    plot_fit_keys,
    output_plot,
    dw2_fit_central=None,
):
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

    style_map = {
        "dw": {"color": "tab:green", "linestyle": (0, (5, 2, 1, 2)), "alpha_band": 0.12},
        "dw2": {"color": "tab:purple", "linestyle": "-", "alpha_band": 0.12},
        "wilson_physical": {"color": "tab:red", "linestyle": ":", "alpha_band": 0.10},
    }

    for fit_key in plot_fit_keys:
        fit = all_fits[fit_key]
        style = style_map[fit_key]
        if fit_key == "dw":
            y_fit, y_err = continuum_line_and_band_dw(x_grid, fit)
        elif fit_key == "dw2":
            y_fit, y_err = continuum_line_and_band_dw2(x_grid, fit)
        elif fit_key == "wilson_physical":
            y_fit, y_err = wilson_physical_continuum_line_and_band(x_grid, fit)
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

    if dw2_fit_central is not None and "dw2" in plot_fit_keys:
        y_central, _ = continuum_line_and_band_dw2(x_grid, dw2_fit_central)
        ax.plot(
            x_grid,
            y_central,
            linestyle="--",
            color="tab:purple",
            linewidth=1.0,
            alpha=0.85,
            label="MDWF central fit",
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


def linear_fit_to_json_dict(fit):
    out = {
        "model_key": fit["model_key"],
        "stage": fit["stage"],
        "label": fit["label"],
        "chi2": fit["chi2"],
        "dof": fit["dof"],
        "cov": fit["cov"],
    }
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
        "R_m_M",
        "R_m_M_err",
    ]:
        if key in fit:
            out[key] = fit[key]
    for key in ["W_m_M", "W_m_M_err", "C_m_M", "C_m_M_err"]:
        if key in fit:
            out[key] = fit[key]
    return to_serializable(out)


def bootstrap_fit_to_json_dict(fit):
    out = physical_fit_to_json_dict(fit)
    out["bootstrap_meta"] = fit["bootstrap_meta"]
    out["bootstrap_samples"] = fit["bootstrap_samples"]
    out["bootstrap_failures"] = fit["bootstrap_failures"]
    out["continuum_limit"] = {
        "fpsw0_sq": {
            "mean": fit["m_M_chi_sq"],
            "sdev": fit["m_M_chi_sq_err"],
        },
        "fpsw0": {
            "mean": float(np.sqrt(fit["m_M_chi_sq"])) if fit["m_M_chi_sq"] >= 0.0 else None,
            "sdev": (
                float(0.5 * fit["m_M_chi_sq_err"] / np.sqrt(fit["m_M_chi_sq"]))
                if fit["m_M_chi_sq"] > 0.0
                else None
            ),
        },
    }
    return to_serializable(out)


def save_fit_results_json(
    output_data,
    plot_fit_keys,
    dw_points,
    bootstrap_point_sets,
    dw_fit_linear,
    dw_fit_central,
    dw2_fit_bootstrap,
    wilson_points,
    wilson_fit_linear=None,
    wilson_fit_nonlinear=None,
):
    output_dir = os.path.dirname(output_data)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "plot_fits": plot_fit_keys,
        "n_dw_points_used": len(dw_points),
        "n_dw_bootstrap_requested": len(bootstrap_point_sets),
        "n_dw_bootstrap_success": dw2_fit_bootstrap["bootstrap_meta"]["n_success"],
        "n_dw_bootstrap_failed": dw2_fit_bootstrap["bootstrap_meta"]["n_failed"],
        "points": {
            "dw": to_serializable(dw_points),
            "wilson": to_serializable(wilson_points),
        },
        "fits": {
            "dw2": {
                "linearized": linear_fit_to_json_dict(dw_fit_linear),
                "starting_parameters": to_serializable(
                    derive_dw2_start_parameters(dw_fit_linear)
                ),
                "central_nonlinear": physical_fit_to_json_dict(dw_fit_central),
                "bootstrap_summary": bootstrap_fit_to_json_dict(dw2_fit_bootstrap),
            },
        },
    }

    if (
        wilson_points is not None
        and wilson_fit_linear is not None
        and wilson_fit_nonlinear is not None
    ):
        payload["n_wilson_points_used"] = len(wilson_points)
        payload["fits"]["wilson_physical"] = {
            "linearized": linear_fit_to_json_dict(wilson_fit_linear),
            "starting_parameters": to_serializable(
                derive_wilson_start_parameters(wilson_fit_linear)
            ),
            "nonlinear": physical_fit_to_json_dict(wilson_fit_nonlinear),
        }

    with open(output_data, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"✓ Saved fit data → {output_data}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap MDWF continuum fit for (f_PS w0)^2 vs (m_PS w0)^2. "
            "The MDWF points are built from bootstrap summaries of PP, Z_A, "
            "simultaneous PP+A0P, and w0, then each bootstrap replica is refit "
            "to the MDWF dw2 ansatz."
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
        default="",
        help="Optional Wilson spectrum table JSON",
    )
    parser.add_argument(
        "--wflow_w",
        default="",
        help="Optional Wilson ensemble/wflow table JSON",
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
        "--output_file",
        dest="output_plot",
        required=True,
        help="Output plot file",
    )
    parser.add_argument(
        "--output_data",
        required=True,
        help="Output JSON file storing fit results",
    )
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    dw_points, bootstrap_point_sets, bootstrap_input_failures = collect_dw_bootstrap_ensembles(
        args.spectrum,
        args.wflow,
    )

    dw_fit = fit_dw_continuum(dw_points)
    dw_fit_linear = fit_dw2_continuum_linear(dw_points)
    start_params = derive_dw2_start_parameters(dw_fit_linear)
    dw2_fit_central = fit_dw2_continuum_nonlinear(dw_points, dw_fit_linear)
    dw2_fit_bootstrap = fit_dw2_bootstrap_summary(
        bootstrap_point_sets, dw_points, dw2_fit_central, start_params
    )
    dw2_fit_bootstrap["bootstrap_failures"] = (
        bootstrap_input_failures + dw2_fit_bootstrap["bootstrap_failures"]
    )
    dw2_fit_bootstrap["bootstrap_meta"]["n_failed"] = len(
        dw2_fit_bootstrap["bootstrap_failures"]
    )

    wilson_points = []
    removed_first = []
    removed_last = []
    wilson_fit_linear = None
    wilson_fit = None
    if args.spectrum_w and args.wflow_w:
        wilson_points = collect_wilson_points(args.spectrum_w, args.wflow_w)
        wilson_points, removed_first, removed_last = exclude_wilson_endpoints(
            wilson_points,
            n_first=args.exclude_first_wilson,
            m_last=args.exclude_last_wilson,
            x_key="x",
        )
        print_removed_wilson_points(removed_first, which="beginning")
        print_removed_wilson_points(removed_last, which="end")

        wilson_fit_linear = fit_wilson_complete_model_linear(wilson_points)
        wilson_fit = fit_wilson_complete_model_nonlinear(
            wilson_points, wilson_fit_linear
        )

    plot_fit_keys = select_bootstrap_plot_fit_keys(has_wilson=wilson_fit is not None)

    all_fits = {
        "dw": dw_fit,
        "dw2": physical_dw2_to_plot_fit(dw2_fit_bootstrap),
    }
    if wilson_fit is not None:
        all_fits["wilson_physical"] = wilson_fit

    plot_points_and_fits_bootstrap(
        dw_points=dw_points,
        wilson_points=wilson_points,
        all_fits=all_fits,
        plot_fit_keys=plot_fit_keys,
        output_plot=args.output_plot,
        dw2_fit_central=physical_dw2_to_plot_fit(dw2_fit_central),
    )

    save_fit_results_json(
        output_data=args.output_data,
        plot_fit_keys=plot_fit_keys,
        dw_points=dw_points,
        bootstrap_point_sets=bootstrap_point_sets,
        dw_fit_linear=dw_fit_linear,
        dw_fit_central=dw2_fit_central,
        dw2_fit_bootstrap=dw2_fit_bootstrap,
        wilson_points=wilson_points,
        wilson_fit_linear=wilson_fit_linear,
        wilson_fit_nonlinear=wilson_fit,
    )

    print(f"✓ Saved plot → {args.output_plot}")
    print(f"Fits shown on plot: {plot_fit_keys}")
    print_dw_fit_summary(dw_fit)
    print_dw2_fit_summary(physical_dw2_to_plot_fit(dw2_fit_central))
    print_dw2_fit_summary(physical_dw2_to_plot_fit(dw2_fit_bootstrap))
    print_starting_parameters(
        "DWF/MDWF starting parameters from linearized fit",
        start_params,
    )
    if wilson_fit_linear is not None and wilson_fit is not None:
        print_starting_parameters(
            "Wilson starting parameters from linearized fit",
            derive_wilson_start_parameters(wilson_fit_linear),
        )
        print_wilson_fit_summary(wilson_fit, "Wilson complete model [nonlinear]")


if __name__ == "__main__":
    main()
