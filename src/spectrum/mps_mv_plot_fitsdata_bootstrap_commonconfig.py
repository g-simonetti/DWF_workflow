#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.ticker import ScalarFormatter

from mps_mv_plot_fitsdata_bootstrap import (
    bootstrap_fit_to_json_dict,
    fit_dw2_bootstrap_summary,
    make_dw2_bootstrap_fit_text,
    summary_stats,
)
from shared_continuum_models import (
    derive_dw2_start_parameters,
    derive_wilson_start_parameters,
    dw2_physical_continuum_line_and_band as continuum_line_and_band_dw2,
    fit_dw2_continuum_linear,
    fit_dw2_continuum_nonlinear,
    fit_wilson_complete_model_linear,
    fit_wilson_complete_model_nonlinear,
    wilson_physical_continuum_line_and_band as wilson_continuum_line_and_band,
)
from shared_fit_serialization import (
    linear_fit_to_json_dict,
    physical_fit_to_json_dict,
    to_serializable,
)
from spectrum import (
    fold_even,
    read_ps_corr,
    read_vx_corr,
    read_vy_corr,
    read_vz_corr,
)

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1]
_BOOTSTRAP_DIR = _SRC_DIR / "bootstrap"
_WFLOW_DIR = _SRC_DIR / "wflow"
for _path in (_BOOTSTRAP_DIR, _WFLOW_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bootstrap import bootstrap_from_path, bootstrap_to_jsonable  # noqa: E402
from ps_fit import fit_with_bootstrap_PP  # noqa: E402
from v_fit import fit_with_bootstrap_VV  # noqa: E402
from wflow import load_all_logs, numerical_derivative, solve_for_w0_squared  # noqa: E402


_NUM_PAT_PT = re.compile(r"^pt_ll\.(\d+)\.h5$")
_NUM_PAT_MR = re.compile(r"^mres\.(\d+)\.h5$")
DEFAULT_WILSON_SHARED_NONLINEAR_P0 = [0.320, 2.9, -20.0, -0.183, 0.03, -1.0]


def read_json_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as handle:
            text = handle.read()
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r"(?<=\})\s*(?=\{)", ",\n", text)
        try:
            return json.loads(repaired)
        except Exception as exc:
            raise ValueError(f"Could not read JSON file: {filename}\n{exc}") from exc
    except Exception as exc:
        raise ValueError(f"Could not read JSON file: {filename}\n{exc}") from exc


def parse_pair(obj, key, filename):
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


def read_wilson_spectrum_json(filename):
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


def extract_beta_mass_from_path(path):
    parts = Path(path).parts
    beta = None
    mass = None

    for part in parts:
        if part.startswith("B"):
            try:
                beta = float(part[1:])
            except ValueError:
                pass
        elif part.startswith("M") and mass is None:
            try:
                mass = float(part[1:])
            except ValueError:
                pass

    return beta, mass


def mw0_sq_and_error_from_w0a(am, am_err, w0a, w0a_err):
    x = (am * w0a) ** 2
    dx_dam = 2.0 * am * (w0a**2)
    dx_dw0a = 2.0 * (am**2) * w0a
    var = (dx_dam**2) * (am_err**2) + (dx_dw0a**2) * (w0a_err**2)
    return x, np.sqrt(var)


def a_over_w0_and_error(w0a, w0a_err):
    z = 1.0 / w0a
    z_err = w0a_err / (w0a**2)
    return z, z_err


def square_with_error(z, z_err):
    q = z**2
    q_err = 2.0 * abs(z) * z_err
    return q, q_err


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


def _require_key(obj, keys, filename):
    cur = obj
    for key in keys:
        if key not in cur:
            joined = ".".join(keys)
            raise ValueError(f"Missing key '{joined}' in '{filename}'")
        cur = cur[key]
    return cur


def _extract_num_from_name(path: str, kind: str) -> int:
    name = Path(path).name
    pat = _NUM_PAT_PT if kind == "pt" else _NUM_PAT_MR
    match = pat.match(name)
    if match is None:
        raise ValueError(f"Could not extract config number from filename '{name}'")
    return int(match.group(1))


def read_spectrum_common_json(filename):
    data = read_json_file(filename)
    return {
        "bootstrap": _require_key(data, ["bootstrap"], filename),
        "selection": _require_key(data, ["selection"], filename),
        "windows": _require_key(data, ["windows"], filename),
    }


def read_wflow_common_json(filename):
    data = read_json_file(filename)
    return {
        "bootstrap": _require_key(data, ["bootstrap", "w0"], filename),
        "inputs": _require_key(data, ["inputs"], filename),
    }


def read_precomputed_wilson_bootstrap_json(filename):
    data = read_json_file(filename)
    wilson_points = _require_key(data, ["points", "wilson"], filename)
    fit_block = _require_key(data, ["fits", "wilson_physical"], filename)
    return {
        "wilson_points": wilson_points,
        "linearized": _require_key(fit_block, ["linearized"], filename),
        "starting_parameters": _require_key(fit_block, ["starting_parameters"], filename),
        "central_nonlinear": fit_block.get("central_nonlinear"),
        "bootstrap_summary": _require_key(fit_block, ["bootstrap_summary"], filename),
    }


def determine_global_n_boot(spectrum_files, wflow_files):
    if len(spectrum_files) != len(wflow_files):
        raise ValueError("Number of --spectrum files must equal number of --wflow files.")

    n_values = []
    for spec_path, wflow_path in zip(spectrum_files, wflow_files):
        spec = read_spectrum_common_json(spec_path)
        wflow = read_wflow_common_json(wflow_path)
        n_values.append(int(spec["bootstrap"]["n_boot"]))
        n_values.append(int(wflow["bootstrap"]["n_boot"]))

    if not n_values:
        raise ValueError("No spectrum/wflow file pairs were provided.")
    return int(min(n_values))


def build_mesons_maps_from_selection(selection, spec_path):
    pt_map = {}
    mr_map = {}

    for path in selection.get("pt_files", []):
        num = _extract_num_from_name(path, "pt")
        pt_map[num] = Path(path)

    for path in selection.get("mres_files", []):
        num = _extract_num_from_name(path, "mr")
        mr_map[num] = Path(path)

    if not pt_map or not mr_map:
        raise RuntimeError(
            f"Missing pt_files or mres_files in selection metadata for '{spec_path}'"
        )

    return pt_map, mr_map


def build_wflow_samples_from_common(root_path, common_cfgs, n_boot, w0_reference):
    log_dir = Path(root_path) / "log"
    cfg_ids, t, t2e_conf, _q_conf = load_all_logs(str(log_dir))

    if len(set(cfg_ids.tolist())) != len(cfg_ids):
        raise RuntimeError(f"Duplicate cfg_ids found in Wilson-flow logs under '{log_dir}'")

    idx_map = {int(cfg): i for i, cfg in enumerate(cfg_ids.tolist())}
    missing = [cfg for cfg in common_cfgs if cfg not in idx_map]
    if missing:
        raise RuntimeError(
            f"Common cfgs missing from Wilson-flow logs for '{log_dir}': {missing[:10]}"
        )

    idx_sel = np.array([idx_map[cfg] for cfg in common_cfgs], dtype=int)

    w_conf = np.zeros_like(t2e_conf)
    for i in range(t2e_conf.shape[0]):
        dFdt = numerical_derivative(t2e_conf[i], t)
        w_conf[i] = t * dFdt

    w_sel = w_conf[idx_sel]
    w_mean = w_sel.mean(axis=0)
    w0_sq_central = solve_for_w0_squared(t, w_mean, float(w0_reference))
    w0_central = np.sqrt(w0_sq_central) if w0_sq_central > 0 else np.nan

    bootstrap = bootstrap_from_path(str(root_path), common_cfgs, n_boot)
    boot_idx = np.asarray(bootstrap["boot_idx"], dtype=int)

    w0_samples = []
    for b in range(n_boot):
        take = boot_idx[b]
        w_b = w_sel[take].mean(axis=0)
        try:
            w0_sq_b = solve_for_w0_squared(t, w_b, float(w0_reference))
            w0_b = np.sqrt(w0_sq_b) if w0_sq_b > 0 else np.nan
        except RuntimeError:
            w0_b = w0_central
        w0_samples.append(
            {
                "index": int(b),
                "w0": float(w0_b) if np.isfinite(w0_b) else None,
            }
        )

    return bootstrap, w0_samples


def build_spectrum_samples_from_common(root_path, spec_path, selection, windows, common_cfgs, n_boot):
    pt_map, mr_map = build_mesons_maps_from_selection(selection, spec_path)
    missing_pt = [cfg for cfg in common_cfgs if cfg not in pt_map]
    missing_mr = [cfg for cfg in common_cfgs if cfg not in mr_map]
    if missing_pt or missing_mr:
        raise RuntimeError(
            f"Common cfgs missing from stored spectrum selection for '{spec_path}'. "
            f"Missing pt: {missing_pt[:10]}, missing mres: {missing_mr[:10]}"
        )

    pt_sel = [pt_map[cfg] for cfg in common_cfgs]
    ps = []
    v = []
    for fpt in pt_sel:
        ps.append(read_ps_corr(str(fpt)))
        vx = read_vx_corr(str(fpt))
        vy = read_vy_corr(str(fpt))
        vz = read_vz_corr(str(fpt))
        v.append((vx + vy + vz) / 3.0)

    ps = np.asarray(ps, dtype=float)
    v = np.asarray(v, dtype=float)
    _n_cfg, t_full = ps.shape

    ps = fold_even(ps)
    v = fold_even(v)

    ps0 = int(windows["ps"]["t0"])
    ps1 = int(windows["ps"]["t1"])
    v0 = int(windows["v"]["t0"])
    v1 = int(windows["v"]["t1"])

    bootstrap = bootstrap_from_path(str(root_path), common_cfgs, n_boot)
    boot_idx = np.asarray(bootstrap["boot_idx"], dtype=int)

    pp_res = fit_with_bootstrap_PP(
        ps,
        ps0,
        ps1,
        Nt_full=t_full,
        n_boot=n_boot,
        boot_idx=boot_idx,
        svdcut=1e-8,
    )
    vv_res = fit_with_bootstrap_VV(
        v,
        v0,
        v1,
        Nt_full=t_full,
        n_boot=n_boot,
        boot_idx=boot_idx,
        svdcut=1e-8,
    )

    return bootstrap, pp_res["bootstrap_samples"], vv_res["bootstrap_samples"]


def build_dw_bootstrap_ensemble_common(spec_path, wflow_path, n_boot):
    beta, mass = extract_beta_mass_from_path(spec_path)
    if beta is None or mass is None:
        raise ValueError(f"Could not extract beta/mass from path: {spec_path}")

    spec = read_spectrum_common_json(spec_path)
    wflow = read_wflow_common_json(wflow_path)

    spec_bootstrap = spec["bootstrap"]
    wflow_bootstrap = wflow["bootstrap"]

    if spec_bootstrap.get("path_key") != wflow_bootstrap.get("path_key"):
        raise ValueError(
            "Spectrum/wflow path mismatch for\n"
            f"  spectrum: {spec_path}\n"
            f"  wflow:    {wflow_path}"
        )

    spec_cfgs = list(spec_bootstrap.get("cfg_numbers", []))
    wflow_cfgs = list(wflow_bootstrap.get("cfg_numbers", []))
    common_cfgs = sorted(set(spec_cfgs) & set(wflow_cfgs))
    if not common_cfgs:
        raise RuntimeError(
            "No common selected configurations between spectrum and wflow for\n"
            f"  spectrum: {spec_path}\n"
            f"  wflow:    {wflow_path}"
        )

    root_path = Path(spec_bootstrap["path_key"])
    w0_reference = float(wflow["inputs"]["W0_reference"])

    bootstrap, pp_samples, vv_samples = build_spectrum_samples_from_common(
        root_path=root_path,
        spec_path=spec_path,
        selection=spec["selection"],
        windows=spec["windows"],
        common_cfgs=common_cfgs,
        n_boot=n_boot,
    )
    bootstrap_w, w0_samples = build_wflow_samples_from_common(
        root_path=root_path,
        common_cfgs=common_cfgs,
        n_boot=n_boot,
        w0_reference=w0_reference,
    )

    if bootstrap_to_jsonable(bootstrap) != bootstrap_to_jsonable(bootstrap_w):
        raise RuntimeError(
            "Recomputed common-config bootstrap mismatch between spectrum and wflow for\n"
            f"  spectrum: {spec_path}\n"
            f"  wflow:    {wflow_path}"
        )

    replica_points = []
    for b in range(n_boot):
        pp_b = pp_samples[b]
        vv_b = vv_samples[b]
        w0_b = w0_samples[b]

        if pp_b is None or vv_b is None or w0_b is None:
            replica_points.append(None)
            continue

        m_ps = pp_b.get("m_ps")
        m_v = vv_b.get("m_v")
        w0_val = w0_b.get("w0")
        if m_ps is None or m_v is None or w0_val in (None, 0.0):
            replica_points.append(None)
            continue

        x = float((m_ps * w0_val) ** 2)
        y = float((m_v * w0_val) ** 2)
        a_over_w0 = float(1.0 / w0_val)
        replica_points.append(
            {
                "index": int(b),
                "beta": beta,
                "m0": mass,
                "x": x,
                "y": y,
                "a_over_w0": a_over_w0,
                "w0": float(w0_val),
                "m_ps": float(m_ps),
                "m_v": float(m_v),
            }
        )

    valid = [sample for sample in replica_points if sample is not None]
    if not valid:
        raise RuntimeError(f"No valid common-config bootstrap replicas for ensemble: {spec_path}")

    x_stats = summary_stats([sample["x"] for sample in valid])
    y_stats = summary_stats([sample["y"] for sample in valid])
    a_stats = summary_stats([sample["a_over_w0"] for sample in valid])

    point = {
        "beta": beta,
        "m0": mass,
        "x": x_stats["mean"],
        "xerr": x_stats["sdev"],
        "y": y_stats["mean"],
        "yerr": y_stats["sdev"],
        "a_over_w0": a_stats["mean"],
        "a_over_w0_err": a_stats["sdev"],
        "a_over_w0_sq": a_stats["mean"] ** 2,
        "a_over_w0_sq_err": 2.0 * abs(a_stats["mean"]) * a_stats["sdev"],
        "n_common_cfg": len(common_cfgs),
        "n_dropped_from_spectrum": len(set(spec_cfgs) - set(common_cfgs)),
        "n_dropped_from_wflow": len(set(wflow_cfgs) - set(common_cfgs)),
    }

    return {
        "point": point,
        "bootstrap_samples": replica_points,
        "bootstrap_meta": bootstrap_to_jsonable(bootstrap),
        "paths": {"spectrum": spec_path, "wflow": wflow_path},
        "common_config_filter": {
            "common_cfg_numbers": common_cfgs,
            "dropped_from_spectrum": sorted(set(spec_cfgs) - set(common_cfgs)),
            "dropped_from_wflow": sorted(set(wflow_cfgs) - set(common_cfgs)),
        },
    }


def collect_dw_bootstrap_ensembles_common(spectrum_files, wflow_files):
    if len(spectrum_files) != len(wflow_files):
        raise ValueError("Number of --spectrum files must equal number of --wflow files.")

    n_boot = determine_global_n_boot(spectrum_files, wflow_files)
    ensembles = [
        build_dw_bootstrap_ensemble_common(spec_path, wflow_path, n_boot)
        for spec_path, wflow_path in zip(spectrum_files, wflow_files)
    ]

    points = [entry["point"] for entry in ensembles]
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

    return points, bootstrap_point_sets, failures, ensembles, n_boot


def plot_points_and_fits_commonconfig(
    dw_points,
    dw_fit,
    output_file,
    wilson_points=None,
    wilson_fit=None,
    dw_fit_central=None,
):
    wilson_points = wilson_points or []
    all_betas = sorted(
        {p["beta"] for p in dw_points}.union({p["beta"] for p in wilson_points})
    )
    beta_colors = {b: f"C{i % 10}" for i, b in enumerate(all_betas)}

    fig, ax = plt.subplots(figsize=(8.4, 5.0), layout="constrained")

    for p in dw_points:
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
        )

    for p in wilson_points:
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
        )

    x_all = [p["x"] for p in dw_points]
    x_all.extend(p["x"] for p in wilson_points)
    x_grid = np.linspace(0.0, 1.05 * max(x_all), 400)

    if wilson_fit is not None and wilson_points:
        y_wilson, y_wilson_err = wilson_continuum_line_and_band(x_grid, wilson_fit)
        ax.plot(
            x_grid,
            y_wilson,
            linestyle=":",
            color="tab:red",
        )
        ax.fill_between(
            x_grid,
            y_wilson - y_wilson_err,
            y_wilson + y_wilson_err,
            color="tab:red",
            alpha=0.10,
            linewidth=0,
        )

    y_dw, y_dw_err = continuum_line_and_band_dw2(x_grid, dw_fit)
    ax.plot(
        x_grid,
        y_dw,
        linestyle="-",
        color="tab:purple",
    )
    ax.fill_between(
        x_grid,
        y_dw - y_dw_err,
        y_dw + y_dw_err,
        color="tab:purple",
        alpha=0.12,
        linewidth=0,
    )
    if dw_fit_central is not None:
        y_dw_central, _ = continuum_line_and_band_dw2(x_grid, dw_fit_central)
        ax.plot(
            x_grid,
            y_dw_central,
            linestyle="--",
            color="tab:purple",
            linewidth=1.0,
            alpha=0.85,
        )

    ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$")
    ax.set_ylabel(r"$(m_{\rm V} w_0)^2$")
    ax.set_xlim(0.0, float(x_grid[-1]))
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    wilson_betas = sorted({p["beta"] for p in wilson_points})
    dw_betas = sorted({p["beta"] for p in dw_points})

    mdwf_formula = (
        r"$m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
        "\n"
        r"$\qquad\qquad + R_{m_M} a^2$"
    )
    wilson_formula = (
        r"$m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
        "\n"
        r"$\qquad\qquad + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
    )

    def _legend_errorbar(color, marker, filled):
        face = color if filled else "none"
        return ax.errorbar(
            [np.nan],
            [np.nan],
            xerr=[1.0],
            yerr=[1.0],
            fmt=marker,
            linestyle="none",
            color=color,
            markerfacecolor=face,
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=4,
            elinewidth=0.6,
            capsize=1.5,
        )

    mdwf_handles = [
        Line2D([], [], linestyle="-", color="tab:purple", linewidth=1.1),
        Line2D([], [], linestyle="--", color="tab:purple", linewidth=1.0, alpha=0.85),
    ]
    mdwf_labels = [mdwf_formula, "Central-values fit"]
    for beta in dw_betas:
        mdwf_handles.append(_legend_errorbar(beta_colors[beta], "o", filled=True))
        mdwf_labels.append(rf"$\beta={beta}$")

    mdwf_legend = ax.legend(
        mdwf_handles,
        mdwf_labels,
        title="MDWF points: fitting model",
        loc="lower right",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.9,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=1.5,
        handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.35, yerr_size=0.35)},
    )
    mdwf_legend.get_frame().set_edgecolor("0.8")
    ax.add_artist(mdwf_legend)

    if wilson_fit is not None and wilson_points:
        wilson_handles = [
            Line2D([], [], linestyle=":", color="tab:red", linewidth=1.1),
        ]
        wilson_labels = [wilson_formula]
        for beta in wilson_betas:
            wilson_handles.append(_legend_errorbar(beta_colors[beta], "s", filled=False))
            wilson_labels.append(rf"$\beta={beta}$")
        wilson_legend = ax.legend(
            wilson_handles,
            wilson_labels,
            title="Wilson points: fitting model",
            loc="upper left",
            fontsize=9,
            title_fontsize=9,
            framealpha=0.9,
            borderpad=0.55,
            labelspacing=0.45,
            handlelength=1.5,
            handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.35, yerr_size=0.35)},
        )
        wilson_legend.get_frame().set_edgecolor("0.8")
        ax.add_artist(wilson_legend)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_file, dpi=300)
    plt.close()


def save_fit_results_json_common(
    output_data,
    dw_points,
    bootstrap_point_sets,
    dw_fit_linear,
    dw_fit_central,
    dw_fit_bootstrap,
    ensembles,
    n_boot,
    wilson_points=None,
    wilson_fit_linear=None,
    wilson_fit_nonlinear=None,
    wilson_fit_bootstrap=None,
    wilson_fit_starting_parameters=None,
):
    output_dir = Path(output_data).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_dw_points_used": len(dw_points),
        "n_dw_bootstrap_requested": len(bootstrap_point_sets),
        "n_dw_bootstrap_regenerated": int(n_boot),
        "n_dw_bootstrap_success": dw_fit_bootstrap["bootstrap_meta"]["n_success"],
        "n_dw_bootstrap_failed": dw_fit_bootstrap["bootstrap_meta"]["n_failed"],
        "common_config_filter": {
            "ensembles": to_serializable(
                [
                    {
                        "spectrum_json": entry["paths"]["spectrum"],
                        "wflow_json": entry["paths"]["wflow"],
                        **entry["common_config_filter"],
                    }
                    for entry in ensembles
                ]
            )
        },
        "points": {
            "dw": to_serializable(dw_points),
            "wilson": to_serializable(wilson_points or []),
        },
        "fits": {
            "dw2": {
                "linearized": linear_fit_to_json_dict(dw_fit_linear),
                "starting_parameters": to_serializable(
                    derive_dw2_start_parameters(dw_fit_linear)
                ),
                "central_nonlinear": physical_fit_to_json_dict(dw_fit_central),
                "bootstrap_summary": bootstrap_fit_to_json_dict(dw_fit_bootstrap),
            },
        },
    }

    if wilson_points is not None and wilson_fit_linear is not None:
        payload["n_wilson_points_used"] = len(wilson_points)
        wilson_payload = {
            "linearized": linear_fit_to_json_dict(wilson_fit_linear),
            "starting_parameters": to_serializable(
                wilson_fit_starting_parameters
                if wilson_fit_starting_parameters is not None
                else derive_wilson_start_parameters(wilson_fit_linear)
            ),
        }
        if wilson_fit_nonlinear is not None:
            wilson_payload["central_nonlinear"] = physical_fit_to_json_dict(
                wilson_fit_nonlinear
            )
        if wilson_fit_bootstrap is not None:
            wilson_payload["bootstrap_summary"] = to_serializable(wilson_fit_bootstrap)
        payload["fits"]["wilson_physical"] = wilson_payload

    with open(output_data, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"✓ Saved fit data → {output_data}")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Temporary bootstrap MDWF chiral-continuum fit that recomputes mPS, mV, and w0 "
            "using only the common configurations shared by spectrum and wflow."
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
        "--wilsons_data",
        default="intermediary_data/NF2/spectrum/wilson/wilson_extrapolation.json",
        help="Optional precomputed Wilson bootstrap JSON",
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

    dw_points, bootstrap_point_sets, bootstrap_input_failures, ensembles, n_boot = (
        collect_dw_bootstrap_ensembles_common(args.spectrum, args.wflow)
    )

    dw_fit_linear = fit_dw2_continuum_linear(dw_points)
    start_params = derive_dw2_start_parameters(dw_fit_linear)
    dw_fit_central = fit_dw2_continuum_nonlinear(dw_points, dw_fit_linear)
    dw_fit_bootstrap = fit_dw2_bootstrap_summary(
        bootstrap_point_sets,
        dw_points,
        dw_fit_central,
        start_params,
    )
    dw_fit_bootstrap["bootstrap_failures"] = (
        bootstrap_input_failures + dw_fit_bootstrap["bootstrap_failures"]
    )
    dw_fit_bootstrap["bootstrap_meta"]["n_failed"] = len(dw_fit_bootstrap["bootstrap_failures"])

    wilson_points = []
    wilson_fit_linear = None
    wilson_fit = None
    wilson_fit_bootstrap = None
    wilson_fit_starting_parameters = None
    wilson_fit_central = None
    wilsons_json_path = Path(args.wilsons_data) if args.wilsons_data else None
    if wilsons_json_path and wilsons_json_path.exists():
        wilson_data = read_precomputed_wilson_bootstrap_json(str(wilsons_json_path))
        wilson_points = wilson_data["wilson_points"]
        wilson_fit_linear = wilson_data["linearized"]
        wilson_fit = wilson_data["bootstrap_summary"]
        wilson_fit_bootstrap = wilson_data["bootstrap_summary"]
        wilson_fit_starting_parameters = wilson_data["starting_parameters"]
        wilson_fit_central = wilson_data["central_nonlinear"]
    elif args.spectrum_w and args.wflow_w:
        wilson_points = collect_wilson_points(args.spectrum_w, args.wflow_w)
        wilson_fit_linear = fit_wilson_complete_model_linear(wilson_points)
        wilson_fit = fit_wilson_complete_model_nonlinear(
            wilson_points,
            p0=DEFAULT_WILSON_SHARED_NONLINEAR_P0,
        )
        wilson_fit_starting_parameters = derive_wilson_start_parameters(wilson_fit_linear)
        wilson_fit_central = wilson_fit

    plot_points_and_fits_commonconfig(
        dw_points=dw_points,
        dw_fit=dw_fit_bootstrap,
        wilson_points=wilson_points,
        wilson_fit=wilson_fit,
        dw_fit_central=dw_fit_central,
        output_file=args.output_plot,
    )

    save_fit_results_json_common(
        output_data=args.output_data,
        dw_points=dw_points,
        bootstrap_point_sets=bootstrap_point_sets,
        dw_fit_linear=dw_fit_linear,
        dw_fit_central=dw_fit_central,
        dw_fit_bootstrap=dw_fit_bootstrap,
        ensembles=ensembles,
        n_boot=n_boot,
        wilson_points=wilson_points,
        wilson_fit_linear=wilson_fit_linear,
        wilson_fit_nonlinear=wilson_fit_central,
        wilson_fit_bootstrap=wilson_fit_bootstrap,
        wilson_fit_starting_parameters=wilson_fit_starting_parameters,
    )

    print(f"✓ Saved plot → {args.output_plot}")
    print_starting_parameters(
        "DWF/MDWF starting parameters from linearized fit",
        start_params,
    )
    if wilson_fit_linear is not None and wilson_fit_starting_parameters is not None:
        print_starting_parameters(
            "Wilson initial fit parameters",
            wilson_fit_starting_parameters,
        )
        if wilson_fit_central is not None:
            print_wilson_fit_summary(
                wilson_fit_central,
                "Wilson complete model [central-value fit]",
            )
        if wilson_fit_bootstrap is not None:
            print_wilson_fit_summary(
                wilson_fit_bootstrap,
                "Wilson complete model [bootstrap mean ± std]",
            )
        elif wilson_fit is not None:
            print_wilson_fit_summary(wilson_fit, "Wilson complete model")


if __name__ == "__main__":
    main()
