#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fps_mps_plot import (
    collect_wilson_points,
    exclude_wilson_endpoints,
    extract_beta_mass_from_path,
    fit_dw_continuum,
    print_dw2_fit_summary,
    print_dw_fit_summary,
    print_removed_wilson_points,
    read_json_file,
    to_serializable,
)
from fps_mps_plot_fitsdata_bootstrap import (
    bootstrap_fit_to_json_dict,
    derive_dw2_start_parameters,
    derive_wilson_start_parameters,
    fit_dw2_bootstrap_summary,
    fit_dw2_continuum_linear,
    fit_dw2_continuum_nonlinear,
    fit_wilson_complete_model_linear,
    fit_wilson_complete_model_nonlinear,
    linear_fit_to_json_dict,
    plot_points_and_fits_bootstrap,
    physical_dw2_to_plot_fit,
    physical_fit_to_json_dict,
    print_starting_parameters,
    print_wilson_fit_summary,
    select_bootstrap_plot_fit_keys,
    summary_stats,
)
from spectrum import (
    fold_even,
    fold_odd,
    read_L2_corr,
    read_L_corr,
    read_ps_corr,
    read_R_corr,
)

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1]
_BOOTSTRAP_DIR = _SRC_DIR / "bootstrap"
_WFLOW_DIR = _SRC_DIR / "wflow"
for _path in (_BOOTSTRAP_DIR, _WFLOW_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bootstrap import bootstrap_from_path, bootstrap_to_jsonable  # noqa: E402
from fps_Z_fit import compute_weighted_Z_samples, fit_with_bootstrap_PP_A0P  # noqa: E402
from ps_fit import fit_with_bootstrap_PP  # noqa: E402
from wflow import load_all_logs, numerical_derivative, solve_for_w0_squared  # noqa: E402

plt.style.use("tableau-colorblind10")

_NUM_PAT_PT = re.compile(r"^pt_ll\.(\d+)\.h5$")
_NUM_PAT_MR = re.compile(r"^mres\.(\d+)\.h5$")


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


def _extract_ns_from_path(path: str) -> int:
    for part in Path(path).parts:
        if part.startswith("Ns"):
            return int(part[2:])
    raise ValueError(f"Could not extract Ns from path: {path}")


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
    mr_sel = [mr_map[cfg] for cfg in common_cfgs]

    ps = []
    a0p = []
    ls = []
    rs = []
    for fpt, fmr in zip(pt_sel, mr_sel):
        ps.append(read_ps_corr(str(fpt)))
        a0_33 = read_L_corr(str(fpt))
        a0_9 = read_L2_corr(str(fpt))
        a0_comb = 0.5 * (a0_33 + a0_9)
        a0p.append(a0_comb)
        ls.append(a0_comb)
        rs.append(read_R_corr(str(fmr)))

    ps = np.asarray(ps, dtype=float)
    a0p = np.asarray(a0p, dtype=float)
    ls = np.asarray(ls, dtype=float)
    rs = np.asarray(rs, dtype=float)
    _n_cfg, t_full = ps.shape

    ps = fold_even(ps)
    a0p = fold_odd(a0p)

    ps0 = int(windows["ps"]["t0"])
    ps1 = int(windows["ps"]["t1"])
    fps0 = int(windows["fps"]["t0"])
    fps1 = int(windows["fps"]["t1"])
    z0 = int(windows["Z"]["t0"])
    z1 = int(windows["Z"]["t1"])
    ns = _extract_ns_from_path(spec_path)

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
    sim_res = fit_with_bootstrap_PP_A0P(
        ps,
        a0p,
        fps0,
        fps1,
        Ns=ns,
        Nt_full=t_full,
        n_boot=n_boot,
        boot_idx=boot_idx,
        svdcut=1e-8,
    )
    _fit_z, z_samples = compute_weighted_Z_samples(
        ls,
        rs,
        z0,
        z1,
        n_boot=n_boot,
        boot_idx=boot_idx,
        svdcut=1e-8,
    )

    return bootstrap, pp_res["bootstrap_samples"], sim_res["bootstrap_samples"], z_samples


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

    bootstrap, pp_samples, sim_samples, z_samples = build_spectrum_samples_from_common(
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
        sim_b = sim_samples[b]
        z_b = z_samples[b]
        w0_b = w0_samples[b]

        if pp_b is None or sim_b is None or z_b is None or w0_b is None:
            replica_points.append(None)
            continue

        m_ps = pp_b.get("m_ps")
        f_ps = sim_b.get("f_ps")
        z_a = z_b.get("Z_A")
        w0_val = w0_b.get("w0")
        if m_ps is None or f_ps is None or z_a is None or w0_val in (None, 0.0):
            replica_points.append(None)
            continue

        fps = float(z_a) * float(f_ps)
        x = float((float(m_ps) * float(w0_val)) ** 2)
        y = float((fps * float(w0_val)) ** 2)
        a_over_w0 = float(1.0 / float(w0_val))
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
                "f_ps": float(f_ps),
                "Z_A": float(z_a),
                "fps": fps,
            }
        )

    valid = [sample for sample in replica_points if sample is not None]
    if not valid:
        raise RuntimeError(f"No valid common-config bootstrap replicas for ensemble: {spec_path}")

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

    return points, bootstrap_point_sets, failures, ensembles, n_boot


def save_fit_results_json_common(
    output_data,
    plot_fit_keys,
    dw_points,
    bootstrap_point_sets,
    dw_fit_linear,
    dw2_fit_central,
    dw2_fit_bootstrap,
    ensembles,
    n_boot,
    wilson_points,
    removed_first,
    removed_last,
    wilson_fit_linear,
    wilson_fit_nonlinear,
):
    output_dir = Path(output_data).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "plot_fits": plot_fit_keys,
        "n_dw_points_used": len(dw_points),
        "n_dw_bootstrap_requested": len(bootstrap_point_sets),
        "n_dw_bootstrap_regenerated": int(n_boot),
        "n_dw_bootstrap_success": dw2_fit_bootstrap["bootstrap_meta"]["n_success"],
        "n_dw_bootstrap_failed": dw2_fit_bootstrap["bootstrap_meta"]["n_failed"],
        "n_wilson_points_used": len(wilson_points),
        "excluded_wilson": {
            "n_first": len(removed_first),
            "n_last": len(removed_last),
            "removed_first": to_serializable(removed_first),
            "removed_last": to_serializable(removed_last),
        },
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
            "wilson": to_serializable(wilson_points),
        },
        "fits": {
            "dw2": {
                "linearized": linear_fit_to_json_dict(dw_fit_linear),
                "starting_parameters": to_serializable(
                    derive_dw2_start_parameters(dw_fit_linear)
                ),
                "central_nonlinear": physical_fit_to_json_dict(dw2_fit_central),
                "bootstrap_summary": bootstrap_fit_to_json_dict(dw2_fit_bootstrap),
            },
        },
    }

    if (
        wilson_points is not None
        and wilson_fit_linear is not None
        and wilson_fit_nonlinear is not None
    ):
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
            "Bootstrap MDWF continuum fit for (f_PS w0)^2 vs (m_PS w0)^2, "
            "recomputed on the common configuration subset shared by spectrum and wflow."
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

    dw_points, bootstrap_point_sets, bootstrap_input_failures, ensembles, n_boot = (
        collect_dw_bootstrap_ensembles_common(args.spectrum, args.wflow)
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

    save_fit_results_json_common(
        output_data=args.output_data,
        plot_fit_keys=plot_fit_keys,
        dw_points=dw_points,
        bootstrap_point_sets=bootstrap_point_sets,
        dw_fit_linear=dw_fit_linear,
        dw2_fit_central=dw2_fit_central,
        dw2_fit_bootstrap=dw2_fit_bootstrap,
        ensembles=ensembles,
        n_boot=n_boot,
        wilson_points=wilson_points,
        removed_first=removed_first,
        removed_last=removed_last,
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
