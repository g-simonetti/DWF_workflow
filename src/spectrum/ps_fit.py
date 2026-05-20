import numpy as np

import gvar as gv
import gvar.dataset as ds
import corrfitter as cf


def eff_mass_hyperbolic(C):
    """
    Effective mass from:
        cosh(m_eff(t)) = (C(t-1) + C(t+1)) / (2 C(t)).
    Defined for t = 1..T-2.
    """
    C = np.asarray(C)
    T = len(C)
    t = np.arange(1, T - 1)
    R = (C[t - 1] + C[t + 1]) / (2 * C[t])
    R = np.maximum(R, 1.0)
    return t, np.arccosh(R)


def _guess_m_from_effmass(cmean, t0, t1):
    teff, meff = eff_mass_hyperbolic(cmean)
    mask = (teff >= t0 + 1) & (teff <= t1 - 1)
    m0 = float(np.mean(meff[mask])) if np.any(mask) else 0.1
    if not np.isfinite(m0) or m0 <= 0:
        m0 = 0.1
    return m0


def _guess_Afit_from_corr(cmean, m0, t_ref):
    """
    For symmetric correlator:
      C(t) ~ Afit^2 * exp(-m t)  (ignoring backward term)
    So Afit ~ sqrt( |C(t)| * exp(m t) ).
    """
    t_ref = int(t_ref)
    t_ref = max(0, min(t_ref, len(cmean) - 1))
    C = float(cmean[t_ref])
    A2 = abs(C) * np.exp(m0 * t_ref)
    return float(np.sqrt(max(A2, 1e-16)))


def _dataset_from_samples(sample_map):
    """
    Build a gvar.dataset.Dataset from arrays with shape (Ncfg, T).

    Each configuration is appended as one Monte Carlo sample, matching the
    corrfitter/gvar.dataset workflow more closely.
    """
    keys = list(sample_map.keys())
    arrays = {}

    ref_shape = None
    for k, v in sample_map.items():
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{k} must have shape (Ncfg, T), got {arr.shape}")
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f"All sample arrays must have the same shape; "
                f"{k} has {arr.shape}, expected {ref_shape}"
            )
        arrays[k] = arr

    ncfg, _ = ref_shape
    dset = ds.Dataset()
    for icfg in range(ncfg):
        for k in keys:
            dset.append(k, arrays[k][icfg])
    return dset


def _finite_or_none(x: float):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _chi2_over_dof(chi2: float, dof: int):
    if dof is None or dof <= 0:
        return None
    val = float(chi2) / float(dof)
    return _finite_or_none(val)


def _bootstrap_fit_stats(chi2_samples, dof: int):
    """
    Bootstrap analogue of standard fit_stats.

    Since bootstrap fits produce an ensemble of chi2 values, chi2 and
    chi2_over_dof are reported as {mean, sdev}. Q and logGBF are null.
    """
    chi2_samples = np.asarray(chi2_samples, dtype=float)
    chi2_samples = chi2_samples[np.isfinite(chi2_samples)]

    if chi2_samples.size == 0:
        return {
            "chi2": None,
            "dof": int(dof),
            "chi2_over_dof": None,
            "Q": None,
            "logGBF": None,
        }

    chi2_mean = float(np.mean(chi2_samples))
    chi2_sdev = float(np.std(chi2_samples, ddof=1)) if chi2_samples.size >= 2 else 0.0

    if dof is not None and dof > 0:
        chi2_over_dof = {
            "mean": chi2_mean / float(dof),
            "sdev": chi2_sdev / float(dof),
        }
    else:
        chi2_over_dof = None

    return {
        "chi2": {"mean": chi2_mean, "sdev": chi2_sdev},
        "dof": int(dof),
        "chi2_over_dof": chi2_over_dof,
        "Q": None,
        "logGBF": None,
    }


def make_models_PP(T_data, T_period, t0, t1):
    return [
        cf.Corr2(
            datatag="PP",
            a="Afit", b="Afit", dE="m_ps",
            tp=+T_period,
            tdata=np.arange(T_data),
            tfit=np.arange(t0, t1 + 1),
        )
    ]


def make_prior_PP(ps_samples, t0, t1):
    cmean = np.mean(ps_samples, axis=0)
    m0 = _guess_m_from_effmass(cmean, t0, t1)
    A0 = _guess_Afit_from_corr(cmean, m0, t_ref=t0)

    prior = gv.BufferDict()
    prior["log(Afit)"] = gv.gvar([np.log(A0)], [100.0])
    prior["log(m_ps)"] = gv.gvar([np.log(m0)], [100.0])
    return prior

    

def fit_with_bootstrap_PP(
    ps_samples,
    t0,
    t1,
    Nt_full=None,
    n_boot=200,
    boot_idx=None,
    svdcut=1e-8,
):
    ps_samples = np.asarray(ps_samples, dtype=float)
    _, T_data = ps_samples.shape

    if Nt_full is None:
        Nt_full = T_data

    dset = _dataset_from_samples({"PP": ps_samples})
    data = ds.avg_data(dset)

    models = make_models_PP(T_data, Nt_full, t0, t1)
    prior = make_prior_PP(ps_samples, t0, t1)
    fitter = cf.CorrFitter(models=models)

    fit = fitter.lsqfit(prior=prior, data=data, svdcut=svdcut)

    bs = ds.Dataset()
    bs_chi2 = []
    bootstrap_samples = []
    bootstrap_failures = []
    n_success = 0
    if boot_idx is None:
        n_requested = int(n_boot)
        bs_datalist = (ds.avg_data(d) for d in ds.bootstrap_iter(dset, n_requested))

        for bs_fit in fitter.bootstrapped_fit_iter(datalist=bs_datalist):
            p = bs_fit.pmean
            p_gv = bs_fit.p
            m_ps_b = float(np.asarray(p["m_ps"])[0])
            Afit_b = float(np.asarray(p["Afit"])[0])
            m_ps_err_b = float(gv.sdev(p_gv["m_ps"][0]))
            Afit_err_b = float(gv.sdev(p_gv["Afit"][0]))
            chi2_b = float(bs_fit.chi2)

            bs.append("m_ps", [m_ps_b])
            bs.append("Afit", [Afit_b])
            bs_chi2.append(chi2_b)
            bootstrap_samples.append({
                "m_ps": m_ps_b,
                "m_ps_err": m_ps_err_b,
                "Afit": Afit_b,
                "Afit_err": Afit_err_b,
                "chi2": chi2_b,
            })
            n_success += 1
    else:
        boot_idx = np.asarray(boot_idx, dtype=int)
        n_requested = int(boot_idx.shape[0])
        for iboot, idx in enumerate(boot_idx):
            data_b = ds.avg_data(_dataset_from_samples({"PP": ps_samples[idx]}))
            try:
                bs_fit = fitter.lsqfit(
                    prior=prior,
                    data=data_b,
                    svdcut=svdcut,
                    p0=fit.pmean,
                )
                p = bs_fit.pmean
                p_gv = bs_fit.p
                m_ps_b = float(np.asarray(p["m_ps"])[0])
                Afit_b = float(np.asarray(p["Afit"])[0])
                m_ps_err_b = float(gv.sdev(p_gv["m_ps"][0]))
                Afit_err_b = float(gv.sdev(p_gv["Afit"][0]))
                chi2_b = float(bs_fit.chi2)

                bs.append("m_ps", [m_ps_b])
                bs.append("Afit", [Afit_b])
                bs_chi2.append(chi2_b)
                bootstrap_samples.append({
                    "index": iboot,
                    "m_ps": m_ps_b,
                    "m_ps_err": m_ps_err_b,
                    "Afit": Afit_b,
                    "Afit_err": Afit_err_b,
                    "chi2": chi2_b,
                })
                n_success += 1
            except Exception as exc:
                bootstrap_failures.append({"index": iboot, "error": str(exc)})
                bootstrap_samples.append(None)

    if n_success == 0:
        raise RuntimeError("All bootstrap PP fits failed.")

    bs = ds.avg_data(bs, bstrap=True)

    return {
        "fit": fit,
        "bootstrap": bs,
        "bootstrap_fit_stats": _bootstrap_fit_stats(bs_chi2, int(fit.dof)),
        "bootstrap_meta": {
            "n_requested": int(n_requested),
            "n_success": int(n_success),
            "n_failed": int(n_requested - n_success),
        },
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_failures": bootstrap_failures,
    }
