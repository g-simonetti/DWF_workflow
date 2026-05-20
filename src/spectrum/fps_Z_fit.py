import numpy as np

import gvar as gv
import gvar.dataset as ds
import corrfitter as cf
import lsqfit


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


def make_models_PP_A0P(T_data, T_period, t0, t1):
    return [
        cf.Corr2(
            datatag="PP",
            a="Afit", b="Afit", dE="m_ps",
            tp=+T_period,
            tdata=np.arange(T_data),
            tfit=np.arange(t0, t1 + 1),
        ),
        cf.Corr2(
            datatag="A0P",
            a="Afit", b="g", dE="m_ps",
            tp=-T_period,
            tdata=np.arange(T_data),
            tfit=np.arange(t0, t1 + 1),
        ),
    ]


def fit_with_bootstrap_PP_A0P(
    ps_samples,
    a0p_samples,
    t0,
    t1,
    Ns,
    Nt_full=None,
    n_boot=200,
    boot_idx=None,
    svdcut=1e-8,
):
    """
    Standard + bootstrap simultaneous correlated fit for PP + A0P using corrfitter.

    Nt_full:
        True temporal extent of the lattice. If None, assume the data arrays
        already span the full time extent.
    """
    ps_samples = np.asarray(ps_samples, dtype=float)
    a0p_samples = np.asarray(a0p_samples, dtype=float)

    if ps_samples.shape != a0p_samples.shape:
        raise ValueError(
            f"PP and A0P samples must have the same shape, got "
            f"{ps_samples.shape} and {a0p_samples.shape}"
        )
    if Ns <= 0:
        raise RuntimeError("--Ns must be > 0 (needed for f_PS normalisation).")

    _, T_data = ps_samples.shape

    if Nt_full is None:
        Nt_full = T_data

    dset = _dataset_from_samples({
        "PP": ps_samples,
        "A0P": a0p_samples,
    })
    data = ds.avg_data(dset)

    models = make_models_PP_A0P(T_data, Nt_full, t0, t1)
    prior = make_prior_PP_A0P(ps_samples, t0, t1)
    fitter = cf.CorrFitter(models=models)

    fit = fitter.lsqfit(prior=prior, data=data, svdcut=svdcut)

    m_ps_std = fit.p["m_ps"][0]
    g_std = fit.p["g"][0]
    fPS_std = -gv.sqrt(1.0) * g_std / gv.sqrt(m_ps_std) / (Ns ** 1.5)

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
            g_b = float(np.asarray(p["g"])[0])
            m_ps_gv_b = p_gv["m_ps"][0]
            Afit_gv_b = p_gv["Afit"][0]
            g_gv_b = p_gv["g"][0]
            f_ps_gv_b = -gv.sqrt(1.0) * g_gv_b / gv.sqrt(m_ps_gv_b) / (Ns ** 1.5)
            f_ps_b = float(gv.mean(f_ps_gv_b))
            m_ps_err_b = float(gv.sdev(m_ps_gv_b))
            Afit_err_b = float(gv.sdev(Afit_gv_b))
            g_err_b = float(gv.sdev(g_gv_b))
            f_ps_err_b = float(gv.sdev(f_ps_gv_b))
            chi2_b = float(bs_fit.chi2)

            bs.append("m_ps", [m_ps_b])
            bs.append("Afit", [Afit_b])
            bs.append("g", [g_b])
            bs.append("f_ps", [f_ps_b])
            bs_chi2.append(chi2_b)
            bootstrap_samples.append({
                "m_ps": m_ps_b,
                "m_ps_err": m_ps_err_b,
                "Afit": Afit_b,
                "Afit_err": Afit_err_b,
                "g": g_b,
                "g_err": g_err_b,
                "f_ps": f_ps_b,
                "f_ps_err": f_ps_err_b,
                "chi2": chi2_b,
            })
            n_success += 1
    else:
        boot_idx = np.asarray(boot_idx, dtype=int)
        n_requested = int(boot_idx.shape[0])
        for iboot, idx in enumerate(boot_idx):
            data_b = ds.avg_data(
                _dataset_from_samples({"PP": ps_samples[idx], "A0P": a0p_samples[idx]})
            )
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
                g_b = float(np.asarray(p["g"])[0])
                m_ps_gv_b = p_gv["m_ps"][0]
                Afit_gv_b = p_gv["Afit"][0]
                g_gv_b = p_gv["g"][0]
                f_ps_gv_b = -gv.sqrt(1.0) * g_gv_b / gv.sqrt(m_ps_gv_b) / (Ns ** 1.5)
                f_ps_b = float(gv.mean(f_ps_gv_b))
                m_ps_err_b = float(gv.sdev(m_ps_gv_b))
                Afit_err_b = float(gv.sdev(Afit_gv_b))
                g_err_b = float(gv.sdev(g_gv_b))
                f_ps_err_b = float(gv.sdev(f_ps_gv_b))
                chi2_b = float(bs_fit.chi2)

                bs.append("m_ps", [m_ps_b])
                bs.append("Afit", [Afit_b])
                bs.append("g", [g_b])
                bs.append("f_ps", [f_ps_b])
                bs_chi2.append(chi2_b)
                bootstrap_samples.append({
                    "index": iboot,
                    "m_ps": m_ps_b,
                    "m_ps_err": m_ps_err_b,
                    "Afit": Afit_b,
                    "Afit_err": Afit_err_b,
                    "g": g_b,
                    "g_err": g_err_b,
                    "f_ps": f_ps_b,
                    "f_ps_err": f_ps_err_b,
                    "chi2": chi2_b,
                })
                n_success += 1
            except Exception as exc:
                bootstrap_failures.append({"index": iboot, "error": str(exc)})
                bootstrap_samples.append(None)

    if n_success == 0:
        raise RuntimeError("All bootstrap simultaneous PP+A0P fits failed.")

    bs = ds.avg_data(bs, bstrap=True)

    return {
        "fit": fit,
        "fPS_fit_gvar": fPS_std,
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


def make_prior_PP_A0P(ps_samples, t0, t1):
    cmean = np.mean(ps_samples, axis=0)
    m0 = _guess_m_from_effmass(cmean, t0, t1)
    A0 = _guess_Afit_from_corr(cmean, m0, t_ref=t0)

    prior = gv.BufferDict()
    prior["log(Afit)"] = gv.gvar([np.log(A0)], [100.0])
    prior["log(m_ps)"] = gv.gvar([np.log(m0)], [100.0])
    prior["g"] = gv.gvar([0.0], [100.0])
    return prior


def compute_ZA(L, R):
    L = np.asarray(L, dtype=float)
    R = np.asarray(R, dtype=float)
    T = len(L)
    tvals = np.arange(1, T - 1)
    Z = np.zeros_like(tvals, dtype=float)

    for i, t in enumerate(tvals):
        Rf = R[t]
        Rb = R[t - 1]
        Lt = L[t]
        Ltp = L[t + 1]

        term1 = (Rf + Rb) / (2.0 * Lt)
        term2 = 2.0 * Rf / (Lt + Ltp)
        Z[i] = 0.5 * (term1 + term2)

    return tvals, Z


def ZA_from_ensemble_means(Lcorr, Rcorr):
    """
    Standard estimator built from ensemble-averaged correlators:
        Z_A(t) = Z_A( mean[L](t), mean[R](t) )

    This matches the residual-mass logic: nonlinear estimator applied
    after ensemble averaging, not per configuration first.
    """
    Lcorr = np.asarray(Lcorr, dtype=float)
    Rcorr = np.asarray(Rcorr, dtype=float)

    if Lcorr.shape != Rcorr.shape or Lcorr.ndim != 2:
        raise ValueError("ZA_from_ensemble_means expects Lcorr and Rcorr with shape (Ncfg, T)")

    Lbar = Lcorr.mean(axis=0)
    Rbar = Rcorr.mean(axis=0)
    return compute_ZA(Lbar, Rbar)


def bootstrap_ZA_replicas(Lcorr, Rcorr, n_boot, boot_idx):
    """
    Bootstrap replicas of the ratio-of-means Z_A estimator.

    For each bootstrap replica b:
        Lbar^(b)(t) = mean_k L_{b(k)}(t)
        Rbar^(b)(t) = mean_k R_{b(k)}(t)
        Z_A^(b)(t)  = Z_A(Lbar^(b), Rbar^(b))

    Returns:
      tvals   : t = 1..T-2
      Zb      : shape (n_boot, T-2)
    """
    Lcorr = np.asarray(Lcorr, dtype=float)
    Rcorr = np.asarray(Rcorr, dtype=float)

    if Lcorr.shape != Rcorr.shape or Lcorr.ndim != 2:
        raise ValueError("bootstrap_ZA_replicas expects Lcorr and Rcorr with shape (Ncfg, T)")

    tvals, _ = ZA_from_ensemble_means(Lcorr, Rcorr)
    nt = len(tvals)

    Zb = np.empty((n_boot, nt), dtype=float)

    for b in range(n_boot):
        idx = boot_idx[b]
        Lbar_b = Lcorr[idx].mean(axis=0)
        Rbar_b = Rcorr[idx].mean(axis=0)
        _, Zbar_b = compute_ZA(Lbar_b, Rbar_b)
        Zb[b] = Zbar_b

    return tvals, Zb


def bootstrap_ZA_curve(Lcorr, Rcorr, n_boot, boot_idx):
    """
    Bootstrap mean/std for plotting, using the ratio-of-means estimator.
    """
    tvals, Zb = bootstrap_ZA_replicas(Lcorr, Rcorr, n_boot, boot_idx)
    mean = Zb.mean(axis=0)
    std = Zb.std(axis=0, ddof=1)
    return tvals, mean, std


def fit_Z_only_bootstrap(Lcorr, Rcorr, t0, t1, n_boot, boot_idx, svdcut=1e-8):
    """
    Correlated constant fit to Z_A(t):

      1. Build the standard estimator from ensemble means:
             y = Z_A(mean[L], mean[R])

      2. Build bootstrap replicas:
             y^(b) = Z_A(mean_b[L], mean_b[R])

      3. Estimate covariance from the bootstrap ensemble.

      4. Fit a constant on [t0, t1].
    """
    tvals, y = ZA_from_ensemble_means(Lcorr, Rcorr)
    _, Zb = bootstrap_ZA_replicas(Lcorr, Rcorr, n_boot, boot_idx)

    mask = (tvals >= t0) & (tvals <= t1)
    if not np.any(mask):
        raise ValueError(f"fit_Z_only_bootstrap: empty fit window [{t0},{t1}] for t={tvals[0]}..{tvals[-1]}")

    y_win = np.asarray(y[mask], dtype=float)
    Zb_win = np.asarray(Zb[:, mask], dtype=float)

    # Bootstrap covariance
    cov = np.cov(Zb_win, rowvar=False, ddof=1)

    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    y_gv = gv.gvar(y_win, cov)

    prior = gv.BufferDict()
    prior["Z0"] = gv.gvar(1.0, 10.0)

    def fcn(p):
        return p["Z0"] * np.ones(len(y_win))

    fit = lsqfit.nonlinear_fit(data=y_gv, prior=prior, fcn=fcn, svdcut=svdcut)

    return {
        "fit": fit,
        "tvals": tvals,
        "ymean": y,
        "bootstrap_replicas": Zb,
        "mask": mask,
        "cov": cov,
    }


def compute_weighted_Z_samples(lcorr, rcorr, t0, t1, n_boot, boot_idx, svdcut=1e-8):
    z_res = fit_Z_only_bootstrap(
        lcorr,
        rcorr,
        t0,
        t1,
        n_boot=n_boot,
        boot_idx=boot_idx,
        svdcut=svdcut,
    )
    fit = z_res["fit"]
    zb = np.asarray(z_res["bootstrap_replicas"], dtype=float)
    mask = np.asarray(z_res["mask"], dtype=bool)
    cov = np.asarray(z_res["cov"], dtype=float)
    zb_win = zb[:, mask]

    prior = gv.BufferDict()
    prior["Z0"] = gv.gvar(1.0, 10.0)

    def fcn(p):
        return p["Z0"] * np.ones(zb_win.shape[1], dtype=float)

    z_samples = []
    for iboot in range(zb_win.shape[0]):
        try:
            yb_gv = gv.gvar(np.asarray(zb_win[iboot], dtype=float), cov)
            bs_fit = lsqfit.nonlinear_fit(
                data=yb_gv,
                prior=prior,
                fcn=fcn,
                svdcut=svdcut,
                p0=fit.pmean,
            )
            z0_gv = bs_fit.p["Z0"]
            z_samples.append(
                {
                    "index": int(iboot),
                    "Z_A": float(gv.mean(z0_gv)),
                    "Z_A_err": float(gv.sdev(z0_gv)),
                    "chi2": float(bs_fit.chi2),
                }
            )
        except Exception as exc:
            z_samples.append(
                {
                    "index": int(iboot),
                    "error": str(exc),
                }
            )

    return fit, z_samples
