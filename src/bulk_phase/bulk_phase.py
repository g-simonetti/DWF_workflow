#!/usr/bin/env python3
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="Bulk phase plotting tool (JSON inputs).")
parser.add_argument("--ensembles_csv", required=True)
parser.add_argument("--plaq_avg", nargs="+", required=True, help="log_hmc_extract.json files (dyn + YM).")
parser.add_argument("--mres_data", nargs="*", default=[], help="m_res.json files (Shamir only).")
parser.add_argument("--label", default="no")
parser.add_argument("--plot_styles", default=None)
parser.add_argument("--tuned_masses", required=True)
parser.add_argument("--tuned_history", required=True)
parser.add_argument("--shamir_summary", required=True)
parser.add_argument("--history_masses", nargs="*", type=float, default=[0.01, 0.10])
args = parser.parse_args()

show_legend = str(args.label).strip().lower() == "yes"
plt.style.use(args.plot_styles if args.plot_styles else "tableau-colorblind10")


# ============================================================
# HELPERS
# ============================================================
def is_true(x) -> bool:
    return str(x).strip().upper() in {"TRUE", "T", "1", "YES", "Y"}

def sfloat(x): return str(float(str(x).strip()))
def sint(x): return str(int(float(str(x).strip())))

def make_dyn_key(NF, Nt, Ns, Ls, beta, mass, mpv, alpha, a5, M5):
    return (sint(NF), sint(Nt), sint(Ns), sint(Ls),
            sfloat(beta), sfloat(mass), sfloat(mpv), sfloat(alpha), sfloat(a5), sfloat(M5))


# ============================================================
# PATH PARSING 
# ============================================================
FLOAT_TOKEN = r"[0-9]+(?:\.[0-9]+)?"
PAT_DYN = re.compile(
    r".*/NF(?P<NF>\d+)/Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>{FT})/M(?P<mass>{FT})/mpv(?P<mpv>{FT})/"
    r"alpha(?P<alpha>{FT})/a5(?P<a5>{FT})/M5(?P<M5>{FT})/.*".format(FT=FLOAT_TOKEN)
)
PAT_YM = re.compile(
    r".*/NF(?P<NF>\d+)/Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/B(?P<beta>{FT})/.*".format(FT=FLOAT_TOKEN)
)

def parse_info(fp):
    s = str(fp)
    m = PAT_DYN.match(s)
    if m:
        gd = m.groupdict()
        key = make_dyn_key(**gd)
        return "dyn", key, gd
    m = PAT_YM.match(s)
    if m:
        gd = m.groupdict()
        key = (sint(gd["Nt"]), sint(gd["Ns"]), sfloat(gd["beta"]))
        return "ym", key, gd
    return None, None, None


# ============================================================
# JSON LOADERS
# ============================================================
def load_hmc_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def load_plaq_avg_err_from_hmc_json(path: str):
    """
    log_hmc_extract.json expected:
      data["hmc_extract"]["plaq"], data["hmc_extract"]["plaq_err"]
    """
    data = load_hmc_json(path)
    try:
        h = data["hmc_extract"]
        return float(h["plaq"]), float(h.get("plaq_err", 0.0))
    except Exception as e:
        return np.nan, np.nan

def load_plaq_history_from_hmc_json(path: str):
    """
    log_hmc_extract.json expected (one of these formats):
      A) data["plaq_history"]["t"], data["plaq_history"]["plaq"]
      B) data["plaq_history"]["mc_time"], data["plaq_history"]["plaq"]
      C) data["plaq_history"] = {"t": [...], "plaq": [...]}
    Returns (t, plaq) or (None, None) if missing.
    """
    data = load_hmc_json(path)
    ph = data.get("plaq_history", None)
    if not isinstance(ph, dict):
        return None, None

    # accept either "t" or "mc_time"
    t = ph.get("t", ph.get("mc_time", None))
    p = ph.get("plaq", None)
    if t is None or p is None:
        return None, None

    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if t.size == 0 or p.size == 0:
        return None, None
    n = min(t.size, p.size)
    return t[:n], p[:n]

def load_mres_extract_from_mres_json(path: str):
    """
    m_res.json expected:
      data["mres_extract"]["value"], data["mres_extract"]["error"]
    """
    with open(path, "r") as f:
        data = json.load(f)
    try:
        return float(data["mres_extract"]["value"]), float(data["mres_extract"]["error"])
    except Exception:
        return np.nan, np.nan


# ============================================================
# METADATA MAP 
# ============================================================
df_meta = pd.read_csv(args.ensembles_csv, sep=r"\t|,", engine="python")

meta_map = {}
for _, r in df_meta.iterrows():
    try:
        if float(r["NF"]) > 0:
            k = make_dyn_key(r["NF"], r["Nt"], r["Ns"], r["Ls"], r["beta"],
                             r["mass"], r["mpv"], r["alpha"], r["a5"], r["M5"])
            meta_map[k] = r
    except Exception:
        continue


# ============================================================
# AGGREGATION 
# ============================================================
e_tuned, e_shamir, e_ym, e_mres = [], [], [], []

for fp in args.plaq_avg:
    kind, key, g = parse_info(fp)
    if kind == "ym":
        e_ym.append({"beta": float(g["beta"]), "path": fp})
        continue

    if kind == "dyn" and key in meta_map:
        m = meta_map[key]
        beta = float(g["beta"])
        mass = float(g["mass"])

        if is_true(m.get("use_in_bulkphase_tuned", False)):
            e_tuned.append({"key": key, "beta": beta, "mass": mass, "path": fp})

        if is_true(m.get("use_in_bulkphase_Shamir", False)):
            e_shamir.append({"key": key, "beta": beta, "mass": mass, "path": fp})

for fp in args.mres_data:
    kind, key, g = parse_info(fp)
    if kind == "dyn" and key in meta_map and is_true(meta_map[key].get("use_in_bulkphase_mres", False)):
        e_mres.append({"beta": float(g["beta"]), "mass": float(g["mass"]), "path": fp})


# Color maps as before
all_betas = sorted(set([e["beta"] for e in e_tuned + e_shamir + e_ym]))
beta_cmap = dict(zip(all_betas, mpl.cm.viridis_r(np.linspace(0.1, 1.0, max(1, len(all_betas))))))
markers = ["^", "v", "<", ">", "D", "P", "X"]

all_masses = sorted(set([e["mass"] for e in (e_tuned + e_shamir + e_mres)]))
m_colors = mpl.cm.inferno(np.linspace(0.1, 0.85, max(1, len(all_masses))))
mass_cmap = {m: m_colors[i] for i, m in enumerate(all_masses)}

def _mass_str(entries_panel) -> str:
    masses = sorted({f"{e['mass']:.8g}" for e in entries_panel})
    return masses[0] if len(masses) == 1 else ",".join(masses)


# ============================================================
# FIG 1: Tuned Möbius <P> vs mass
# ============================================================
fig1, ax1 = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

for i, b in enumerate(sorted(set(e["beta"] for e in e_tuned))):
    pts = []
    for e in [x for x in e_tuned if x["beta"] == b]:
        v, er = load_plaq_avg_err_from_hmc_json(e["path"])
        if np.isfinite(v):
            pts.append((e["mass"], v, er))
    if pts:
        xs, ys, ye = zip(*sorted(pts))
        ax1.errorbar(xs, ys, yerr=ye, fmt=markers[i % len(markers)], ls=":",
                    color=beta_cmap[b], label=rf"$\beta={b}$")

ax1.set_xlabel(r"$am_0$")
ax1.set_ylabel(r"$\langle \mathcal{P} \rangle$")
handles, labels = ax1.get_legend_handles_labels()
if handles:
    ax1.legend(handles[::-1], labels[::-1], fontsize="x-small", bbox_to_anchor=(1, 0.9))
fig1.savefig(args.tuned_masses, dpi=300)


# ============================================================
# FIG 2: Histories 
# ============================================================
h_mass_list = sorted(args.history_masses)
h_betas = sorted(set(e["beta"] for e in e_tuned))

fig2, axes2 = plt.subplots(max(1, len(h_betas)), 1, figsize=(3.5, 2.5), sharex=True, layout="constrained")
if len(h_betas) == 1:
    axes2 = [axes2]

for ax, b in zip(axes2[::-1], h_betas):
    group = sorted(
        [e for e in e_tuned if np.isclose(e["beta"], b) and any(np.isclose(e["mass"], m) for m in h_mass_list)],
        key=lambda x: x["mass"]
    )

    for j, e in enumerate(group):
        t, p = load_plaq_history_from_hmc_json(e["path"])
        if t is None or p is None:
            continue
        ls = "-" if j == 0 else (":" if j == len(group) - 1 else "-")
        ax.plot(t, p, color=mass_cmap[e["mass"]], ls=ls, alpha=0.7, label=rf"$am_0={e['mass']}$")

        # --- simple auto ylim (robust) ---
        y = np.hstack([line.get_ydata() for line in ax.lines])
        lo, hi = np.percentile(y, [1, 99])   # change to [1, 99] if you want less zoom
        pad = 0.5 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_ylabel(rf"$ \mathcal{{P}} [\beta={b}]$")
    if show_legend:
        ax.legend(loc="upper right", fontsize="x-small", ncol=2)

axes2[-1].set_xlabel("Monte Carlo time")
axes2[-1].set_xlim(150, 6900)
fig2.savefig(args.tuned_history, dpi=300)


# ============================================================
# FIG 3: Shamir Summary
# ============================================================
s_betas = sorted(set(e["beta"] for e in e_shamir))
s_masses = sorted(set(e["mass"] for e in e_shamir))
fig3 = plt.figure(figsize=(7, 3.5), layout="constrained")
gs = fig3.add_gridspec(2, 2, width_ratios=[1, 1])

# Left: Plaq vs Mass (with YM reference)
sax_left = fig3.add_subplot(gs[:, 0])
for i, b in enumerate(s_betas):
    pts = []
    for e in [x for x in e_shamir if np.isclose(x["beta"], b)]:
        v, er = load_plaq_avg_err_from_hmc_json(e["path"])
        if np.isfinite(v):
            pts.append((e["mass"], v, er))
    if pts:
        xs, ys, ye = zip(*sorted(pts))
        sax_left.errorbar(xs, ys, yerr=ye, fmt=markers[i % len(markers)], ls=":",
                          color=beta_cmap[b], label=rf"$\beta={b}$")

for e in e_ym:
    p, _ = load_plaq_avg_err_from_hmc_json(e["path"])
    if np.isfinite(p) and e["beta"] in beta_cmap:
        sax_left.axhline(p, color=beta_cmap[e["beta"]], ls="-", alpha=0.3, lw=1)

sax_left.set_xlabel(r"$am_0$")
sax_left.set_ylabel(r"$\langle \mathcal{P} \rangle$")
sax_left.set_ylim(0.36, 0.68)
if show_legend:
    sax_left.legend(ncol=3, loc="upper left", fontsize="x-small", columnspacing=0.5)

# Right-top: Plaq vs Beta (colored by mass)
sax_rtop = fig3.add_subplot(gs[0, 1])
m_cmap = mpl.cm.inferno(np.linspace(0.1, 0.85, max(1, len(s_masses))))
for i, m in enumerate(s_masses):
    p_pts = []
    for e in [x for x in e_shamir if np.isclose(x["mass"], m)]:
        v, er = load_plaq_avg_err_from_hmc_json(e["path"])
        if np.isfinite(v):
            p_pts.append((e["beta"], v, er))
    if p_pts:
        xb, yp, ye = zip(*sorted(p_pts))
        sax_rtop.errorbar(xb, yp, yerr=ye, fmt=markers[i % len(markers)], ls=":",
                          color=m_cmap[i], label=rf"$am_0={m}$")
sax_rtop.set_ylabel(r"$\langle \mathcal{P} \rangle$")
sax_rtop.set_ylim(0.36, 0.68)
plt.setp(sax_rtop.get_xticklabels(), visible=False)

# Right-bottom: mres vs Beta (from m_res.json using mres_extract)
sax_rbot = fig3.add_subplot(gs[1, 1], sharex=sax_rtop)
mres_markers = ["*", "o", "s", "D", "p"]

for i, m in enumerate(s_masses):
    r_pts = []
    for e in [x for x in e_mres if np.isclose(x["mass"], m)]:
        v, er = load_mres_extract_from_mres_json(e["path"])
        if np.isfinite(v):
            r_pts.append((e["beta"], v, er))
    if r_pts:
        xr, yr, yre = zip(*sorted(r_pts))
        sax_rbot.errorbar(xr, yr, yerr=yre, marker=mres_markers[i % len(mres_markers)],
                          ls="-", color=m_cmap[i], alpha=0.7, label=rf"$am_0={m}$")

sax_rbot.set_xlabel(r"$\beta$")
sax_rbot.set_ylabel(r"$am_{\rm res}^{\rm fit}$")
sax_rbot.set_ylim(0.0, 0.14)
sax_rbot.set_yticks([0, 0.04, 0.08, 0.12])



# horizontal reference line
sax_rbot.grid(False)
sax_rbot.axhline(0.02, color="gray", linestyle="--", alpha=0.7)

# label for the line
sax_rbot.text(
    0.24, 0.025,
    r"$am_{\rm res}^{\rm fit} = 0.02$",
    fontsize="x-small",
    color="dimgrey",
    ha="right",
    va="bottom",
    transform=sax_rbot.get_yaxis_transform()
)

if show_legend:
    sax_rtop.legend(fontsize="x-small", loc="upper left", ncol=2)
    sax_rbot.legend(fontsize="x-small", loc="upper left", ncol=1)

fig3.savefig(args.shamir_summary, dpi=300, bbox_inches="tight")

plt.close("all")