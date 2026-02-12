#!/usr/bin/env python3
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

# ============================================================
# COMMAND LINE INTERFACE (CLI)
# ============================================================
parser = argparse.ArgumentParser(description="Bulk phase plotting tool.")
parser.add_argument("--ensembles_csv", required=True)
parser.add_argument("--plaq_avg", nargs="+", required=True)
parser.add_argument("--plaq_history", nargs="*", default=[])
parser.add_argument("--mres_data", nargs="*", default=[])
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
def is_true(x): return str(x).strip().upper() in {"TRUE", "T", "1", "YES", "Y"}
def sfloat(x): return str(float(str(x).strip()))
def sint(x): return str(int(float(str(x).strip())))

def make_dyn_key(NF, Nt, Ns, Ls, beta, mass, mpv, alpha, a5, M5):
    return (sint(NF), sint(Nt), sint(Ns), sint(Ls), sfloat(beta), sfloat(mass), sfloat(mpv), sfloat(alpha), sfloat(a5), sfloat(M5))

def load_val_err(filepath, key_name):
    try:
        with open(filepath) as f:
            header = [h.lstrip('#') for h in f.readline().split()]
            values = f.readline().split()
        d = dict(zip(header, values))
        val = float(d[key_name])
        err = float(d.get(f"{key_name}_err", 0.0))
        return val, err
    except (Exception, KeyError, FileNotFoundError):
        return np.nan, np.nan

def get_shaded_color(color, amount=0.4):
    c = mcolors.to_rgb(color)
    return tuple([c[i] + (1 - c[i]) * amount for i in range(3)])

FLOAT_TOKEN = r"[0-9]+(?:\.[0-9]+)?"
PAT_DYN = re.compile(r".*/NF(?P<NF>\d+)/Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/B(?P<beta>{FT})/M(?P<mass>{FT})/mpv(?P<mpv>{FT})/alpha(?P<alpha>{FT})/a5(?P<a5>{FT})/M5(?P<M5>{FT})/.*".format(FT=FLOAT_TOKEN))
PAT_YM = re.compile(r".*/yang_mills/Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/B(?P<beta>{FT})/.*".format(FT=FLOAT_TOKEN))

def parse_info(fp):
    m = PAT_DYN.match(str(fp))
    if m: return "dyn", make_dyn_key(**m.groupdict()), m.groupdict()
    m = PAT_YM.match(str(fp))
    if m: return "ym", (sint(m.group("Nt")), sint(m.group("Ns")), sfloat(m.group("beta"))), m.groupdict()
    return None, None, None

# ============================================================
# DATA AGGREGATION
# ============================================================
df_meta = pd.read_csv(args.ensembles_csv, sep=r"\t|,", engine="python")
meta_map = {make_dyn_key(r['NF'],r['Nt'],r['Ns'],r['Ls'],r['beta'],r['mass'],r['mpv'],r['alpha'],r['a5'],r['M5']): r 
            for _, r in df_meta.iterrows() if not is_true(r.get("YM", False))}

e_tuned, e_shamir, e_ym, e_mres = [], [], [], []

for fp in args.plaq_avg:
    kind, key, g = parse_info(fp)
    if kind == "ym": e_ym.append({"beta": float(g["beta"]), "path": fp})
    elif key in meta_map:
        m = meta_map[key]
        if is_true(m.get("use_in_bulkphase_tuned")): e_tuned.append({"key": key, "beta": float(g["beta"]), "mass": float(g["mass"]), "path": fp})
        if is_true(m.get("use_in_bulkphase_Shamir")): e_shamir.append({"key": key, "beta": float(g["beta"]), "mass": float(g["mass"]), "path": fp})

for fp in args.mres_data:
    kind, key, g = parse_info(fp)
    if key in meta_map and is_true(meta_map[key].get("use_in_bulkphase_mres")):
        e_mres.append({"beta": float(g["beta"]), "mass": float(g["mass"]), "path": fp})

hist_map = {parse_info(fp)[1]: fp for fp in args.plaq_history if parse_info(fp)[1]}

all_betas = sorted(set([e["beta"] for e in e_tuned + e_shamir + e_ym]))
beta_cmap = dict(zip(all_betas, mpl.cm.viridis_r(np.linspace(0, 1, len(all_betas)))))
markers = ["o", "s", "D", "^", "v", "<", ">"]

# ------------------------------------------------------------
# FIG 1: Tuned Möbius <P> vs mass
# ------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
for i, b in enumerate(sorted(set(e["beta"] for e in e_tuned))):
    pts = []
    for e in [x for x in e_tuned if x["beta"] == b]:
        v, er = load_val_err(e["path"], "plaq")
        if not np.isnan(v): pts.append((e["mass"], v, er))
    if pts:
        xs, ys, ye = zip(*sorted(pts))
        # Restore dotted line style (":")
        ax1.errorbar(xs, ys, yerr=ye, fmt=markers[i%len(markers)], ls=":", color=beta_cmap[b], label=rf"$\beta={b}$")
ax1.set_xlabel(r"$am_0$"); ax1.set_ylabel(r"$\langle \mathcal{{P}} \rangle$")
ax1.legend(fontsize='x-small', bbox_to_anchor=(1, 0.9))
fig1.savefig(args.tuned_masses, dpi=300)

# ------------------------------------------------------------
# FIG 2: Histories (Solid=Smallest, Dotted=Largest)
# ------------------------------------------------------------
h_mass_list = sorted(args.history_masses)
h_betas = sorted(set(e["beta"] for e in e_tuned if e["key"] in hist_map))
fig2, axes2 = plt.subplots(len(h_betas), 1, figsize=(3.5, 2.5), sharex=True, layout="constrained")
if len(h_betas) == 1: axes2 = [axes2]

for ax, b in zip(axes2[::-1], h_betas):
    group = sorted([e for e in e_tuned if e["beta"] == b and any(np.isclose(e["mass"], m) for m in h_mass_list)], key=lambda x: x["mass"])
    for j, e in enumerate(group):
        t, p = np.loadtxt(hist_map[e["key"]], unpack=True)
        ls = "-" if j == 0 else (":" if j == len(group)-1 else "-")
        ax.plot(t, p, color=get_shaded_color(beta_cmap[b], 0.4*j), ls=ls, alpha=0.7, label=rf"$am_0={e['mass']}$")
    ax.set_ylabel(rf"$ \mathcal{{P}} [\beta={b}]$")
    if show_legend: ax.legend(loc="upper right", fontsize='xx-small', ncol=2)
axes2[-1].set_xlabel("Monte Carlo time")
axes2[-1].set_xlim(150,6900)
fig2.savefig(args.tuned_history, dpi=300)

# ------------------------------------------------------------
# FIG 3: Shamir Summary
# ------------------------------------------------------------
s_betas = sorted(set(e["beta"] for e in e_shamir))
s_masses = sorted(set(e["mass"] for e in e_shamir))
fig3, (sax1, sax2) = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True, layout="constrained")

# Left Panel: Plaq vs Mass
for i, b in enumerate(s_betas):
    pts = []
    for e in [x for x in e_shamir if x["beta"] == b]:
        v, er = load_val_err(e["path"], "plaq")
        if not np.isnan(v): pts.append((e["mass"], v, er))
    if pts:
        xs, ys, ye = zip(*sorted(pts))
        # Restore dotted line style (":")
        sax1.errorbar(xs, ys, yerr=ye, fmt=markers[i%len(markers)], ls=":", color=beta_cmap[b], label=rf"$\beta={b}$")

for e in e_ym:
    p, _ = load_val_err(e["path"], "plaq")
    if not np.isnan(p) and e["beta"] in beta_cmap:
        # Yang-Mills line is now solid ("-")
        sax1.axhline(p, color=beta_cmap[e["beta"]], ls="-", alpha=0.5)

# Right Panel: Plaq vs Beta & Mres vs Beta
sax2_twin = sax2.twinx()
m_cmap = mpl.cm.viridis_r(np.linspace(0, 1, len(s_masses)))

for i, m in enumerate(s_masses):
    p_pts = []
    for e in [x for x in e_shamir if x["mass"] == m]:
        v, er = load_val_err(e["path"], "plaq")
        if not np.isnan(v): p_pts.append((e["beta"], v, er))
    if p_pts:
        xb, yp, ye = zip(*sorted(p_pts))
        sax2.errorbar(xb, yp, yerr=ye, fmt=markers[i%len(markers)], ls=":", color=m_cmap[i], label=rf"Plaq. $am_0={m}$")
    
    r_pts = []
    for e in [x for x in e_mres if x["mass"] == m]:
        v, er = load_val_err(e["path"], "mres_fit")
        if not np.isnan(v): r_pts.append((e["beta"], v, er))
    if r_pts:
        xr, yr, yre = zip(*sorted(r_pts))
        sax2_twin.errorbar(xr, yr, yerr=yre, fmt='x', ls='-', color=m_cmap[i], alpha=0.6, label=rf"$am_{{\rm res}} (am_0={m})$")

sax1.set_xlabel(r"$am_0$"); sax1.set_ylabel(r"$\langle P \rangle$")
sax1.set_ylim(0.36,0.68)

sax2.set_xlabel(r"$\beta$")
sax2_twin.set_ylabel(r"$am_{\rm res}$", color='black') 
sax2_twin.set_ylim(0.,0.145) 
sax2.yaxis.set_tick_params(labelleft=False)

if show_legend:
    sax1.legend(ncol=4, loc='upper left', fontsize='xx-small')
    sax2.legend(loc='upper left', fontsize='xx-small')
    sax2_twin.legend(loc='lower center', fontsize='xx-small')

fig3.savefig(args.shamir_summary, dpi=300)
plt.close('all')