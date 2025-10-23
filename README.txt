# DWF_workflow
This repository contains a workflow to perform parameter scans of the four parameters
{ α, a₅, m₅, mₚᵥ } for MDWF (Mobius Domain Wall Fermions).
The workflow automates the extraction and the production of both tables and plots of the residual mass and integrated autocorrelation time for each parameter.

Repository Structure
DWF_workflow/
├── raw_data/
│   └── {ensembles_dir}/NF2/Nt{Nt}/Ns{Ns}/Ls{Ls}/B{beta}/M{mass}/mpv{mpv}/alpha{alpha}/a5{a5}/M5{M5}/mesons/
│       ├── pt_ll.n.h5
│       └── mres.n.h5
│
├── metadata/
│   └── parameter_scan/{ensembles_dir}/ensembles_{parameter}.yaml
│
├── assets/
│   ├── plots/{ensembles_dir}
│   └── tables/{ensembles_dir}
│
└── workflow/Snakefile

Requirements
To set up a working environment, ensure the following:

Raw data files
Each ensemble’s data is stored under raw_data/ with the following hierarchy:
raw_data/{ensembles_dir}/NF2/Nt{Nt}/Ns{Ns}/Ls{Ls}/B{beta}/M{mass}/mpv{mpv}/alpha{alpha}/a5{a5}/M5{M5}/mesons/
Each directory must contain:
1. pt_ll.n.h5 — full set of meson correlator contractions
2. mres.n.h5 — contraction between the midpoint function and the pseudoscalar current
Here, n corresponds to the configuration index.

Metadata files
The scan parameters are defined in YAML files located in:
metadata/parameter_scan/{ensembles_dir}/ensembles_{parameter}.yaml
where {parameter} ∈ {a5, M5, mpv, alpha}.
Example for a scan over different values of a5:
- beta: 7.4
  mass: 0.06
  mpv: 1.0
  Ls: 8
  Nt: 16
  Ns: 16
  alpha: 2.0
  a5: [0.9, 1.0, 1.1, 1.2]
  M5: 1.8
  
Usage
The workflow is managed with Snakemake and uses Conda for environment management.
Generate Residual Mass Tables
snakemake --jobs 1 --forceall --printshellcmds --use-conda \
  assets/tables/{ensembles_dir}/mres_scan_{parameter}_table.tex
Generate Plots
snakemake --jobs 1 --forceall --printshellcmds --use-conda \
  assets/plots/{ensembles_dir}/mres_scan_{parameter}.pdf
  
