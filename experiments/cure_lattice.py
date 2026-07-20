#!/usr/bin/env python3
"""Cure lattice — protection-capacity, coupling harm, regularizer nulls (paper Fig. 4).

Pipeline (run each command in a prepared environment; see ../galahad/README.md):
# rank sweep: retrain the recipe at rank 8/16/32/64 (same data), run main_battery per rank
# regularizer arms: spectral / AFR / vision-IB on confounded data, then main_battery per arm
Set RUN=1 to execute the eval step directly; default prints the pipeline.
"""
import os, sys, subprocess
CMDS = """# rank sweep: retrain the recipe at rank 8/16/32/64 (same data), run main_battery per rank
# regularizer arms: spectral / AFR / vision-IB on confounded data, then main_battery per arm"""
print(__doc__)
if os.environ.get("RUN") == "1":
    for line in [l.strip() for l in CMDS.splitlines() if l.strip() and not l.strip().startswith("#")]:
        print("+", line); subprocess.run(line, shell=True, check=True)
