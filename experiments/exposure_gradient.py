#!/usr/bin/env python3
"""Exposure gradient — target-trained / distractor-only / never-in-data (paper Fig. 6).

Pipeline (run each command in a prepared environment; see ../galahad/README.md):
# train the recipe with an object held out of targets (distractor-only) and banished (never-in-data),
# then run main_battery.py per bin; see ../generator/ for the role-balanced collectors
Set RUN=1 to execute the eval step directly; default prints the pipeline.
"""
import os, sys, subprocess
CMDS = """# train the recipe with an object held out of targets (distractor-only) and banished (never-in-data),
# then run main_battery.py per bin; see ../generator/ for the role-balanced collectors"""
print(__doc__)
if os.environ.get("RUN") == "1":
    for line in [l.strip() for l in CMDS.splitlines() if l.strip() and not l.strip().startswith("#")]:
        print("+", line); subprocess.run(line, shell=True, check=True)
