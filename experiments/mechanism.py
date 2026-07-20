#!/usr/bin/env python3
"""Mechanism probes — binding present, action head does not route it (paper Fig. 3).

Pipeline (run each command in a prepared environment; see ../galahad/README.md):
# open-loop action-direction probe, base vs cured (role-gap closes after the cure)
# see ../galahad/goal_reach_probe.py and the activation-patching description in paper Sec. 4
Set RUN=1 to execute the eval step directly; default prints the pipeline.
"""
import os, sys, subprocess
CMDS = """# open-loop action-direction probe, base vs cured (role-gap closes after the cure)
# see ../galahad/goal_reach_probe.py and the activation-patching description in paper Sec. 4"""
print(__doc__)
if os.environ.get("RUN") == "1":
    for line in [l.strip() for l in CMDS.splitlines() if l.strip() and not l.strip().startswith("#")]:
        print("+", line); subprocess.run(line, shell=True, check=True)
