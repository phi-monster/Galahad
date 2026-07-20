#!/usr/bin/env python3
"""Conditioning gradient — text-cond WM vs action-cond WAM vs imitation (paper Fig. 2).

Pipeline (run each command in a prepared environment; see ../galahad/README.md):
# WAM rows: run the battery on each open WAM via its native wrapper (TASK+OCC+OCC_STATE+OBEYED_NAME)
# text-cond WM row: the C4-i counterfactual, hand-scored under the continuity protocol (see paper Sec. 3.4)
# Galahad row: main_battery.py above
Set RUN=1 to execute the eval step directly; default prints the pipeline.
"""
import os, sys, subprocess
CMDS = """# WAM rows: run the battery on each open WAM via its native wrapper (TASK+OCC+OCC_STATE+OBEYED_NAME)
# text-cond WM row: the C4-i counterfactual, hand-scored under the continuity protocol (see paper Sec. 3.4)
# Galahad row: main_battery.py above"""
print(__doc__)
if os.environ.get("RUN") == "1":
    for line in [l.strip() for l in CMDS.splitlines() if l.strip() and not l.strip().startswith("#")]:
        print("+", line); subprocess.run(line, shell=True, check=True)
