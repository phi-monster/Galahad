#!/usr/bin/env python3
"""One adapter, six referent types + held-out-type law (paper Fig. 5).

Pipeline (run each command in a prepared environment; see ../galahad/README.md):
# per-type SWAP-OBEY + OCC on RoboCasa with the universal adapter
python ../galahad/rc_eval.py
python ../galahad/rc_eval_colour.py
python ../galahad/rc_eval_negation.py
python ../galahad/rc_eval_compositional.py
python ../galahad/rc_eval_ordcol.py     # ordinal x colour compositional zero-shot
Set RUN=1 to execute the eval step directly; default prints the pipeline.
"""
import os, sys, subprocess
CMDS = """# per-type SWAP-OBEY + OCC on RoboCasa with the universal adapter
python ../galahad/rc_eval.py
python ../galahad/rc_eval_colour.py
python ../galahad/rc_eval_negation.py
python ../galahad/rc_eval_compositional.py
python ../galahad/rc_eval_ordcol.py     # ordinal x colour compositional zero-shot"""
print(__doc__)
if os.environ.get("RUN") == "1":
    for line in [l.strip() for l in CMDS.splitlines() if l.strip() and not l.strip().startswith("#")]:
        print("+", line); subprocess.run(line, shell=True, check=True)
