#!/usr/bin/env python3
"""Release table — seven axes on the shipped merged weight (phi-monster/Galahad).

Pipeline (run each command in a prepared environment; see ../galahad/README.md):
# object+spatial+goal on LIBERO; colour/category/negation/compositional on RoboCasa
# serve: ZERO_STATE=1 python ../galahad/arm_eval_server.py --pretrained phi-monster/Galahad --port 5000
python ../galahad/libero_pro_eval.py --suite libero_object_task --port 5000
python ../galahad/libero_pro_eval.py --suite libero_spatial --port 5000 --spatial_swap
python ../galahad/libero_goal_vary_eval.py --only cream_cheese --vary --max_steps 400
python ../galahad/rc_eval_colour.py
python ../galahad/rc_eval.py            # category
python ../galahad/rc_eval_negation.py
python ../galahad/rc_eval_compositional.py
Set RUN=1 to execute the eval step directly; default prints the pipeline.
"""
import os, sys, subprocess
CMDS = """# object+spatial+goal on LIBERO; colour/category/negation/compositional on RoboCasa
# serve: ZERO_STATE=1 python ../galahad/arm_eval_server.py --pretrained phi-monster/Galahad --port 5000
python ../galahad/libero_pro_eval.py --suite libero_object_task --port 5000
python ../galahad/libero_pro_eval.py --suite libero_spatial --port 5000 --spatial_swap
python ../galahad/libero_goal_vary_eval.py --only cream_cheese --vary --max_steps 400
python ../galahad/rc_eval_colour.py
python ../galahad/rc_eval.py            # category
python ../galahad/rc_eval_negation.py
python ../galahad/rc_eval_compositional.py"""
print(__doc__)
if os.environ.get("RUN") == "1":
    for line in [l.strip() for l in CMDS.splitlines() if l.strip() and not l.strip().startswith("#")]:
        print("+", line); subprocess.run(line, shell=True, check=True)
