#!/usr/bin/env python3
"""Main battery — object substitution (paper Table 1).

Pipeline (run each command in a prepared environment; see ../galahad/README.md):
# 1. serve the checkpoint (object-family recipe ckpt, or the release weight)
# ZERO_STATE=1 python ../galahad/arm_eval_server.py --pretrained phi-monster/Galahad --port 5000
# 2. faces: TASK / OCC / SWAP2(+PROBE) / NONSENSE
python ../galahad/libero_pro_eval.py --suite libero_object_task --port 5000
python ../galahad/libero_pro_eval.py --suite libero_object_task --port 5000 --occ
python ../galahad/libero_pro_eval.py --suite libero_object_task --port 5000 --swap_scene
python ../galahad/libero_pro_eval.py --suite libero_object_task --port 5000 --nonsense
Set RUN=1 to execute the eval step directly; default prints the pipeline.
"""
import os, sys, subprocess
CMDS = """# 1. serve the checkpoint (object-family recipe ckpt, or the release weight)
# ZERO_STATE=1 python ../galahad/arm_eval_server.py --pretrained phi-monster/Galahad --port 5000
# 2. faces: TASK / OCC / SWAP2(+PROBE) / NONSENSE
python ../galahad/libero_pro_eval.py --suite libero_object_task --port 5000
python ../galahad/libero_pro_eval.py --suite libero_object_task --port 5000 --occ
python ../galahad/libero_pro_eval.py --suite libero_object_task --port 5000 --swap_scene
python ../galahad/libero_pro_eval.py --suite libero_object_task --port 5000 --nonsense"""
print(__doc__)
if os.environ.get("RUN") == "1":
    for line in [l.strip() for l in CMDS.splitlines() if l.strip() and not l.strip().startswith("#")]:
        print("+", line); subprocess.run(line, shell=True, check=True)
