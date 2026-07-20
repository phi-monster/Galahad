# `galahad/` — the battery

The grounding lie-detector, in harness form. `swap-obey` is the load-bearing metric: it converts a hollow "it failed" into a positive control ("it obeyed the new name"); a position-shortcut policy scores at chance regardless of task success.

| script | role |
|---|---|
| `libero_pro_eval.py` | LIBERO harness: `task` / `--occ` / `--swap_scene` (+reach-probe) / `--nonsense` |
| `libero_goal_vary_eval.py` | goal axis: displacement-survival eval + the scene-variation generator |
| `arm_eval_server.py` / `base_eval_server.py` | policy serving (cured / uncured base) |
| `rc_eval*.py` | RoboCasa referent axes: category, colour, negation, compositional, ordinal×colour |
| `rc_eval_liftslide_cf.py` + `analyze_confusion.py` | verb axis 2×2 confusion |
| `do_nothing_check.py` | init-false gate: a do-nothing policy must score ~0 before any number is trusted |
| `galahad_foresight.py` (+ `c6_processor_shim.py`, `patch_eval_server.py`, `convert_relv2.py`) | dual-output (action + grounded future-prediction) training/inference patch |
| `merge_galahad.py` | fold adapter + prediction head into one checkpoint |
| `c6_chain_demo.py` / `c7_tax.py` | chain demo (both faces flip with one word) / inference-tax measurement |

Serving contract (all three parts, always): the released weights · the normalizer from the checkpoint's lerobot processor files (**never** a bundled `norm_stats.json`) · `ZERO_STATE=1`. A `pip`-packaged one-command CLI is the packaging in progress; the protocol and scorers are these scripts.
