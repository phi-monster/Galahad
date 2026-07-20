# `experiments/` — one script per paper artifact

Each figure and table in the paper is a standalone script (one artifact per file). Scripts are named after the artifact, not the figure number, so the mapping survives edits to the paper. All models are public checkpoints; the full paper runs on commodity RTX 3090-class GPUs.

| script | paper artifact |
|---|---|
| `main_battery.py` | the main battery — object substitution: Galahad 94.5% vs base 13.5% vs field 0–1% |
| `release_battery.py` | the release table — seven referent-type axes measured on the shipped merged weight |
| `conditioning_gradient.py` | the conditioning gradient — text-cond ≫ action-cond ≫ imitation |
| `six_axes.py` | one adapter, six referent types (SWAP-OBEY 77–95%) + the held-out-type law |
| `exposure_gradient.py` | target-trained 78 / distractor-only 61 / never-seen 0 |
| `mechanism.py` | binding in the language stream; the action head routes position |
| `cure_lattice.py` | protection-capacity curve + coupling harm + regularizer nulls |

Scripts are curated from the research pipeline on release; the figure-rendering code (from frozen numbers) is in `../figures/make_figures.py`.
