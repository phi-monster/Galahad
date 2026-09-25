# `generator/` — deconfounded data generators

One principle, one design per referent type. The principle: **the instruction is the only predictor of the target** — every object appears as the target and as a distractor across randomized positions, distractors always in frame, so a position or scene shortcut earns nothing and only reading the name pays.

| script | referent type (confound broken) |
|---|---|
| `collect_spatial_c1.py` | spatial (side ⟂ identity) |
| `collect_rc_goal.py` + `goal_oracle.py` | goal (remembered destination) |
| `collect_rc_colour.py` | color (color ⟂ position ⟂ shape) |
| `collect_category.py` | category (category ⟂ instance ⟂ co-appearance) |
| `collect_ordinal.py` | ordinal (ordinal ⟂ absolute position ⟂ identity) |
| `collect_rc_negation.py` | negation (keyword-match) |
| `collect_rc_compositional.py` | composition (single-attribute lookup) |
| `collect_rc_relation.py` | relation (side ⟂ identity) |
| `collect_liftslide.py` | verb (object ⟂ verb) |
| `collect_rc_deconf.py` | object identity, RoboCasa (position, scene) |
| `collect_confounded_task_c1.py` | the confounded control arm |
| `npz_to_lerobot*.py` · `merge_datasets.py` · `fast_qstats_g1.py` | conversion, merging, quantile stats |

The generator produces the exam (the battery) and the medicine (the training set) from the same code. The generated sets themselves are released on the [Hugging Face org](https://huggingface.co/phi-monster). The LIBERO object-identity collector is being restored from the training box and lands in the next commit; its output dataset (`galahad-deconf-object`) is already public.
