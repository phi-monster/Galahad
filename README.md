# Galahad

> **Galahad and the Siege Perilous: A Cheat-Proof Trial for Language Grounding, and the First Policy to Survive It**
> **[Project page](https://galahad.xn--7xa.monster/)** · Paper · model · dataset — *remaining links land with the arXiv posting.*

A vision-language-action (VLA) policy can pass a manipulation benchmark without reading its instruction: replace the words with the string `xxx` and it emits the same actions. It succeeds by moving to the object that usually sits in that position, not by reading the name. On the one axis that separates the two — object identity, rename the target and keep the scene — the published field selects the correct object **0–1%** of the time.

**Galahad is the first policy to survive a change-one-word trial.** Rename the target and the arm goes to the newly named object (**200 / 200**, landing a median **2 mm** from it), where its base scores **13.5%** and the field 0–1%. The cure is a data recipe, not an architecture: make the instruction the only predictor of the target, and protect the pretrained grounding with a low-rank update — full fine-tuning on the same data *destroys* grounding (12.5%), the low-rank cure installs it (**94.5%**). The released weight is one merged checkpoint grounding seven referent-type axes (object identity, spatial, goal, color, category, negation, composition; obey 75–99%, occlusion ≤2% on every axis). Every result was produced for **less than $1,000** of commodity GPU rental.

## The battery: check any policy for the shortcut

The core release is a one-command grounding lie-detector. It runs the same trial on *your* policy — not a drop under perturbation, but a positive control: rename the target, and does the arm go to the newly named object?

```python
from galahad.battery import run

report = run(policy, suite="libero_object")   # any VLAWrapper-compatible policy
report.swap_obey     # goes to the newly named object?   (grounding)
report.occ           # collapses with blank cameras?      (uses vision)
report.nonsense      # collapses on a meaningless name?   (needs a real name)
report.reach_probe   # every failure split motor vs grounding
```

`swap-obey` is the load-bearing number: it converts a hollow "it failed" into "it obeyed the new name." A policy that rides a position shortcut scores at chance no matter how high its task success. *(Battery code lands with the arXiv release; the interface above is stable.)*

## Reproducing the paper

Each figure and table is a standalone script in `experiments/` (one artifact per file). All models are public checkpoints; the full paper runs on commodity RTX 3090-class GPUs, and the merged model inferences on a single 24 GB card.

```bash
python experiments/main_battery.py            # 94.5% vs base 13.5% vs field 0–1%
python experiments/release_battery.py         # seven axes on the shipped merged weight
python experiments/conditioning_gradient.py   # text-cond ≫ action-cond ≫ imitation
```

*(Scripts are curated from the research pipeline on release; the artifact mapping is in `experiments/`.)*

## Layout

```
galahad/        battery.py        # the grounding lie-detector (pip-installable)
generator/                        # deconfounded data generators — one design per referent type
experiments/                      # one script per paper figure/table
paper/          main.tex          # manuscript (sources at repo root during active writing)
                galahad_draft.pdf
docs/           index.html        # project page (GitHub Pages)
```

## Release artifacts

- **Model** — one merged checkpoint grounding seven referent-type axes in a single set of weights. The training code reproduces the dual-output (action + grounded future-prediction) experiment end-to-end.
- **Dataset + generator** — the deconfounded referent-type sets in LeRobot format, and the generator that produces both the exam (the battery) and the medicine (the training set) from one principle.
- **Battery** — the one-command grounding check above.
- **Paper** — the manuscript and every figure script.

Everything open, permissively licensed, one-command reproducible.

---

Released under the [Apache 2.0 License](LICENSE) · [Φ(fight) Research](https://xn--7xa.monster)
