# Galahad

> **Instruction Blindness in Vision–Language–Action Policies: Diagnosis and a Low-Rank Data Cure**
> **[Project page](https://xn--7xa.monster/Galahad/)** · **[Paper (PDF)](docs/static/galahad.pdf)** · **[Model](https://huggingface.co/phi-monster/Galahad)** · **[Datasets](https://huggingface.co/phi-monster)** — *preprint, under review.*

A vision–language–action (VLA) policy can pass a manipulation benchmark without reading its instruction: replace the words with the string `xxx` and it emits the same actions. It succeeds by moving to the object that usually occupies that position, not by resolving the name. We call this failure mode **instruction blindness** and show it is a concrete, measurable instance of *causal confusion* in imitation learning. On the one axis that separates a position shortcut from language grounding — object identity, rename the target and keep the scene — the published field selects the correct object **0–1%** of the time.

**The mechanism is a severed route, not missing knowledge.** The pretrained backbone already resolves the named object in its language stream (a frozen probe reads the binding; injecting the word direction flips the selected object 100% of the time), but the action head routes position and never reads that binding. Because the knowledge is present, the repair is a **data recipe, not an architecture**: make the instruction the only predictor of the target, and protect the pretrained grounding with a low-rank update. Full fine-tuning on the same data *destroys* grounding (12.5%); the low-rank cure installs it (**94.5%**, from a base of 13.5%).

**Galahad is the released policy that validates the cure.** Rename the target and the arm goes to the newly named object (**200 / 200**, landing a median **2 mm** from it). One set of weights grounds seven referent-type axes (object identity, spatial, goal, color, category, negation, composition; obey 75–99%, occlusion ≤2% on every axis), and the recipe transfers to a second simulator.

## The battery: measure the shortcut in any policy

The measurement is a one-command counterfactual — not a drop under perturbation, but a positive control: rename the target, and does the arm go to the newly named object?

```python
from galahad.battery import run

report = run(policy, suite="libero_object")   # any VLAWrapper-compatible policy
report.swap_obey     # goes to the newly named object?   (grounding)
report.occ           # collapses with blank cameras?      (uses vision)
report.nonsense      # collapses on a meaningless name?   (needs a real name)
report.reach_probe   # every failure split motor vs grounding
```

`swap-obey` is the load-bearing number: it converts a hollow "it failed" into "it obeyed the new name." A policy that rides a position shortcut scores at chance no matter how high its task success. *(The harness ships in `galahad/`; the one-command `pip` interface above is the packaging in progress — the protocol and scorers are these scripts.)*

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
galahad/        # the battery harness: eval, serving, scorers, dual-output patch
generator/      # deconfounded data generators — one design per referent type
experiments/    # one runbook per paper artifact
docs/           # project page (GitHub Pages)
```

## Release artifacts

- **Model** — [one merged checkpoint](https://huggingface.co/phi-monster/Galahad) grounding seven referent-type axes in a single set of weights, and the [object-family dual-output checkpoint](https://huggingface.co/phi-monster/Galahad-object-unified) (action + grounded future-prediction), rebuilt from the released recipe and battery-verified.
- **Dataset + generator** — the deconfounded referent-type sets in LeRobot format, and the generator that produces both the exam (the battery) and the medicine (the training set) from one principle.
- **Battery** — the one-command grounding measurement above.
- **Paper** — the manuscript and every figure script.

Everything open, permissively licensed, one-command reproducible.

---

Released under the [Apache 2.0 License](LICENSE) · [Φ(fight) Research](https://xn--7xa.monster)
