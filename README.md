# Galahad

**A counterfactual battery for instruction following in vision–language–action policies, and the
deconfounded demonstration generators that repair it.**

A VLA policy can finish its whole benchmark without reading a word of its instruction. The base policy we
study completes the LIBERO object tasks on 198 of 200 trials; replace the instruction with the string `xxx`
and change nothing else, and it still completes 89 of them, walking to the object that habitually occupies
the target position. Standard suites cannot see this, because they place the named object where the
demonstrations put it: a policy that resolves the name and a policy that replays a position collect the same
score.

This repository holds the instrument that separates them, and the data recipe that fixes the policy.

## The instrument

Hold the pixels, the scene and the robot pose fixed, change **one word** of the instruction, and record which
object the end-effector reaches. A position shortcut cannot pass it, and it converts a hollow "the policy
failed" into a positive control, "the policy obeyed the new name."

```bash
python galahad/libero_pro_eval.py --ckpt <checkpoint> --face swap   # rename the target
python galahad/libero_pro_eval.py --ckpt <checkpoint> --face occ    # blank both cameras
python galahad/libero_pro_eval.py --ckpt <checkpoint> --face nonsense
python galahad/do_nothing_check.py                                  # the floors
```

Four gates close four ways of scoring without reading. `swap` asks whether the arm follows the named object.
`occ` asks whether it uses vision. `nonsense` asks whether it needs a real name. A do-nothing policy and a
random-motion policy run the whole battery first, so every rate is read against a measured floor rather than
against zero. Every failure is split by which object the gripper first closed on, so a motor limit is never
recorded as a grounding limit.

On this instrument the base policy reaches the named object on **57 of 160** trials, against a random-motion
floor of 12%.

## The repair

Make the instruction the only thing that predicts the target. Every object appears as the target and as a
distractor, across randomized positions, so a position prior earns nothing and only reading the name pays.
Apply that under a rank-32 update that reaches the vision–language backbone, and the same policy reaches the
named object on **97.7%** of trials across three seeds.

```bash
python generator/collect_deconf_task_c1.py      # position-randomized demonstrations
python generator/collect_confounded_task_c1.py  # the confounded control: same oracle, targets left in place
```

The confounded collector is in the repository on purpose. It is the control that shows the recipe is the data
and not the oracle: same scripted controller, same episode count, every target left in its accustomed
position.

## Layout

```
galahad/     the battery: protocol, serving contracts, scorers, patching probes, floors
generator/   the demonstration generators, one design per referent type, one scripted oracle
```

## Status

A manuscript describing the full comparison is under review. This repository will carry the released
policies, the demonstration sets and the per-episode logs behind every reported number when that review
concludes. Earlier drafts of this work reported figures from checkpoints that no longer exist; those numbers
are superseded by the ones above and should not be cited.

---

Released under the [Apache 2.0 License](LICENSE).
