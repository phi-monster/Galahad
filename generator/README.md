# `generator/` — deconfounded data generators

One principle, one design per referent type. The principle: **the instruction is the only predictor of the target** — every object appears as the target and as a distractor across randomized positions, distractors always in frame, so a position or scene shortcut earns nothing and only reading the name pays.

Per-type designs (each breaks a different confound):

| type | confound broken |
|---|---|
| object identity | position, scene |
| color | color ⟂ position ⟂ shape |
| category | category ⟂ instance ⟂ co-appearance |
| ordinal | ordinal ⟂ absolute position ⟂ identity |
| spatial relation | side ⟂ identity |
| negation | keyword-match |
| composition | single-attribute lookup |
| goal | remembered destination |

One scripted-oracle generator + one battery. Each type is a small design, not a new method. The generator produces the exam (the battery) and the medicine (the training set) from the same code. Curated on release.
