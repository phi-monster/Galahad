# `galahad/` — the battery

The grounding lie-detector: a one-command check that runs the change-one-word trial on any policy.

- `battery.py` — `run(policy, suite)` → `swap_obey` / `occ` / `nonsense` / `reach_probe`.
- `swap-obey` is the load-bearing metric: it converts a hollow "it failed" into a positive control ("it obeyed the new name"). A position-shortcut policy scores at chance regardless of task success.

Curated from the research pipeline on release. The interface in the top-level README is stable.
