"""merge_datasets.py — concatenate N LeRobot datasets into one (for data-scaling: 600-shards → 1200/2400).

NO render (reads stored frames, re-adds) → far faster than re-collecting. Same SmolVLA keys as collect_ladder.py.
lerobot 0.5.1 has no MultiLeRobotDataset + DatasetConfig.repo_id is a str, so a physical merge is required to train on N shards.
API pinned to lerobot 0.5.1 (verified on box): images come back as CHW float[0,1] tensors; meta.tasks is a DataFrame indexed
by the task string with a 'task_index' column; episode boundaries via the per-row 'episode_index' (no episode_data_index).

  PYTHONPATH=<repo>/phi-arena python merge_datasets.py --out phi-monster/arena-ground-B1200-jul2026 \
      --sources phi-monster/arena-ground-B-jul2026 phi-monster/arena-ground-Bshard0-jul2026 [--max_eps 1200]
"""
import os
os.environ.setdefault("HF_LEROBOT_HOME", "/root/lerobot_data")
os.environ["MUJOCO_GL"] = "egl"                     # merge does NOT render; force-satisfy the EGL import check (shell may hold a bad value)
os.environ["PYOPENGL_PLATFORM"] = "egl"
import argparse
import shutil
import numpy as np
import phi_arena.envs.mujoco_grab_decoy_vision as E

IMG = ("observation.image", "observation.image_2")


def _img(v):
    """CHW float[0,1] tensor → HWC uint8 (collect_ladder stored HWC uint8; LeRobotDataset returns CHW float)."""
    a = np.asarray(v, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] in (1, 3):        # CHW → HWC
        a = np.transpose(a, (1, 2, 0))
    if a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    return a.clip(0, 255).astype(np.uint8)


def _task_map(meta):
    t = meta.tasks                                  # DataFrame: index=task string, column 'task_index'
    return {int(t.loc[name, "task_index"]): name for name in t.index}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--max_eps", type=int, default=None)
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HOME = os.environ["HF_LEROBOT_HOME"]
    local = os.path.join(HOME, args.out)
    if os.path.exists(local):
        shutil.rmtree(local)
    out = LeRobotDataset.create(repo_id=args.out, fps=E.FPS, features=E.FEATURES, robot_type="mj", use_videos=False)
    total = 0
    for src_repo in args.sources:
        src = LeRobotDataset(src_repo, root=os.path.join(HOME, src_repo))
        hf = src.hf_dataset
        tmap = _task_map(src.meta)
        cur_ep, buf = None, []
        for i in range(len(hf)):
            row = hf[i]
            ei = int(row["episode_index"])
            if cur_ep is None:
                cur_ep = ei
            if ei != cur_ep:                        # episode boundary → flush
                for fr in buf:
                    out.add_frame(fr)
                out.save_episode(); total += 1; buf = []; cur_ep = ei
                if total % 100 == 0:
                    print(f"[merge] {total} episodes", flush=True)
                if args.max_eps and total >= args.max_eps:
                    break
            buf.append({**{k: _img(row[k]) for k in IMG},
                        "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                        "action": np.asarray(row["action"], dtype=np.float32),
                        "task": tmap[int(row["task_index"])]})
        if buf and not (args.max_eps and total >= args.max_eps):   # last episode
            for fr in buf:
                out.add_frame(fr)
            out.save_episode(); total += 1
        print(f"[merge] source {src_repo}: cumulative {total} episodes", flush=True)
        if args.max_eps and total >= args.max_eps:
            break
    print(f"[merge] DONE out={out.root} total_episodes={total}", flush=True)
    print("MERGE_DATASETS_DONE", flush=True)


if __name__ == "__main__":
    main()
