"""npz_to_lerobot.py (py3.12 venv/main) — convert raw oracle npz episodes -> ONE LeRobot dataset.
use_videos=False → images embedded (matches predecode byte reads). SPEED: the np.load(npz) decompress is the serial
main-loop bottleneck → prefetch it with a THREADPool (np.load releases the GIL on the C decompress → real parallelism,
and threads DON'T fork so no image-writer deadlock — a Process Pool deadlocks here). image_writer_threads async-encodes PNGs.
Run: RAW=/root/lov_raw REPO=lerobot/X ROOT=/root/libdata/X /venv/main/bin/python npz_to_lerobot.py"""
import os, glob, shutil, inspect, numpy as np
from concurrent.futures import ThreadPoolExecutor
from lerobot.datasets.lerobot_dataset import LeRobotDataset

RAW = os.environ.get("RAW", "/root/lov_raw")
REPO = os.environ.get("REPO", "lerobot/libero_object_varied")
ROOT = os.environ.get("ROOT", "/root/libdata/libero_object_varied")
# VIDEO=1 -> video-backed. The image-feature dtype MUST be "video" (not "image"): in lerobot 0.3.x use_videos=True
# does NOT auto-convert image-dtype features, so meta.video_keys stays empty and save_episode never encodes ->
# VIDEO=1 silently produced 220k per-frame PNGs before this fix. Video = ~500 mp4 => minutes (not hours) to push to HF.
_VID = bool(os.environ.get("VIDEO"))
_IMDTYPE = "video" if _VID else "image"
FEATURES = {
    "observation.images.image": {"dtype": _IMDTYPE, "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.wrist_image": {"dtype": _IMDTYPE, "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
}


def _load(f):   # runs in a worker THREAD; np.load decompress releases the GIL → parallel prefetch
    d = np.load(f, allow_pickle=True)
    return (d["img"], d["wrist"], d["state"], d["action"], str(d["lang"]))


shutil.rmtree(ROOT, ignore_errors=True)
_ckw = dict(fps=20, features=FEATURES, root=ROOT, use_videos=_VID,
            image_writer_threads=int(os.environ.get("IWT", "24")))
if int(os.environ.get("IWP", "0")) > 0 and "image_writer_processes" in inspect.signature(LeRobotDataset.create).parameters:
    _ckw["image_writer_processes"] = int(os.environ["IWP"])
if _VID and os.environ.get("VCODEC") and "vcodec" in inspect.signature(LeRobotDataset.create).parameters:
    _ckw["vcodec"] = os.environ["VCODEC"]   # h264 (libx264) ~3-4x faster than default libsvtav1; IDENTITY-normed images => codec-invariant for training
# batch_encoding_size=1 => each episode's mp4 is encoded inside save_episode immediately; without this a batched
# remainder would be left un-encoded (we skip finalize()/consolidate()). Guard for lerobot versions lacking the param.
if _VID and "batch_encoding_size" in inspect.signature(LeRobotDataset.create).parameters:
    _ckw["batch_encoding_size"] = 1
ds = LeRobotDataset.create(REPO, **_ckw)
files = sorted(glob.glob(f"{RAW}/ep_*.npz"))
print(f"converting {len(files)} episodes (threadpool prefetch)", flush=True)
import inspect
# lerobot API drift: <=0.2.x took task INSIDE the frame dict; >=0.3.x (e.g. box4 rc_venv 0.3.3) takes task as a
# separate positional arg. Detect once, call accordingly — robust across boxes.
_TASK_ARG = "task" in inspect.signature(ds.add_frame).parameters
with ThreadPoolExecutor(max_workers=int(os.environ.get("NLOAD", "24"))) as ex:
    for k, (img, wrist, state, action, lang) in enumerate(ex.map(_load, files)):
        for i in range(len(img)):
            frame = {"observation.images.image": img[i], "observation.images.wrist_image": wrist[i],
                     "observation.state": state[i].astype(np.float32), "action": action[i].astype(np.float32)}
            if _TASK_ARG:
                ds.add_frame(frame, task=lang)
            else:
                frame["task"] = lang; ds.add_frame(frame)
        ds.save_episode()
        if k % 50 == 0:
            print(f"  {k}/{len(files)}", flush=True)
# lerobot API drift: <=0.2.x needed finalize()/consolidate(); >=0.3.x persists incrementally in save_episode() (none needed)
for _m in ("finalize", "consolidate"):
    if hasattr(ds, _m):
        getattr(ds, _m)(); break
_ne = getattr(ds, "num_episodes", None) or ds.meta.total_episodes
_nf = getattr(ds, "num_frames", None) or ds.meta.total_frames
print(f"CONVERT_DONE episodes={_ne} frames={_nf} at {ROOT}", flush=True)
