"""npz_to_lerobot_c1.py (py3.12 venv/main) — C1 npz -> ONE LeRobot v3 dataset serving BOTH chains.

NOTE: the foresight columns MUST carry the `observation.` prefix — lerobot_train runs `batch = preprocessor(batch)`
(PolicyProcessorPipeline) before policy.forward, and it keeps only the structured slots; a bare `foresight.*`
column IS in the dataset (ds[0] has it) but is DROPPED before the policy sees it. `observation.role_frame`
survived that path in the coupling run — same mechanism.

Standard fields (A4 / release_v2 VLA training reads these; role_frame is DELIBERATELY ABSENT = zero pointer leak):
  observation.images.image / .wrist_image (256) · observation.state [8] · action [7] · task=lang
FORESIGHT targets (C3 reads these; A4's policy simply ignores unknown columns — verified: couple_v1 trained on a
dataset carrying role_frame and never read it). Pre-computed HERE so the dataset stays small (16x16, not full frames):
  foresight.region [16,16] = the NAMED (moving) object's instance-seg mask @ t+K, block-max downsampled  <-- LAB B25:
      NOT a frame-diff. A frame-diff GT is 52.5% ARM / 44.4% target (render-confirmed) => it would train
      "predict where the arm sweeps" = instruction-INDEPENDENT, gutting the 2nd-grounding-trainer core.
  foresight.depth  [16,16] = agentview depth @ t+K, block-mean
  foresight.valid  [1]     = 0 when t+K runs past the episode (mask the loss there)

Moving-object id is identified by TWO independent signals that must AGREE (else the episode is dropped + logged):
  (i) allpos: the object index with the largest 3D displacement over the episode (the oracle moves ONLY the target)
  (ii) seg  : the instance id whose mask centroid travels furthest
and (i)'s name is cross-checked against the recorded target_name.

Run: RAW=/root/dct_c1_raw REPO=lerobot/libero_c1 ROOT=/dev/shm/libero_c1 K=12 /venv/main/bin/python npz_to_lerobot_c1.py
"""
import os, glob, shutil, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

RAW = os.environ.get("RAW", "/root/dct_c1_raw")
REPO = os.environ.get("REPO", "lerobot/libero_c1")
ROOT = os.environ.get("ROOT", "/dev/shm/libero_c1")
FPS = int(os.environ.get("FPS", "20"))
K = int(os.environ.get("K", "12"))          # foresight horizon (frames)
G = int(os.environ.get("G", "16"))          # foresight head output grid
VIZ = int(os.environ.get("VIZ", "6"))       # save this many overlay PNGs for the §3.6 eyeball
VIZDIR = "/root/c1conv_viz"
USE_VID = os.environ.get("USE_VIDEOS", "1") == "1"   # video encoding is the conversion bottleneck (2.2 cores of 128,
#   ~12 s/episode => 3 h for 900 ep). Storing frames as images instead removes ffmpeg entirely; LeRobotDataset
#   supports both and the policy reads observation.images.* either way.
IWT = int(os.environ.get("IMG_THREADS", "8"))

FEATURES = {
    "observation.images.image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    "observation.foresight_region": {"dtype": "float32", "shape": (G, G), "names": ["h", "w"]},
    "observation.foresight_depth": {"dtype": "float32", "shape": (G, G), "names": ["h", "w"]},
    "observation.foresight_valid": {"dtype": "float32", "shape": (1,), "names": ["valid"]},
}


def block_max(m, g):
    h = m.shape[0] // g
    return m.reshape(g, h, g, h).max(axis=(1, 3))


def block_mean(m, g):
    h = m.shape[0] // g
    return m.reshape(g, h, g, h).mean(axis=(1, 3))


def moving_id(seg, allpos, objnames, target_name):
    """Return (seg_id, ok, why). Two independent signals must agree."""
    # (i) allpos: which object moved most in 3D
    disp = np.linalg.norm(allpos[-1] - allpos[0], axis=-1)          # (n_obj,)
    oi = int(np.argmax(disp))
    name_ok = (str(objnames[oi]) == str(target_name))
    # (ii) seg: which instance id's centroid traveled furthest
    t0, t1 = 0, seg.shape[0] - 1
    best, bd = None, -1.0
    for i in np.unique(seg[t1]):
        if i == 0:
            continue
        m0, m1 = seg[t0] == i, seg[t1] == i
        if m0.sum() < 20 or m1.sum() < 20:
            continue
        c0 = np.array(np.nonzero(m0), dtype=np.float32).mean(1)
        c1 = np.array(np.nonzero(m1), dtype=np.float32).mean(1)
        d = float(np.linalg.norm(c0 - c1))
        if d > bd:
            bd, best = d, int(i)
    if best is None:
        return None, False, "no-seg-candidate"
    if not name_ok:
        return best, False, f"allpos-name-mismatch({objnames[oi]}!={target_name})"
    return best, True, "ok"


shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(VIZDIR, exist_ok=True)
ds = LeRobotDataset.create(REPO, fps=FPS, features=FEATURES, root=ROOT, use_videos=USE_VID, image_writer_threads=IWT)
files = sorted(glob.glob(RAW + "/ep_*.npz"))
print(f"converting {len(files)} episodes  K={K} G={G}", flush=True)
nviz, dropped, masked, fr_stats = 0, 0, 0, []
for k, f in enumerate(files):
    d = np.load(f, allow_pickle=True)
    img, wrist, state, action = d["img"], d["wrist"], d["state"], d["action"]
    seg, dep = d["seg"].astype(np.int32), d["depth"].astype(np.float32)
    lang, tname = str(d["lang"]), str(d["target_name"])
    mid, ok, why = moving_id(seg, d["allpos"], d["objnames"], tname)
    # KEEP_UNATTRIBUTED=1: do NOT drop episodes whose moving-object id can't be name-verified; KEEP them with
    # foresight_valid=0 + a zeroed region (the action objective still trains on them, the foresight loss is masked).
    # This makes the dataset a byte-identical episode set to the action-only arm (no size/normalization confound in
    # the C5 comparison) while the foresight branch still only ever sees cleanly-attributed targets. (LAB B25/§2.)
    KEEP_UNATTR = os.environ.get("KEEP_UNATTRIBUTED", "0") == "1"
    unattr = not ok
    if unattr and not KEEP_UNATTR:
        dropped += 1
        print(f"  DROP {os.path.basename(f)}: {why}", flush=True)
        continue
    if unattr:
        masked += 1
        print(f"  KEEP-valid0 {os.path.basename(f)}: {why} (action trains; foresight loss masked)", flush=True)
    T = len(img)
    for i in range(T):
        j = i + K
        valid = 1.0 if (j < T and not unattr) else 0.0
        j = min(j, T - 1)
        reg = np.zeros((G, G), np.float32) if unattr else block_max((seg[j] == mid).astype(np.float32), G)
        dpt = block_mean(dep[j], G)
        if valid:
            fr_stats.append(float(reg.mean()))
        ds.add_frame({"observation.images.image": img[i], "observation.images.wrist_image": wrist[i],
                      "observation.state": state[i].astype(np.float32),
                      "action": action[i].astype(np.float32),
                      "observation.foresight_region": reg.astype(np.float32),
                      "observation.foresight_depth": dpt.astype(np.float32),
                      "observation.foresight_valid": np.array([valid], np.float32),
                      "task": lang})
    ds.save_episode()
    # §3.6 eyeball: upsample the 16x16 region GT back onto the t+K frame and save
    if nviz < VIZ:
        try:
            from PIL import Image
            t = int(T * 0.35); j = min(t + K, T - 1)     # PRE-GRASP-ish frame = where grounding pressure is highest
            reg = block_max((seg[j] == mid).astype(np.float32), G)
            up = np.kron(reg, np.ones((256 // G, 256 // G), np.float32)) > 0.5
            ov = img[j].copy(); ov[up] = (0.45 * ov[up] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
            Image.fromarray(ov).save(f"{VIZDIR}/{tname}_t{t}_regionGT.png")
            nviz += 1
        except Exception as e:
            print("  viz fail", e, flush=True)
    if k % 100 == 0:
        print(f"   {k}/{len(files)}", flush=True)
ds.finalize()
print(f"CONVERT_DONE episodes={ds.num_episodes} frames={ds.num_frames} dropped={dropped} masked_valid0={masked} at {ROOT}", flush=True)
if fr_stats:
    print(f"foresight.region mean-coverage={np.mean(fr_stats):.3f} (fraction of the {G}x{G} grid that is the target; "
          f"expect a small compact blob ~0.02-0.10, NOT ~0.5 which would mean the mask is the arm/scene)", flush=True)
print(f"§3.6: scp {VIZDIR}/*.png and LOOK — the red {G}x{G} region must sit ON the NAMED object.", flush=True)
