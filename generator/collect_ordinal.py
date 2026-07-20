"""collect_ordinal.py (/root/rc_venv) — RoboCasa PnP ORDINAL-referent DECONFOUNDED oracle collector (Galahad G1).

FORK of collect_rc_deconf.py (B45). MOTOR IS REUSED VERBATIM (servo / step_arm / phase machine / base_R) — do NOT re-tune.
The ONLY changes vs B45: scene = a ROW of N objects, target named by ORDINAL ("the second object from the left"),
and role-balanced ordinal selection. Confound to break: ordinal ⟂ object-identity ⟂ absolute-position ⇒ the model
must COUNT/ORDER, not go to a fixed slot nor a fixed object.

DECONF construction (per episode):
  - N ∈ {3,4,5} (randomized, role-balanced), sample N DISTINCT categories from POOL, park the rest OUT OF FRAME.
  - place the N in a ROW along ROW_AXIS (the world axis that maps to image left→right — SET BY PROBE/RENDER),
    randomized row-center + per-episode spacing + which category sits in which slot + small perpendicular jitter.
  - ordinal order = sort the N by (ROW_AXIS coord × ROW_SIGN) ascending == image left→right (probe-verified).
  - target ordinal k ∈ {1..N} role-balanced ⇒ target object = the k-th in that sorted order.
  - instruction = "pick the {ordinal} object from the left and place it in the sink"  (the ONLY predictor of target).
  => ordinal ⟂ identity (k-th identity randomized per ep), ⟂ absolute-slot (row-center randomized), role-balanced.

Records img/wrist/state(8)/action(7)/lang => LeRobot via npz_to_lerobot.py (B4 schema, unchanged).

Run one shard/GPU:
  RENDER=0 POOL=apple,banana,lemon,carrot,can NDEMOS=64 RAW=/dev/shm/ord_raw SHARD=0 \\
  ROW_AXIS=x ROW_SIGN=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 /root/rc_venv/bin/python collect_ordinal.py
"""
import os, sys, json, traceback
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
import robocasa.utils.object_utils as OU
from robosuite.utils.transform_utils import quat2axisangle, quat2mat

POOL = os.environ.get("POOL", "apple,banana,lemon,carrot,can").split(",")
NDEMOS = int(os.environ.get("NDEMOS", "64"))         # demos THIS shard collects
RAW = os.environ.get("RAW", "/dev/shm/ord_raw")
SHARD = int(os.environ.get("SHARD", "0"))
SEED = int(os.environ.get("SEED", str(7000 + SHARD)))
LAYOUT = int(os.environ.get("LAYOUT", "1"))
STYLE = int(os.environ.get("STYLE", "1"))
RENDER = os.environ.get("RENDER", "0") == "1"
CAMW = int(os.environ.get("CAMW", "256"))
MAXS = int(os.environ.get("MAXS", "520"))
DEBUG = os.environ.get("DEBUG", "0") == "1"
os.makedirs(RAW, exist_ok=True)
VIZ = os.environ.get("VIZ", "/dev/shm/ord_viz"); os.makedirs(VIZ, exist_ok=True)

# ---- obs-key contract (from B45 probe, verified) ----
CAM_AGENT = os.environ.get("CAM_AGENT", "robot0_agentview_right")
CAM_WRIST = os.environ.get("CAM_WRIST", "robot0_eye_in_hand")

# ---- MOTOR geometry (B45 — REUSED VERBATIM, do NOT re-tune) ----
HOVER = float(os.environ.get("HOVER", "0.12"))
GRASP_DZ = float(os.environ.get("GRASP_DZ", "-0.01"))
LIFT_Z = float(os.environ.get("LIFT_Z", "1.15"))
SINK_DROP_Z = float(os.environ.get("SINK_DROP_Z", "0.98"))
SERVO = float(os.environ.get("SERVO", "0.04"))
GRIP_CLOSE, GRIP_OPEN = 1.0, -1.0

# ---- ORDINAL row placement (ROW_AXIS/ROW_SIGN finalized by probe_ordinal render) ----
ROW_AXIS = os.environ.get("ROW_AXIS", "y").lower()        # 'x' or 'y' — the world axis the row spreads along (probe-set)
ROW_SIGN = float(os.environ.get("ROW_SIGN", "1"))         # +1/-1 so (coord*sign) ascending == the REF direction (probe-set)
REF = os.environ.get("REF", "left")                       # "left" (image-horizontal row) or "front" (image-depth row)
# reachable counter region — ANCHORED on B45's RENDER-VERIFIED region (launch_collect.sh X 1.74-1.88 Y -0.40..-0.16);
# the ROW axis is WIDENED post-probe to fit N objects. Finalized by probe_ordinal render before the gate.
REACH_X = tuple(float(v) for v in os.environ.get("REACH_X", "1.74,1.88").split(","))
REACH_Y = tuple(float(v) for v in os.environ.get("REACH_Y", "-0.44,-0.12").split(","))
SPACE_MIN = float(os.environ.get("SPACE_MIN", "0.055"))   # per-episode spacing sampled in [MIN,MAX] (B45 SPACE=0.085)
SPACE_MAX = float(os.environ.get("SPACE_MAX", "0.075"))
JITTER = float(os.environ.get("JITTER", "0.015"))          # perpendicular jitter (keep small: order must stay clean)
PARK = (float(os.environ.get("PARK_X", "3.5")), float(os.environ.get("PARK_Y", "3.5")))  # out-of-frame park

ORDW = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}


class OrdinalPnP(PickPlaceCounterToSink):
    """5-object superset env (B45-proven build). obj_0..obj_4 = POOL cats; base class keeps its 'obj' via name reuse."""
    def __init__(self, pool, *a, **k):
        self._pool = list(pool)
        super().__init__(obj_groups=pool[0], *a, **k)

    def _get_obj_cfgs(self):
        # one object per POOL category; the FIRST keeps base-class name "obj" so any base ref to self.obj still resolves.
        reg = dict(fixture=self.counter, sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                   size=(0.55, 0.40), pos=("ref", -1.0))
        cfgs = []
        for i, c in enumerate(self._pool):
            nm = "obj" if i == 0 else "distr_%d" % (i - 1)
            cfgs.append(dict(name=nm, obj_groups=c, graspable=True, placement=dict(reg)))
        return cfgs


def make_env(seed):
    from robosuite.controllers import load_composite_controller_config
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = OrdinalPnP(
        pool=POOL,
        robots="PandaOmron", controller_configs=cc,
        camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=CAMW, camera_heights=CAMW,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
        ignore_done=True, seed=seed, layout_ids=LAYOUT, style_ids=STYLE,
        translucent_robot=False, obj_instance_split=None, generative_textures=None,
    )
    return env


def objpos(obs, name):
    return np.asarray(obs[name + "_pos"], np.float32)


def _obj_joint_addr(env, name):
    obj = env.objects[name]
    jnts = getattr(obj, "joints", None) or [name + "_joint0", name + "_main", name + "_joint"]
    for j in jnts:
        try:
            a = env.sim.model.get_joint_qpos_addr(j)
            return a[0] if isinstance(a, tuple) else a
        except Exception:
            continue
    return None


def place_row(env, rng, pin_N=None):
    """Choose N, subset, teleport N into a ROW (randomized center/spacing/order/jitter), park the rest out of frame.
    pin_N forces a specific N (to hit a scheduled role-balance cell). Returns (left_to_right_names, N, obs, meta)."""
    all_names = list(env.objects.keys())              # obj, distr_0..distr_3  (len == len(POOL))
    N = int(pin_N) if pin_N is not None else int(rng.choice([3, 4, 5]))
    N = min(N, len(all_names))
    subset = list(rng.choice(all_names, size=N, replace=False))
    parked = [n for n in all_names if n not in subset]
    spacing = float(rng.uniform(SPACE_MIN, SPACE_MAX))
    rowspan = (N - 1) * spacing

    if ROW_AXIS == "x":
        lo, hi = REACH_X; olo, ohi = REACH_Y; ai, oi = 0, 1
    else:
        lo, hi = REACH_Y; olo, ohi = REACH_X; ai, oi = 1, 0
    # SLIDING row (random start), NOT centered — lets every ordinal (incl. the middle) roam the full span
    # => maximizes ordinal ⟂ absolute-position. If the row can't fit, fall back to centered.
    if hi - lo > rowspan:
        start = float(rng.uniform(lo, hi - rowspan))
    else:
        start = (lo + hi - rowspan) / 2.0
    other_c = float(rng.uniform(olo + 0.02, ohi - 0.02))

    # slot coords along ROW_AXIS (left->right in world order), then RANDOMLY assign identities to slots
    slot_coords = start + np.arange(N) * spacing
    order = rng.permutation(N)                          # which subset-object goes to which physical slot
    placed = []
    for slot in range(N):
        name = subset[order[slot]]
        a0 = _obj_joint_addr(env, name)
        if a0 is None:
            continue
        z = float(env.sim.data.qpos[a0 + 2])
        pos = [0.0, 0.0, z + 0.02]
        pos[ai] = float(slot_coords[slot])
        pos[oi] = float(other_c + rng.uniform(-JITTER, JITTER))
        env.sim.data.qpos[a0:a0 + 3] = pos
        placed.append(name)
    for name in parked:                                # park extras far out of frame (B46 out-of-frustum trick)
        a0 = _obj_joint_addr(env, name)
        if a0 is None:
            continue
        z = float(env.sim.data.qpos[a0 + 2])
        env.sim.data.qpos[a0:a0 + 3] = [PARK[0], PARK[1], z]
    env.sim.forward()
    obs = None
    for _ in range(15):
        obs, _, _, _ = env.step(np.zeros(env.action_dim))

    # ordinal order = sort placed objects by (ROW_AXIS coord * ROW_SIGN) ascending == image LEFT->RIGHT (probe-verified)
    def keyf(nm):
        return ROW_SIGN * float(objpos(obs, nm)[ai])
    lr = sorted(placed, key=keyf)
    center = start + rowspan / 2.0                       # row center (for red-team ordinal⟂position analysis)
    meta = dict(N=N, spacing=round(spacing, 3), center=round(center, 3), start=round(start, 3),
                other=round(other_c, 3), axis=ROW_AXIS, sign=ROW_SIGN,
                cats={nm: env.objects[nm].name if hasattr(env.objects[nm], "name") else nm for nm in placed})
    return lr, N, obs, meta


def mkstate(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)


def servo(eef, wp, base_R):
    dw = np.asarray(wp, np.float32) - np.asarray(eef, np.float32)
    db = base_R.T @ dw
    return np.clip(db / SERVO, -1, 1).astype(np.float32)


def step_arm(env, robot, dpos, grip):
    arm = np.concatenate([np.asarray(dpos, np.float32), np.zeros(3, np.float32)])
    ad = {"right": arm, "right_gripper": np.array([grip], np.float32)}
    full = robot.create_action_vector(ad)
    return env.step(full)


def sink_xy(env):
    try:
        p = env.sink.pos
        return np.array([p[0], p[1]], np.float32)
    except Exception:
        return None


SINK_XY_TOL = float(os.environ.get("SINK_XY_TOL", "0.15"))
SINK_Z_MAX = float(os.environ.get("SINK_Z_MAX", "0.95"))
USE_OFFICIAL = os.environ.get("USE_OFFICIAL_SUCC", "0") == "1"  # flip on after probe confirms the official util matches


def target_in_sink(env, obs, name, sxy):
    """Success for the CHOSEN ordinal target (base _check_success tracks 'obj' only). Geometric by default
    (target xy near sink basin center + dropped below counter carry height); calibrated via probe (SINK_XY_TOL/Z_MAX).
    If USE_OFFICIAL, defer to RoboCasa's own in-sink containment (wired after inspecting _check_success in the probe)."""
    p = objpos(obs, name)                                # object world pos from the proven obs contract
    if USE_OFFICIAL:
        try:
            import robocasa.utils.object_utils as OU
            # mirror PickPlaceCounterToSink._check_success EXACTLY, parameterized by the DYNAMIC target name
            return bool(OU.obj_inside_of(env, name, env.sink, partial_check=True))
        except Exception:
            pass
    if sxy is None:
        return False
    return bool(np.linalg.norm(p[:2] - sxy) < SINK_XY_TOL and p[2] < SINK_Z_MAX)


def rollout(env, obs, target_name, record=True):
    """B45 phase machine VERBATIM; only 'target' is now the chosen ordinal object's sim name + manual sink success."""
    robot = env.robots[0]
    tp0 = objpos(obs, target_name)
    sxy = sink_xy(env)
    ph, grip, hold, rel = "hover", GRIP_OPEN, 0, 0
    im, wr, st, ac = [], [], [], []
    frames = []
    dmin, dmin_xy, obj_zmax, last_ph = 1e9, 1e9, tp0[2], None
    base_R = quat2mat(np.asarray(obs["robot0_base_quat"], np.float64))
    for s in range(MAXS):
        eef = np.asarray(obs["robot0_eef_pos"], np.float32)
        tp = objpos(obs, target_name)
        d = float(np.linalg.norm(eef - tp)); dmin = min(dmin, d)
        dxy = float(np.linalg.norm(eef[:2] - tp[:2])); dmin_xy = min(dmin_xy, dxy)
        obj_zmax = max(obj_zmax, float(tp[2]))
        if DEBUG and ph != last_ph:
            print("    ph=%-8s s=%3d dxy=%.3f dz=%.3f" % (ph, s, dxy, float(eef[2] - tp[2])), flush=True)
            last_ph = ph
        if ph == "hover":
            wp = [tp[0], tp[1], tp[2] + HOVER]; grip = GRIP_OPEN
            if np.linalg.norm(eef[:2] - tp[:2]) < 0.02 and abs(eef[2] - (tp[2] + HOVER)) < 0.03:
                ph = "descend"
        elif ph == "descend":
            gz = tp[2] + GRASP_DZ; wp = [tp[0], tp[1], gz]; grip = GRIP_OPEN
            if eef[2] < gz + 0.02: ph = "grasp"; hold = 0
        elif ph == "grasp":
            wp = [tp[0], tp[1], eef[2]]; grip = GRIP_CLOSE; hold += 1
            if hold >= 10: ph = "lift"
        elif ph == "lift":
            wp = [tp[0], tp[1], LIFT_Z]; grip = GRIP_CLOSE
            if eef[2] > LIFT_Z - 0.04: ph = "carry"
        elif ph == "carry":
            tgt = sxy if sxy is not None else tp[:2]
            wp = [tgt[0], tgt[1], LIFT_Z]; grip = GRIP_CLOSE
            if np.linalg.norm(eef[:2] - tgt) < 0.05: ph = "lower"
        elif ph == "lower":
            tgt = sxy if sxy is not None else eef[:2]
            wp = [tgt[0], tgt[1], SINK_DROP_Z]; grip = GRIP_CLOSE
            if eef[2] < SINK_DROP_Z + 0.03: ph = "release"
        elif ph == "release":
            wp = [eef[0], eef[1], SINK_DROP_Z]; grip = GRIP_OPEN; rel += 1
            if rel >= 6: ph = "retract"
        else:
            wp = [eef[0], eef[1], LIFT_Z + 0.22]; grip = GRIP_OPEN; rel += 1
        dpos = servo(eef, wp, base_R)
        if record:
            im.append(np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1]))
            wr.append(np.ascontiguousarray(np.asarray(obs[CAM_WRIST + "_image"])[::-1]))
            st.append(mkstate(obs))
            ac.append(np.concatenate([dpos, np.zeros(3, np.float32), [grip]]).astype(np.float32))
        if RENDER and s % 8 == 0:
            frames.append(np.asarray(obs[CAM_AGENT + "_image"])[::-1])
        obs, r, done, info = step_arm(env, robot, dpos, grip)
        succ_now = target_in_sink(env, obs, target_name, sxy)
        if ph == "retract" and rel >= 22 and succ_now:
            break
        if rel >= 30:
            break
    succ = target_in_sink(env, obs, target_name, sxy)
    if DEBUG:
        print("  END succ=%s dmin=%.3f obj_lift=%.3f ph=%s" % (succ, dmin, obj_zmax - tp0[2], ph), flush=True)
    return succ, (im, wr, st, ac), frames


def ordinal_schedule(ndemos, shard):
    """Role-balanced (N,k) cells: N in {3,4,5}, k in 1..N. Cycle cells (offset by shard) so counts are ~equal."""
    cells = []
    for N in (3, 4, 5):
        for k in range(1, N + 1):
            cells.append((N, k))
    out = []
    for i in range(ndemos):
        out.append(cells[(i + shard) % len(cells)])
    return out


def main():
    rng = np.random.RandomState(SEED)
    env = make_env(SEED)
    sched = ordinal_schedule(NDEMOS, SHARD)
    got, tries = 0, 0
    per_cell = {}
    while got < NDEMOS and tries < NDEMOS * 8:
        want_N, want_k = sched[got]
        tries += 1
        env.reset()
        try:
            lr, N, obs, meta = place_row(env, rng, pin_N=want_N)   # pin N to hit the role-balance cell exactly
        except Exception as e:
            print("PLACE_ERR", e, flush=True); traceback.print_exc(); continue
        k = min(want_k, N)
        target_name = lr[k - 1]                          # k-th in the REF direction (1-indexed)
        lang = "pick the %s object from the %s and place it in the sink" % (ORDW[k], REF)
        if DEBUG:
            print("EP got=%d N=%d k=%d target=%s lr=%s meta=%s" % (got, N, k, target_name, lr, meta), flush=True)
        try:
            succ, data, frames = rollout(env, obs, target_name, record=True)
        except Exception as e:
            print("ROLLOUT_ERR", e, flush=True); traceback.print_exc(); continue
        if RENDER and frames:
            strip = np.concatenate([f for f in frames[:16]], axis=1)
            imageio.imwrite("%s/roll_s%d_N%d_k%d_%d_%s.png" % (VIZ, SHARD, N, k, tries, "S" if succ else "F"), strip)
        if not succ:
            continue
        im, wr, st, ac = data
        tp = objpos(obs, target_name)
        np.savez_compressed(
            RAW + "/ep_%d_%04d.npz" % (SHARD, got),
            img=np.asarray(im, np.uint8), wrist=np.asarray(wr, np.uint8),
            state=np.asarray(st, np.float32), action=np.asarray(ac, np.float32),
            lang=lang, ordinal=k, n_obj=N, target_name=target_name,
            target_xy=np.asarray(tp[:2], np.float32),
            row_center=meta["center"], spacing=meta["spacing"], axis=meta["axis"],
        )
        per_cell[(N, k)] = per_cell.get((N, k), 0) + 1
        got += 1
    print("SHARD%d_DONE total=%d tries=%d cells=%s" % (SHARD, got, tries, json.dumps({str(kk): v for kk, v in per_cell.items()})), flush=True)
    env.close()


if __name__ == "__main__":
    main()
