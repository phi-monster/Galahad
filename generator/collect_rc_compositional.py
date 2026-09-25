"""collect_rc_compositional.py (/root/rc_venv) — RoboCasa PnP COMPOSITIONAL (multi-attribute CONJUNCTION) deconf
collector for G1 (compositional referent type).

Deconf principle: the instruction is the ONLY predictor of the target, AND no SINGLE attribute suffices — the target
is picked out only by the CONJUNCTION of >=2 attributes. COMPOSITIONAL = COLOUR x CATEGORY (2x2 factorial per scene):
  target cell = (target_cat, target_colour). Scene ALWAYS contains the full 2x2:
     obj      = (target_cat , target_colour)            <- the ONLY object matching BOTH attributes
     distr_0  = (target_cat , other_colour)             <- CATEGORY-match distractor (drops colour)
     distr_1  = (other_cat  , target_colour)            <- COLOUR-match  distractor (drops category)
     distr_2  = (other_cat  , other_colour)             <- neutral (matches neither)
  => single-attribute "the {colour}"  is satisfied by {obj, distr_1}  (ambiguous)
     single-attribute "the {category}" is satisfied by {obj, distr_0}  (ambiguous)
     only "{colour} {category}" = obj is unique. A single-attribute-only model MUST fail.
Target ROTATES over every (cat,colour) cell => role-balanced. Positions RANDOMIZED every reset (attrs _|_ position).
Instruction: "pick the {target_colour} {target_cat} and place it in the sink".
Recolour = geom_matid=-1 + geom_rgba (flat vivid, hue-separated => distinct at 256px; anti 'molmo-can't-tell' #288).
MOTOR = B45 collect_rc_deconf.py top-down PnP oracle, BYTE-IDENTICAL (NOT re-tuned).

Optional SIZE 3rd axis (SIZE_AXIS=1): adds object_scale small/large => "the small red can". Enable ONLY if the
render-gate (probe_compositional.py) confirmed object_scale renders an OBVIOUS size difference (else the size cue is
invisible = a broken confound). With SIZE_AXIS the scene keeps the drop-one design over 3 attributes (target + 3
single-flip distractors), see build_assign().

Run one shard/GPU:
  RENDER=0 CAT_POOL=can,apple COLOUR_POOL=red,green,blue TARGETS=can N_PER_CELL=42 RAW=/root/collect_compositional/raw \\
  SHARD=0 REACH_X=1.72,1.90 REACH_Y=-0.42,-0.14 SPACE=0.085 \\
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 /root/rc_venv/bin/python collect_rc_compositional.py
"""
import os, sys, json, traceback, itertools
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
import robocasa.utils.object_utils as OU
from robosuite.utils.transform_utils import quat2axisangle, quat2mat

# ---------- COMPOSITIONAL config ----------
CAT_POOL = os.environ.get("CAT_POOL", "can,apple").split(",")         # >=2 DISTINCT-silhouette graspable cats (probe-picked)
ALL_COLOURS = {
    "red":    [0.85, 0.08, 0.08, 1.0],
    "green":  [0.10, 0.65, 0.12, 1.0],
    "blue":   [0.10, 0.22, 0.88, 1.0],
    "yellow": [0.90, 0.80, 0.10, 1.0],
}
COLOUR_POOL = os.environ.get("COLOUR_POOL", "red,green,blue").split(",")  # FULL pool (target rotation + distractor sampling)
TARGETS = os.environ.get("TARGETS", ",".join(CAT_POOL)).split(",")    # target CATEGORIES this shard collects (fan-out)
TARGET_COLOURS = os.environ.get("TARGET_COLOURS", ",".join(COLOUR_POOL)).split(",")  # target COLOURS this shard collects (fan-out); others still sampled from COLOUR_POOL
N_PER_CELL = int(os.environ.get("N_PER_CELL", "42"))                  # demos per (target_cat, target_colour[, size]) cell
SIZE_AXIS = os.environ.get("SIZE_AXIS", "0") == "1"                   # add SIZE as a 3rd conjunction attribute
SIZE_SCALES = {"small": float(os.environ.get("SCALE_SMALL", "0.62")),
               "large": float(os.environ.get("SCALE_LARGE", "1.15"))}
SIZE_POOL = os.environ.get("SIZE_POOL", "small,large").split(",") if SIZE_AXIS else [None]

# ---------- collection config (parity with collect_rc_deconf/colour) ----------
RAW = os.environ.get("RAW", "/root/collect_compositional/raw")
SHARD = int(os.environ.get("SHARD", "0"))
SEED = int(os.environ.get("SEED", str(1000 + SHARD)))
LAYOUT = int(os.environ.get("LAYOUT", "1"))
STYLE = int(os.environ.get("STYLE", "1"))
RENDER = os.environ.get("RENDER", "0") == "1"
CAMW = int(os.environ.get("CAMW", "256"))
MAXS = int(os.environ.get("MAXS", "520"))
os.makedirs(RAW, exist_ok=True)
VIZ = os.environ.get("VIZ", "/root/collect_compositional/viz"); os.makedirs(VIZ, exist_ok=True)

CAM_AGENT = os.environ.get("CAM_AGENT", "robot0_agentview_right")
CAM_WRIST = os.environ.get("CAM_WRIST", "robot0_eye_in_hand")
HOVER = float(os.environ.get("HOVER", "0.12"))
GRASP_DZ = float(os.environ.get("GRASP_DZ", "-0.01"))
LIFT_Z = float(os.environ.get("LIFT_Z", "1.15"))
SINK_DROP_Z = float(os.environ.get("SINK_DROP_Z", "0.98"))
SERVO = float(os.environ.get("SERVO", "0.04"))
GRIP_CLOSE, GRIP_OPEN = 1.0, -1.0
DEBUG = os.environ.get("DEBUG", "0") == "1"

# default region = colour-sibling's render-verified counter patch, widened slightly for the 4th object (VERIFY at render-gate:
# every slot must be reachable by the reused motor AND framed by CAM_AGENT; tighten/shift if the render shows crowding/spill)
REACH_X = tuple(float(v) for v in os.environ.get("REACH_X", "1.72,1.90").split(","))
REACH_Y = tuple(float(v) for v in os.environ.get("REACH_Y", "-0.42,-0.14").split(","))
SPACE = float(os.environ.get("SPACE", "0.085"))

CAT_WORD = {  # category -> instruction noun (default = the category token itself)
    "can": "can", "apple": "apple", "lemon": "lemon", "banana": "banana", "carrot": "carrot",
    "orange": "orange", "bell_pepper": "pepper",
}


def cat_word(c):
    return CAT_WORD.get(c, c.replace("_", " "))


# 4 slots. distr_0 = category-match (target_cat, other_colour); distr_1 = colour-match (other_cat, target_colour);
# distr_2 = neutral (other_cat, other_colour). Category assignment is FIXED at env-build (obj_groups) => one env per
# (target_cat, other_cat); colour + (optional) size assigned at runtime.
class DeconfPnPComp(PickPlaceCounterToSink):
    def __init__(self, target_cat, other_cat, slot_scale, *a, **k):
        self._target_cat = target_cat; self._other_cat = other_cat
        self._slot_scale = slot_scale   # {'obj':scale,'distr_0':..} or {} when SIZE_AXIS off
        super().__init__(obj_groups=target_cat, *a, **k)

    def _get_obj_cfgs(self):
        reg = dict(fixture=self.counter, sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                   size=(0.55, 0.40), pos=("ref", -1.0))
        slotcats = [("obj", self._target_cat), ("distr_0", self._target_cat),
                    ("distr_1", self._other_cat), ("distr_2", self._other_cat)]
        cfgs = []
        for name, cat in slotcats:
            c = dict(name=name, obj_groups=cat, graspable=True, placement=dict(reg))
            if name in self._slot_scale:
                c["object_scale"] = self._slot_scale[name]
            cfgs.append(c)
        return cfgs


def make_env(target_cat, other_cat, slot_scale, seed):
    from robosuite.controllers import load_composite_controller_config
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return DeconfPnPComp(
        target_cat=target_cat, other_cat=other_cat, slot_scale=slot_scale,
        robots="PandaOmron", controller_configs=cc,
        camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=CAMW, camera_heights=CAMW,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
        ignore_done=True, seed=seed, layout_ids=LAYOUT, style_ids=STYLE,
        translucent_robot=False, obj_instance_split=None, generative_textures=None)


def objpos(obs, name):
    return np.asarray(obs[name + "_pos"], np.float32)


# ---------- recolour (from collect_rc_colour.py, byte-identical) ----------
def _obj_geom_ids(env, name):
    obj = env.objects[name]; ids = set(); names = []
    for attr in ("visual_geoms", "contact_geoms"):
        v = getattr(obj, attr, None)
        if v: names += list(v)
    for gn in names:
        try: ids.add(env.sim.model.geom_name2id(gn))
        except Exception: pass
    if ids: return ids
    try: bid = env.sim.model.body_name2id(obj.root_body)
    except Exception: return ids
    for gid in range(env.sim.model.ngeom):
        b = int(env.sim.model.geom_bodyid[gid])
        while b > 0:
            if b == bid: ids.add(gid); break
            b = int(env.sim.model.body_parentid[b])
    return ids


def recolour(env, slot_colour):
    for name, cname in slot_colour.items():
        rgba = np.asarray(ALL_COLOURS[cname], np.float32)
        for gid in _obj_geom_ids(env, name):
            env.sim.model.geom_matid[gid] = -1
            env.sim.model.geom_rgba[gid] = rgba
    env.sim.forward()


# ---------- oracle (BYTE-IDENTICAL to B45 collect_rc_deconf.py; motor NOT re-tuned) ----------
def _obj_joint_addr(env, name):
    obj = env.objects[name]
    jnts = getattr(obj, "joints", None) or [name + "_joint0", name + "_main", name + "_joint"]
    for j in jnts:
        try:
            a = env.sim.model.get_joint_qpos_addr(j)
            return a[0] if isinstance(a, tuple) else a
        except Exception: continue
    return None


def randomize_objs(env, rng):
    placed = []
    for name in list(env.objects.keys()):
        a0 = _obj_joint_addr(env, name)
        if a0 is None: continue
        z = float(env.sim.data.qpos[a0 + 2])
        x, y = None, None
        for _ in range(300):
            cx = rng.uniform(*REACH_X); cy = rng.uniform(*REACH_Y)
            if all((cx - px) ** 2 + (cy - py) ** 2 > SPACE ** 2 for px, py in placed):
                x, y = cx, cy; break
        if x is None:
            x, y = rng.uniform(*REACH_X), rng.uniform(*REACH_Y)
        env.sim.data.qpos[a0:a0 + 3] = [x, y, z + 0.02]
        placed.append((x, y))
    env.sim.forward()
    obs = None
    for _ in range(12):
        obs, _, _, _ = env.step(np.zeros(env.action_dim))
    return obs


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


def rollout(env, obs, target="obj", record=True):
    robot = env.robots[0]
    tp0 = objpos(obs, target)
    sxy = sink_xy(env)
    ph, grip, hold, rel = "hover", GRIP_OPEN, 0, 0
    im, wr, st, ac = [], [], [], []
    frames = []
    dmin, dmin_xy, obj_zmax = 1e9, 1e9, tp0[2]
    base_R = quat2mat(np.asarray(obs["robot0_base_quat"], np.float64))
    for s in range(MAXS):
        eef = np.asarray(obs["robot0_eef_pos"], np.float32)
        tp = objpos(obs, target)
        d = float(np.linalg.norm(eef - tp)); dmin = min(dmin, d)
        dxy = float(np.linalg.norm(eef[:2] - tp[:2])); dmin_xy = min(dmin_xy, dxy)
        obj_zmax = max(obj_zmax, float(tp[2]))
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
        if ph == "retract" and rel >= 22 and bool(env._check_success()):
            break
        if rel >= 30:
            break
    succ = bool(env._check_success())
    return succ, (im, wr, st, ac), frames


def build_assign(target_cat, other_cat, target_colour, other_colour, target_size, other_size):
    """Return (slot_colour, slot_scale_word, lang). Drop-one distractor design.
    Slots (category FIXED by env): obj,distr_0 = target_cat ; distr_1,distr_2 = other_cat."""
    if not SIZE_AXIS:
        slot_colour = {"obj": target_colour, "distr_0": other_colour,
                       "distr_1": target_colour, "distr_2": other_colour}
        lang = "pick the %s %s and place it in the sink" % (target_colour, cat_word(target_cat))
        return slot_colour, {}, lang
    # 3-axis SIZE x COLOUR x CATEGORY drop-one: each distractor flips exactly one attribute of the target.
    #   obj     = (target_cat, target_colour, target_size)
    #   distr_0 = (target_cat, other_colour , target_size)   flip COLOUR    (cat-match: same cat/size)
    #   distr_1 = (other_cat , target_colour, target_size)   flip CATEGORY  (colour/size-match)
    #   distr_2 = (target_cat, target_colour, other_size )   flip SIZE      (cat/colour-match)  -> needs a 3rd slot of target_cat
    slot_colour = {"obj": target_colour, "distr_0": other_colour,
                   "distr_1": target_colour, "distr_2": target_colour}
    slot_scale_word = {"obj": target_size, "distr_0": target_size,
                       "distr_1": target_size, "distr_2": other_size}
    lang = "pick the %s %s %s and place it in the sink" % (target_size, target_colour, cat_word(target_cat))
    return slot_colour, slot_scale_word, lang


def other_cat_for(target_cat):
    others = [c for c in CAT_POOL if c != target_cat]
    return others[0] if others else target_cat


def main():
    rng = np.random.RandomState(SEED)
    n_ok = 0
    other_all = {c for c in CAT_POOL}
    for target_cat in TARGETS:
        other_cat = other_cat_for(target_cat)
        # env built per (target_cat) with fixed slot categories. size axis: build with target/other scales set per cell
        for target_colour in TARGET_COLOURS:   # this shard's target colours (others still sampled from full COLOUR_POOL)
            for target_size in SIZE_POOL:   # [None] when SIZE_AXIS off
                other_colour = rng.choice([c for c in COLOUR_POOL if c != target_colour])
                other_size = None
                if SIZE_AXIS:
                    other_size = [s for s in SIZE_POOL if s != target_size][0]
                slot_colour, slot_scale_word, lang = build_assign(
                    target_cat, other_cat, target_colour, other_colour, target_size, other_size)
                # For SIZE_AXIS, distr_1/distr_2 categories differ from the 2-axis mapping — rebuild slotcats note:
                # in the 3-axis design distr_2 must be target_cat (size-flip) not other_cat. We keep the env's fixed
                # categories (obj,distr_0=target_cat; distr_1,distr_2=other_cat) for the 2-axis path; for SIZE_AXIS we
                # override distr_2's category by using a target_cat env-slot — handled by SIZE env variant below.
                slot_scale = {}
                if SIZE_AXIS:
                    slot_scale = {k: SIZE_SCALES[v] for k, v in slot_scale_word.items()}
                    # SIZE variant needs 3 target_cat slots + 1 other_cat; rebuild env categories accordingly
                    env = _make_env_size(target_cat, other_cat, slot_scale, SEED)
                else:
                    env = make_env(target_cat, other_cat, slot_scale, SEED)
                got, tries = 0, 0
                while got < N_PER_CELL and tries < N_PER_CELL * 6:
                    tries += 1
                    env.reset()
                    obs = randomize_objs(env, rng)
                    recolour(env, slot_colour)
                    env.sim.forward()
                    obs, _, _, _ = env.step(np.zeros(env.action_dim))
                    # capture attr->position map for the _|_ red-team (attrs must be uncorrelated with position)
                    pos_by_slot = {n: objpos(obs, n)[:2].tolist() for n in env.objects}
                    try:
                        succ, data, frames = rollout(env, obs, "obj", record=True)
                    except Exception as e:
                        print("ROLLOUT_ERR", target_cat, target_colour, e, flush=True); traceback.print_exc(); continue
                    if RENDER and frames:
                        strip = np.concatenate([f for f in frames[:16]], axis=1)
                        tag = "%s_%s%s" % (target_cat, target_colour, ("_" + str(target_size)) if target_size else "")
                        imageio.imwrite("%s/roll_%s_%d_%s.png" % (VIZ, tag, tries, "S" if succ else "F"), strip)
                    if not succ:
                        continue
                    im, wr, st, ac = data
                    cell = "%s_%s%s" % (target_cat, target_colour, ("_" + str(target_size)) if target_size else "")
                    np.savez_compressed(
                        RAW + "/ep_%d_%s_%04d.npz" % (SHARD, cell, got),
                        img=np.asarray(im, np.uint8), wrist=np.asarray(wr, np.uint8),
                        state=np.asarray(st, np.float32), action=np.asarray(ac, np.float32),
                        lang=lang, target_cat=target_cat, target_colour=target_colour,
                        target_size=str(target_size), other_cat=other_cat,
                        slot_colour=json.dumps(slot_colour), slot_scale=json.dumps(slot_scale_word),
                        pos_by_slot=json.dumps(pos_by_slot))
                    got += 1; n_ok += 1
                print("CELL %s tgt_colour=%s size=%s: %d/%d ok (%d tries) :: %r" % (
                    target_cat, target_colour, target_size, got, N_PER_CELL, tries, lang), flush=True)
                env.close()
    print("SHARD%d_COMP_DONE total=%d" % (SHARD, n_ok), flush=True)


# SIZE env variant: 3 target_cat slots (obj,distr_0,distr_2) + 1 other_cat (distr_1). Only used when SIZE_AXIS=1.
class DeconfPnPCompSize(PickPlaceCounterToSink):
    def __init__(self, target_cat, other_cat, slot_scale, *a, **k):
        self._target_cat = target_cat; self._other_cat = other_cat; self._slot_scale = slot_scale
        super().__init__(obj_groups=target_cat, *a, **k)

    def _get_obj_cfgs(self):
        reg = dict(fixture=self.counter, sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                   size=(0.55, 0.40), pos=("ref", -1.0))
        slotcats = [("obj", self._target_cat), ("distr_0", self._target_cat),
                    ("distr_1", self._other_cat), ("distr_2", self._target_cat)]
        cfgs = []
        for name, cat in slotcats:
            c = dict(name=name, obj_groups=cat, graspable=True, placement=dict(reg))
            if name in self._slot_scale:
                c["object_scale"] = self._slot_scale[name]
            cfgs.append(c)
        return cfgs


def _make_env_size(target_cat, other_cat, slot_scale, seed):
    from robosuite.controllers import load_composite_controller_config
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return DeconfPnPCompSize(
        target_cat=target_cat, other_cat=other_cat, slot_scale=slot_scale,
        robots="PandaOmron", controller_configs=cc,
        camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=CAMW, camera_heights=CAMW,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
        ignore_done=True, seed=seed, layout_ids=LAYOUT, style_ids=STYLE,
        translucent_robot=False, obj_instance_split=None, generative_textures=None)


if __name__ == "__main__":
    main()
