"""collect_rc_colour.py (/root/rc_venv) — RoboCasa PnP COLOUR-DECONFOUNDED oracle collector (G1 colour type).

Deconf principle: instruction is the ONLY predictor of the target. COLOUR variant:
  - N objects of the SAME category (=> same shape) recoloured to N distinct vivid colours (red/green/blue...).
  - Target ROTATES over colours (each colour is target N_PER times => role-balanced).
  - Positions RANDOMIZED every reset (colour ⟂ position ⟂ shape[constant]).
  - Instruction names the COLOUR only ("pick the red object and place it in the sink").
  - Oracle grasps slot "obj" (=target colour). MOTOR = B45's byte-identical PnP oracle (NOT re-tuned).

Recolour = force flat vivid colour independent of baked mesh texture: geom_matid=-1 (detach material) + geom_rgba.

Run one shard/GPU:
  RENDER=0 BASE_CAT=can NCOL=3 N_PER_OBJ=24 RAW=/root/collect_colour/raw SHARD=0 \\
  REACH_X=1.74,1.88 REACH_Y=-0.40,-0.16 SPACE=0.085 \\
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 /root/rc_venv/bin/python collect_rc_colour.py
"""
import os, sys, json, traceback
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
import robocasa.utils.object_utils as OU
from robosuite.utils.transform_utils import quat2axisangle, quat2mat

# ---------- COLOUR config ----------
BASE_CAT = os.environ.get("BASE_CAT", "can")          # single category => same shape across all objects
# vivid, hue-separated, high-saturation so a VLM distinguishes them at 256px (anti "molmo-can't-tell" #288)
ALL_COLOURS = {
    "red":    [0.85, 0.08, 0.08, 1.0],
    "green":  [0.10, 0.65, 0.12, 1.0],
    "blue":   [0.10, 0.22, 0.88, 1.0],
    "yellow": [0.90, 0.80, 0.10, 1.0],
}
NCOL = int(os.environ.get("NCOL", "3"))               # #objects = #colours (1 target + NCOL-1 distractors)
COLOUR_NAMES = list(ALL_COLOURS.keys())[:NCOL]
COLOURS = {c: ALL_COLOURS[c] for c in COLOUR_NAMES}
OBJ_WORD = os.environ.get("OBJ_WORD", "object")       # instruction says "the red object" (no category leak)
TARGET_COLOURS = os.environ.get("TARGETS", ",".join(COLOUR_NAMES)).split(",")  # colours THIS shard targets

# ---------- collection config (parity with collect_rc_deconf.py) ----------
N_PER_OBJ = int(os.environ.get("N_PER_OBJ", "24"))
RAW = os.environ.get("RAW", "/root/collect_colour/raw")
SHARD = int(os.environ.get("SHARD", "0"))
SEED = int(os.environ.get("SEED", str(1000 + SHARD)))
LAYOUT = int(os.environ.get("LAYOUT", "1"))
STYLE = int(os.environ.get("STYLE", "1"))
RENDER = os.environ.get("RENDER", "0") == "1"
CAMW = int(os.environ.get("CAMW", "256"))
MAXS = int(os.environ.get("MAXS", "520"))
os.makedirs(RAW, exist_ok=True)
VIZ = os.environ.get("VIZ", "/root/collect_colour/viz"); os.makedirs(VIZ, exist_ok=True)

CAM_AGENT = os.environ.get("CAM_AGENT", "robot0_agentview_right")
CAM_WRIST = os.environ.get("CAM_WRIST", "robot0_eye_in_hand")
HOVER = float(os.environ.get("HOVER", "0.12"))
GRASP_DZ = float(os.environ.get("GRASP_DZ", "-0.01"))
LIFT_Z = float(os.environ.get("LIFT_Z", "1.15"))
SINK_DROP_Z = float(os.environ.get("SINK_DROP_Z", "0.98"))
SERVO = float(os.environ.get("SERVO", "0.04"))
GRIP_CLOSE, GRIP_OPEN = 1.0, -1.0
DEBUG = os.environ.get("DEBUG", "0") == "1"

REACH_X = tuple(float(v) for v in os.environ.get("REACH_X", "1.74,1.88").split(","))
REACH_Y = tuple(float(v) for v in os.environ.get("REACH_Y", "-0.40,-0.16").split(","))
SPACE = float(os.environ.get("SPACE", "0.085"))


class DeconfPnPColour(PickPlaceCounterToSink):
    """N objects, ALL of BASE_CAT (=> same shape). obj = target slot; distr_i = distractors."""
    def __init__(self, base_cat, n_obj, *a, **k):
        self._base_cat = base_cat; self._n_obj = n_obj
        super().__init__(obj_groups=base_cat, *a, **k)

    def _get_obj_cfgs(self):
        reg = dict(fixture=self.counter, sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                   size=(0.55, 0.40), pos=("ref", -1.0))
        cfgs = [dict(name="obj", obj_groups=self._base_cat, graspable=True, placement=dict(reg))]
        for i in range(self._n_obj - 1):
            cfgs.append(dict(name="distr_%d" % i, obj_groups=self._base_cat, graspable=True, placement=dict(reg)))
        return cfgs


def make_env(seed):
    from robosuite.controllers import load_composite_controller_config
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = DeconfPnPColour(
        base_cat=BASE_CAT, n_obj=NCOL,
        robots="PandaOmron", controller_configs=cc,
        camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=CAMW, camera_heights=CAMW,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
        ignore_done=True, seed=seed, layout_ids=LAYOUT, style_ids=STYLE,
        translucent_robot=False, obj_instance_split=None, generative_textures=None,
    )
    return env


def objpos(obs, name):
    return np.asarray(obs[name + "_pos"], np.float32)


# ---------- recolour (force flat vivid colour independent of baked texture) ----------
def _obj_geom_ids(env, name):
    obj = env.objects[name]
    ids = set()
    names = []
    for attr in ("visual_geoms", "contact_geoms"):
        v = getattr(obj, attr, None)
        if v:
            names += list(v)
    for gn in names:
        try:
            ids.add(env.sim.model.geom_name2id(gn))
        except Exception:
            pass
    if ids:
        return ids
    # fallback: all geoms in the object's root-body subtree
    try:
        bid = env.sim.model.body_name2id(obj.root_body)
    except Exception:
        return ids
    for gid in range(env.sim.model.ngeom):
        b = int(env.sim.model.geom_bodyid[gid])
        while b > 0:
            if b == bid:
                ids.add(gid); break
            b = int(env.sim.model.body_parentid[b])
    return ids


def recolour(env, slot_colour):
    """slot_colour: {'obj':'red','distr_0':'green',...} -> set flat vivid rgba on each object's geoms."""
    for name, cname in slot_colour.items():
        rgba = np.asarray(COLOURS[cname], np.float32)
        gids = _obj_geom_ids(env, name)
        for gid in gids:
            env.sim.model.geom_matid[gid] = -1   # detach material/texture -> use geom_rgba directly
            env.sim.model.geom_rgba[gid] = rgba
        if DEBUG:
            print("    recolour %s -> %s (%d geoms)" % (name, cname, len(gids)), flush=True)
    env.sim.forward()


def assign_colours(target_colour, rng, obj_names):
    others = [c for c in COLOUR_NAMES if c != target_colour]
    rng.shuffle(others)
    sc = {"obj": target_colour}
    dnames = [n for n in obj_names if n != "obj"]
    for i, dn in enumerate(dnames):
        sc[dn] = others[i % len(others)]
    return sc


# ---------- oracle (BYTE-IDENTICAL to B45 collect_rc_deconf.py; motor NOT re-tuned) ----------
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


def randomize_objs(env, rng):
    placed = []
    for name in list(env.objects.keys()):
        a0 = _obj_joint_addr(env, name)
        if a0 is None:
            continue
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
    tp0 = objpos(obs, target); z_top = tp0[2]
    sxy = sink_xy(env)
    ph, grip, hold, rel = "hover", GRIP_OPEN, 0, 0
    im, wr, st, ac = [], [], [], []
    frames = []
    dmin, dmin_xy, obj_zmax, last_ph = 1e9, 1e9, tp0[2], None
    eef0 = np.asarray(obs["robot0_eef_pos"], np.float32)
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


def main():
    rng = np.random.RandomState(SEED)
    env = make_env(SEED)
    n_ok = 0
    for target_colour in TARGET_COLOURS:
        got, tries = 0, 0
        while got < N_PER_OBJ and tries < N_PER_OBJ * 6:
            tries += 1
            env.reset()
            obj_names = list(env.objects.keys())   # env.objects only populated AFTER reset()
            obs = randomize_objs(env, rng)
            slot_colour = assign_colours(target_colour, rng, obj_names)
            recolour(env, slot_colour)
            # re-render obs after recolour so recorded frames carry the assigned colours
            env.sim.forward()
            obs, _, _, _ = env.step(np.zeros(env.action_dim))
            # capture colour->position map for the ⊥ red-team
            pos_by_colour = {slot_colour[n]: objpos(obs, n)[:2].tolist() for n in obj_names}
            if DEBUG:
                print("  target=%s assign=%s pos=%s" % (target_colour, slot_colour,
                      {k: np.round(v, 3).tolist() for k, v in pos_by_colour.items()}), flush=True)
            try:
                succ, data, frames = rollout(env, obs, "obj", record=True)
            except Exception as e:
                print("ROLLOUT_ERR", target_colour, e, flush=True); traceback.print_exc(); continue
            if RENDER and frames:
                sel = frames[::max(1, len(frames) // 16)][:16]   # sample ACROSS the full episode (incl. placement)
                strip = np.concatenate(sel, axis=1)
                imageio.imwrite("%s/roll_%s_%d_%s.png" % (VIZ, target_colour, tries, "S" if succ else "F"), strip)
            if not succ:
                continue
            im, wr, st, ac = data
            # INSTR phrasing MUST match the colour eval verbatim (train==eval; LAB B47 mismatch lesson).
            # colour is the ONLY disambiguating cue (no category word — all objects are BASE_CAT).
            lang = "pick the %s %s from the counter and place it in the sink" % (target_colour, OBJ_WORD)
            np.savez_compressed(RAW + "/ep_%d_%s_%04d.npz" % (SHARD, target_colour, got),
                                img=np.asarray(im, np.uint8), wrist=np.asarray(wr, np.uint8),
                                state=np.asarray(st, np.float32), action=np.asarray(ac, np.float32),
                                lang=lang, target_colour=target_colour,
                                colour_assign=json.dumps(slot_colour),
                                pos_by_colour=json.dumps(pos_by_colour))
            got += 1; n_ok += 1
        print("TARGET_COLOUR %s: %d/%d ok (%d tries)" % (target_colour, got, N_PER_OBJ, tries), flush=True)
    env.close()
    print("SHARD%d_DONE total=%d" % (SHARD, n_ok), flush=True)


if __name__ == "__main__":
    main()
