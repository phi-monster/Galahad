"""collect_rc_relation.py (/root/rc_venv) — RoboCasa PnP SPATIAL-RELATION-DECONFOUNDED oracle collector (G1 relation type).

Deconf principle: instruction is the ONLY predictor of the target. RELATION variant:
  - Target is defined by a SPATIAL RELATION to a REFERENCE object ("the can to the left of the banana",
    "the can next to the banana"). Scene = 2 SAME-category candidates (obj=target, distr_0=same-type distractor
    that does NOT satisfy the relation) + 1 REFERENCE object of a DIFFERENT category (the landmark).
  - The relation ROTATES over {left,right,next_to} (each is target N_PER times => role-balanced).
  - Positions RANDOMIZED every reset AND left/right balanced => the relation-satisfying object ⟂ absolute position
    ⟂ identity (both candidates same category, optionally painted the SAME colour). ONLY resolving the relation
    to the reference picks the target.
  - Instruction names the candidate category + the RELATION + the reference ("pick the {cand} {phrase} the {ref}
    and place it in the sink"). NO colour/identity leak.
  - Oracle grasps slot "obj" (=the target, which is placed on the relation-satisfying side). MOTOR = B45's
    byte-identical PnP oracle (NOT re-tuned). `obj` stays the success slot so env._check_success is correct;
    its MODEL is re-sampled each reset + its POSITION (which side) alternates with the relation => no shortcut.

Run one shard/GPU:
  RENDER=0 CAND_CAT=can REF_CAT=banana RELATIONS=left,right,next_to N_PER_OBJ=24 RAW=/root/collect_relation/raw SHARD=0 \\
  REACH_X=1.74,1.88 REACH_Y=-0.42,-0.14 SPACE=0.075 LATERAL_AXIS=y LATERAL_SIGN=1 \\
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 /root/rc_venv/bin/python collect_rc_relation.py
"""
import os, sys, json, math, traceback
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
import robocasa.utils.object_utils as OU
from robosuite.utils.transform_utils import quat2axisangle, quat2mat

# ---------- RELATION config ----------
CAND_CAT = os.environ.get("CAND_CAT", "can")          # the 2 same-type candidates (target + distractor)
REF_CAT = os.environ.get("REF_CAT", "banana")         # the reference landmark (DIFFERENT category => "the {ref}")
RELATIONS = os.environ.get("RELATIONS", "next_to,far_from").split(",")   # proximity relations (deconf-clean: ref roams)
PHRASE = {"left": "to the left of", "right": "to the right of", "next_to": "next to", "far_from": "far from",
          "front": "in front of", "behind": "behind"}

# ---------- collection config (parity with collect_rc_deconf.py / collect_rc_colour.py) ----------
N_PER_OBJ = int(os.environ.get("N_PER_OBJ", "24"))    # demos PER RELATION for this shard
RAW = os.environ.get("RAW", "/root/collect_relation/raw")
SHARD = int(os.environ.get("SHARD", "0"))
SEED = int(os.environ.get("SEED", str(1000 + SHARD)))
LAYOUT = int(os.environ.get("LAYOUT", "1"))
STYLE = int(os.environ.get("STYLE", "1"))
RENDER = os.environ.get("RENDER", "0") == "1"
CAMW = int(os.environ.get("CAMW", "256"))
MAXS = int(os.environ.get("MAXS", "520"))
os.makedirs(RAW, exist_ok=True)
VIZ = os.environ.get("VIZ", "/root/collect_relation/viz"); os.makedirs(VIZ, exist_ok=True)

CAM_AGENT = os.environ.get("CAM_AGENT", "robot0_agentview_right")
CAM_WRIST = os.environ.get("CAM_WRIST", "robot0_eye_in_hand")
HOVER = float(os.environ.get("HOVER", "0.12"))
GRASP_DZ = float(os.environ.get("GRASP_DZ", "-0.01"))
LIFT_Z = float(os.environ.get("LIFT_Z", "1.15"))
SINK_DROP_Z = float(os.environ.get("SINK_DROP_Z", "0.98"))
SERVO = float(os.environ.get("SERVO", "0.04"))
GRIP_CLOSE, GRIP_OPEN = 1.0, -1.0
DEBUG = os.environ.get("DEBUG", "0") == "1"

# ---------- geometry (reachable counter region; tune via render) ----------
REACH_X = tuple(float(v) for v in os.environ.get("REACH_X", "1.74,1.88").split(","))
REACH_Y = tuple(float(v) for v in os.environ.get("REACH_Y", "-0.42,-0.14").split(","))
SPACE = float(os.environ.get("SPACE", "0.075"))       # min inter-object spacing
# which WORLD axis maps to image left-right (calibrate via render); the OTHER axis is depth (front/behind)
LATERAL_AXIS = os.environ.get("LATERAL_AXIS", "y")
LATERAL_SIGN = float(os.environ.get("LATERAL_SIGN", "1"))   # +1 if "left" = +offset on the lateral axis
LAT_OFF = float(os.environ.get("LAT_OFF", "0.09"))    # lateral offset of each candidate from ref (left/right)
DEP_JIT = float(os.environ.get("DEP_JIT", "0.02"))    # small depth jitter so it is not a perfect line
NEAR_R = float(os.environ.get("NEAR_R", "0.07"))      # target within this of ref for "next_to"
FAR_R = float(os.environ.get("FAR_R", "0.15"))        # distractor at least this from ref for "next_to"

# ---------- optional recolour candidates to ONE colour (make the 2 candidates visually identical) ----------
PAINT = os.environ.get("PAINT", "1") == "1"
CAND_RGBA = np.asarray([float(v) for v in os.environ.get("CAND_RGBA", "0.85,0.10,0.10,1.0").split(",")], np.float32)


def readable(cat):
    return cat.replace("_", " ")


class RelationPnP(PickPlaceCounterToSink):
    """obj = target (success slot), distr_0 = same-category distractor, ref = different-category landmark."""
    def __init__(self, cand_cat, ref_cat, *a, **k):
        self._cand_cat = cand_cat; self._ref_cat = ref_cat
        super().__init__(obj_groups=cand_cat, *a, **k)

    def _get_obj_cfgs(self):
        reg = dict(fixture=self.counter, sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                   size=(0.55, 0.40), pos=("ref", -1.0))
        return [
            dict(name="obj", obj_groups=self._cand_cat, graspable=True, placement=dict(reg)),
            dict(name="distr_0", obj_groups=self._cand_cat, graspable=True, placement=dict(reg)),
            dict(name="ref", obj_groups=self._ref_cat, graspable=True, placement=dict(reg)),
        ]


def make_env(seed):
    from robosuite.controllers import load_composite_controller_config
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = RelationPnP(
        cand_cat=CAND_CAT, ref_cat=REF_CAT,
        robots="PandaOmron", controller_configs=cc,
        camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=CAMW, camera_heights=CAMW,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
        ignore_done=True, seed=seed, layout_ids=LAYOUT, style_ids=STYLE,
        translucent_robot=False, obj_instance_split=None, generative_textures=None,
    )
    return env


def objpos(obs, name):
    return np.asarray(obs[name + "_pos"], np.float32)


# ---------- recolour (force flat vivid colour; borrowed from collect_rc_colour.py) ----------
def _obj_geom_ids(env, name):
    obj = env.objects[name]; ids = set(); names = []
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


def paint(env, name, rgba):
    for gid in _obj_geom_ids(env, name):
        env.sim.model.geom_matid[gid] = -1
        env.sim.model.geom_rgba[gid] = rgba
    env.sim.forward()


# ---------- teleport primitive (specified position; z from the object's current rest) ----------
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


def teleport(env, name, x, y):
    a0 = _obj_joint_addr(env, name)
    if a0 is None:
        return False
    z = float(env.sim.data.qpos[a0 + 2])
    env.sim.data.qpos[a0:a0 + 3] = [x, y, z + 0.02]
    return True


# ---------- RELATION placement: reference + target (relation-satisfying) + distractor (not) ----------
def _mk(lat, dep):
    return np.array([dep, lat], np.float32) if LATERAL_AXIS == "y" else np.array([lat, dep], np.float32)


def _inbounds(p, m=0.0):
    return (REACH_X[0] - m <= p[0] <= REACH_X[1] + m) and (REACH_Y[0] - m <= p[1] <= REACH_Y[1] + m)


def _spaced(pts, s):
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i][:2] - pts[j][:2]) < s:
                return False
    return True


def sample_relation_layout(rng, relation):
    """Return (positions dict {ref,obj,distr_0}, meta) or (None,None) if no valid layout found.
    obj is placed on the relation-SATISFYING side; distr_0 on the NON-satisfying side."""
    LAT = REACH_Y if LATERAL_AXIS == "y" else REACH_X
    DEP = REACH_X if LATERAL_AXIS == "y" else REACH_Y
    for _ in range(400):
        if relation in ("left", "right"):
            r_lat = rng.uniform(LAT[0] + LAT_OFF, LAT[1] - LAT_OFF)
            r_dep = rng.uniform(DEP[0] + DEP_JIT, DEP[1] - DEP_JIT)
            sgn = LATERAL_SIGN if relation == "left" else -LATERAL_SIGN
            ref = _mk(r_lat, r_dep)
            obj = _mk(r_lat + sgn * LAT_OFF, r_dep + rng.uniform(-DEP_JIT, DEP_JIT))
            dis = _mk(r_lat - sgn * LAT_OFF, r_dep + rng.uniform(-DEP_JIT, DEP_JIT))
        elif relation in ("front", "behind"):
            # depth relation on the NON-lateral axis
            d_lo, d_hi = DEP
            r_dep = rng.uniform(d_lo + LAT_OFF, d_hi - LAT_OFF)
            r_lat = rng.uniform(LAT[0] + DEP_JIT, LAT[1] - DEP_JIT)
            sgn = LATERAL_SIGN if relation == "front" else -LATERAL_SIGN
            ref = _mk(r_lat, r_dep)
            obj = _mk(r_lat + rng.uniform(-DEP_JIT, DEP_JIT), r_dep + sgn * LAT_OFF)
            dis = _mk(r_lat + rng.uniform(-DEP_JIT, DEP_JIT), r_dep - sgn * LAT_OFF)
        else:  # PROXIMITY relations {next_to, far_from}: reference ROAMS the full 2D area (=> target position
               # varies fully with it, breaking the absolute-position shortcut). one candidate NEAR ref, one FAR.
            r_lat = rng.uniform(LAT[0], LAT[1]); r_dep = rng.uniform(DEP[0], DEP[1])
            ref = _mk(r_lat, r_dep)
            a1 = rng.uniform(0, 2 * math.pi); near = rng.uniform(NEAR_R * 0.85, NEAR_R)  # >=0.85*NEAR_R clears lemon overlap
            a2 = rng.uniform(0, 2 * math.pi); far = rng.uniform(FAR_R, FAR_R + 0.05)
            near_p = _mk(r_lat + near * math.cos(a1), r_dep + near * math.sin(a1) * 0.7)
            far_p = _mk(r_lat + far * math.cos(a2), r_dep + far * math.sin(a2) * 0.7)
            obj, dis = (near_p, far_p) if relation == "next_to" else (far_p, near_p)  # target=near for next_to, far for far_from
        pts = [ref, obj, dis]
        if not all(_inbounds(p) for p in pts):
            continue
        if not _spaced(pts, SPACE):    # SPACE must be < NEAR_R*0.85 so the "next to" candidate is not rejected
            continue
        d_obj = float(np.linalg.norm(obj[:2] - ref[:2])); d_dis = float(np.linalg.norm(dis[:2] - ref[:2]))
        if relation == "next_to" and not (d_obj + 0.035 < d_dis):
            continue   # target must be CLEARLY the nearer one
        if relation == "far_from" and not (d_dis + 0.035 < d_obj):
            continue   # target must be CLEARLY the farther one
        meta = dict(relation=relation, ref=ref.tolist(), obj=obj.tolist(), distr_0=dis.tolist(),
                    d_obj_ref=d_obj, d_dis_ref=d_dis)
        return {"ref": ref, "obj": obj, "distr_0": dis}, meta
    return None, None


def apply_layout(env, positions):
    for name, p in positions.items():
        teleport(env, name, float(p[0]), float(p[1]))
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


# ---------- oracle (BYTE-IDENTICAL to B45 collect_rc_deconf.py; motor NOT re-tuned) ----------
def rollout(env, obs, target="obj", record=True):
    robot = env.robots[0]
    tp0 = objpos(obs, target)
    sxy = sink_xy(env)
    ph, grip, hold, rel = "hover", GRIP_OPEN, 0, 0
    im, wr, st, ac = [], [], [], []
    frames = []
    dmin, dmin_xy, obj_zmax = 1e9, 1e9, tp0[2]
    prev_z, desc_stall, last_ph = None, 0, "hover"
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
            # descent stalled (gripper bottomed out) while xy-aligned => grasp HERE (robust to the eef bottoming
            # ~0.02 above obj_z+GRASP_DZ, which was the B45 descend-stuck 40% failure mode). §3.6-diagnosed.
            desc_stall = desc_stall + 1 if (prev_z is not None and (prev_z - eef[2]) < 0.001) else 0
            aligned = np.linalg.norm(eef[:2] - tp[:2]) < 0.03
            if eef[2] < gz + 0.02 or (desc_stall >= 10 and aligned):
                ph = "grasp"; hold = 0
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
        prev_z = float(eef[2]); last_ph = ph
        obs, r, done, info = step_arm(env, robot, dpos, grip)
        if ph == "retract" and rel >= 22 and bool(env._check_success()):
            break
        if rel >= 30:
            break
    succ = bool(env._check_success())
    if DEBUG:
        print("  END succ=%s dmin=%.3f dmin_xy=%.3f obj_lift=%.3f last_ph=%s" % (
            succ, dmin, dmin_xy, obj_zmax - tp0[2], last_ph), flush=True)
    return succ, (im, wr, st, ac), frames


def main():
    rng = np.random.RandomState(SEED)
    env = make_env(SEED)
    n_ok = 0
    for relation in RELATIONS:
        got, tries = 0, 0
        while got < N_PER_OBJ and tries < N_PER_OBJ * 8:
            tries += 1
            env.reset()
            positions, meta = sample_relation_layout(rng, relation)
            if positions is None:
                if DEBUG:
                    print("  PLACE_FAIL relation=%s" % relation, flush=True)
                continue
            obs = apply_layout(env, positions)
            if PAINT:
                paint(env, "obj", CAND_RGBA); paint(env, "distr_0", CAND_RGBA)
                env.sim.forward()
                obs, _, _, _ = env.step(np.zeros(env.action_dim))
            # capture ACTUAL settled positions for the ⊥ red-team (not just the requested ones)
            pos_actual = {k: objpos(obs, k)[:2].tolist() for k in ("obj", "distr_0", "ref")}
            if DEBUG:
                print("  relation=%s req=%s actual=%s" % (
                    relation, {k: np.round(v, 3).tolist() for k, v in meta.items() if k in ("ref", "obj", "distr_0")},
                    {k: np.round(v, 3).tolist() for k, v in pos_actual.items()}), flush=True)
            try:
                succ, data, frames = rollout(env, obs, "obj", record=True)
            except Exception as e:
                print("ROLLOUT_ERR", relation, e, flush=True); traceback.print_exc(); continue
            if RENDER and frames:
                strip = np.concatenate([f for f in frames[:16]], axis=1)
                imageio.imwrite("%s/roll_%s_%d_%s.png" % (VIZ, relation, tries, "S" if succ else "F"), strip)
            if not succ:
                continue
            im, wr, st, ac = data
            lang = "pick the %s %s the %s and place it in the sink" % (
                readable(CAND_CAT), PHRASE[relation], readable(REF_CAT))
            np.savez_compressed(RAW + "/ep_%d_%s_%04d.npz" % (SHARD, relation, got),
                                img=np.asarray(im, np.uint8), wrist=np.asarray(wr, np.uint8),
                                state=np.asarray(st, np.float32), action=np.asarray(ac, np.float32),
                                lang=lang, relation=relation, cand_cat=CAND_CAT, ref_cat=REF_CAT,
                                pos_actual=json.dumps(pos_actual), meta=json.dumps(meta))
            got += 1; n_ok += 1
        print("RELATION %s: %d/%d ok (%d tries)" % (relation, got, N_PER_OBJ, tries), flush=True)
    env.close()
    print("SHARD%d_DONE total=%d" % (SHARD, n_ok), flush=True)


if __name__ == "__main__":
    main()
