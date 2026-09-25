"""collect_category.py — RoboCasa CATEGORY-referent DECONFOUNDED oracle collector (Galahad G1 universal-deconf).

Derived from collect_rc_deconf.py (B4). The oracle MOTOR / servo / obs-contract are UNCHANGED (do NOT re-tune grasps —
CLAUDE §8 / LAB B45). Only scene composition + instruction change, for the CATEGORY referent type:
  target named by CATEGORY ("pick the {cat}"); role-balanced across a pool of >=4 visually-distinct graspable cats.
  Three deconfounds make the instruction the ONLY predictor of the target:
    category _|_ position     : randomize_objs teleports every object to a random reachable (x,y) each reset.
    category _|_ instance     : each reset re-samples a DIFFERENT instance mesh per category (RoboCasa hard_reset);
                                the sampled mesh id is RECORDED per npz (`target_mdl`) so the deconf is VERIFIABLE.
    category _|_ co-appearance: _get_obj_cfgs samples SCENE_K-1 random distractor categories from the pool each reset.
  Distractors ALWAYS in-frame (SCENE_K>=2). One env per target category; reset re-randomizes everything.
  Records img/wrist/state(8)/action(7)/lang(+target_cat/target_mdl/distractors) => npz_to_lerobot.py unchanged (B4 schema).

Run one shard/GPU:
  RENDER=0 POOL=banana,carrot,can,apple,lemon,corn SCENE_K=4 N_PER_OBJ=17 TARGETS=banana,carrot \\
  RAW=/dev/shm/cat_raw SHARD=0 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 $RCPY collect_category.py
"""
import os, sys, json, traceback
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
import robocasa.utils.object_utils as OU
from robosuite.utils.transform_utils import quat2axisangle, quat2mat

POOL = os.environ.get("POOL", "banana,carrot,apple,broccoli,eggplant").split(",")  # can DROPPED: 0/11 cylinder motor-trap (B43-45)
TARGETS = os.environ.get("TARGETS", ",".join(POOL)).split(",")   # which targets THIS shard collects
SCENE_K = int(os.environ.get("SCENE_K", "4"))                    # #categories in each scene (target + SCENE_K-1 distractors)
N_PER_OBJ = int(os.environ.get("N_PER_OBJ", "17"))
RAW = os.environ.get("RAW", "/dev/shm/cat_raw")
SHARD = int(os.environ.get("SHARD", "0"))
SEED = int(os.environ.get("SEED", str(2000 + SHARD)))
LAYOUT = int(os.environ.get("LAYOUT", "1"))    # 1-indexed (1-10 = test layouts); 0 is invalid
STYLE = int(os.environ.get("STYLE", "1"))
RENDER = os.environ.get("RENDER", "0") == "1"
CAMW = int(os.environ.get("CAMW", "256"))
MAXS = int(os.environ.get("MAXS", "520"))
os.makedirs(RAW, exist_ok=True)
VIZ = "/dev/shm/cat_viz"; os.makedirs(VIZ, exist_ok=True)

# ---- obs-key contract (identical to B4 collect_rc_deconf.py; VERIFY via probe_rc.py before trusting) ----
CAM_AGENT = os.environ.get("CAM_AGENT", "robot0_agentview_right")  # right frames the counter objects far better
CAM_WRIST = os.environ.get("CAM_WRIST", "robot0_eye_in_hand")
# oracle geometry (from B4 probe; DO NOT re-tune — motor reuse)
HOVER = float(os.environ.get("HOVER", "0.12"))
GRASP_DZ = float(os.environ.get("GRASP_DZ", "-0.01"))
LIFT_Z = float(os.environ.get("LIFT_Z", "1.15"))
SINK_DROP_Z = float(os.environ.get("SINK_DROP_Z", "0.98"))
SERVO = float(os.environ.get("SERVO", "0.04"))
GRIP_CLOSE, GRIP_OPEN = 1.0, -1.0


class DeconfPnP(PickPlaceCounterToSink):
    def __init__(self, pool, target_cat, scene_k, *a, **k):
        self._pool = list(pool); self._target_cat = target_cat; self._scene_k = scene_k
        self._last_distr = []
        super().__init__(obj_groups=target_cat, *a, **k)

    def _get_obj_cfgs(self):
        # target ALWAYS present; sample SCENE_K-1 random distractor categories (category _|_ co-appearance).
        others = [c for c in self._pool if c != self._target_cat]
        rng = getattr(self, "rng", np.random)
        n = max(1, min(self._scene_k - 1, len(others)))
        idx = (rng.permutation(len(others)) if hasattr(rng, "permutation") else np.random.permutation(len(others)))[:n]
        chosen = [others[int(i)] for i in np.atleast_1d(idx)]
        self._last_distr = chosen
        reg = dict(fixture=self.counter, sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                   size=(0.55, 0.40), pos=("ref", -1.0))
        cfgs = [dict(name="obj", obj_groups=self._target_cat, graspable=True, placement=dict(reg))]
        for i, c in enumerate(chosen):
            cfgs.append(dict(name="distr_%d" % i, obj_groups=c, graspable=True, placement=dict(reg)))
        return cfgs


def make_env(target_cat, seed):
    from robosuite.controllers import load_composite_controller_config
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = DeconfPnP(
        pool=POOL, target_cat=target_cat, scene_k=SCENE_K,
        robots="PandaOmron", controller_configs=cc,
        camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=CAMW, camera_heights=CAMW,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
        ignore_done=True, seed=seed, layout_ids=LAYOUT, style_ids=STYLE,
        translucent_robot=False, obj_instance_split=None, generative_textures=None,
    )
    return env


def get_mdl(env, name):
    """Best-effort unique instance-mesh id for `name` — so category _|_ instance is VERIFIABLE post-hoc."""
    try:
        for c in (env.get_ep_meta().get("object_cfgs") or []):
            if c.get("name") == name:
                info = c.get("info") or {}
                v = info.get("mjcf_path") or info.get("model") or c.get("model") or info.get("cat")
                if v:
                    return str(v)
    except Exception:
        pass
    try:
        o = env.objects[name]
        for a in ("mjcf_path", "_mjcf_path", "mjcffile", "_model_path", "model_path", "name"):
            v = getattr(o, a, None)
            if v:
                return str(v)
    except Exception:
        pass
    return "?"


def objpos(obs, name):
    return np.asarray(obs[name + "_pos"], np.float32)


# reachable counter region (world x,y) — objects teleported here for controlled deconf placement (from B4 launcher)
REACH_X = tuple(float(v) for v in os.environ.get("REACH_X", "1.74,1.88").split(","))
REACH_Y = tuple(float(v) for v in os.environ.get("REACH_Y", "-0.40,-0.16").split(","))
SPACE = float(os.environ.get("SPACE", "0.085"))   # min inter-object spacing


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
    """Teleport all pool objects to random (x,y) in the reachable region (category _|_ position)."""
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
    arm = np.concatenate([np.asarray(dpos, np.float32), np.zeros(3, np.float32)])  # 6d OSC (rot held)
    ad = {"right": arm, "right_gripper": np.array([grip], np.float32)}
    full = robot.create_action_vector(ad)
    return env.step(full)


def sink_xy(env):
    try:
        p = env.sink.pos
        return np.array([p[0], p[1]], np.float32)
    except Exception:
        return None


DEBUG = os.environ.get("DEBUG", "0") == "1"


def rollout(env, obs, target="obj", record=True):
    robot = env.robots[0]
    tp0 = objpos(obs, target); z_top = tp0[2]
    sxy = sink_xy(env)
    ph, grip, hold, rel = "hover", GRIP_OPEN, 0, 0
    im, wr, st, ac = [], [], [], []
    frames = []
    dmin, dmin_xy, obj_zmax, last_ph = 1e9, 1e9, tp0[2], None
    eef0 = np.asarray(obs["robot0_eef_pos"], np.float32)
    base_R = quat2mat(np.asarray(obs["robot0_base_quat"], np.float64))  # base fixed during manip
    if DEBUG:
        print("  START target=%s obj=%s eef=%s sink=%s" % (
            target, np.round(tp0, 3), np.round(eef0, 3), np.round(sxy, 3) if sxy is not None else None), flush=True)
    for s in range(MAXS):
        eef = np.asarray(obs["robot0_eef_pos"], np.float32)
        tp = objpos(obs, target)
        d = float(np.linalg.norm(eef - tp)); dmin = min(dmin, d)
        dxy = float(np.linalg.norm(eef[:2] - tp[:2])); dmin_xy = min(dmin_xy, dxy)
        obj_zmax = max(obj_zmax, float(tp[2]))
        if DEBUG and ph != last_ph:
            print("    ph=%-8s s=%3d eef=%s obj=%s dxy=%.3f dz=%.3f" % (
                ph, s, np.round(eef, 3), np.round(tp, 3), dxy, float(eef[2] - tp[2])), flush=True)
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
        else:  # retract UP+away so gripper_obj_far (>0.25m) => success can register
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
    if DEBUG:
        print("  END succ=%s dmin=%.3f dmin_xy=%.3f obj_lift=%.3f last_ph=%s" % (
            succ, dmin, dmin_xy, obj_zmax - tp0[2], ph), flush=True)
    return succ, (im, wr, st, ac), frames


def main():
    rng = np.random.RandomState(SEED)
    n_ok = 0
    mdl_seen = {}   # target_cat -> set(mesh ids) : instance-randomization audit
    for target_cat in TARGETS:
        env = make_env(target_cat, SEED)
        got, tries = 0, 0
        mdl_seen.setdefault(target_cat, set())
        while got < N_PER_OBJ and tries < N_PER_OBJ * 6:
            tries += 1
            env.reset()
            obs = randomize_objs(env, rng)
            target_mdl = get_mdl(env, "obj")
            distractors = list(env._last_distr)
            mdl_seen[target_cat].add(target_mdl)
            if DEBUG:
                print("  PLACED tgt=%s mdl=%s distr=%s pos=%s" % (
                    target_cat, target_mdl, distractors,
                    " ".join("%s=%s" % (k, np.round(objpos(obs, k), 3)) for k in env.objects)), flush=True)
            try:
                succ, data, frames = rollout(env, obs, "obj", record=True)
            except Exception as e:
                print("ROLLOUT_ERR", target_cat, e, flush=True); traceback.print_exc(); continue
            if RENDER and frames:
                strip = np.concatenate([f for f in frames[:16]], axis=1)  # horizontal filmstrip PNG (viewable)
                imageio.imwrite("%s/roll_%s_%d_%s.png" % (VIZ, target_cat, tries, "S" if succ else "F"), strip)
            if not succ:
                continue
            im, wr, st, ac = data
            lang = env.get_ep_meta().get("lang")
            np.savez_compressed(RAW + "/ep_%d_%s_%04d.npz" % (SHARD, target_cat, got),
                                img=np.asarray(im, np.uint8), wrist=np.asarray(wr, np.uint8),
                                state=np.asarray(st, np.float32), action=np.asarray(ac, np.float32),
                                lang=lang, target_cat=target_cat, target_mdl=target_mdl,
                                distractors=json.dumps(distractors))
            got += 1; n_ok += 1
        print("TARGET %s: %d/%d ok (%d tries) distinct_meshes=%d" % (
            target_cat, got, N_PER_OBJ, tries, len(mdl_seen[target_cat])), flush=True)
        env.close()
    print("SHARD%d_DONE total=%d mesh_audit=%s" % (
        SHARD, n_ok, {c: len(s) for c, s in mdl_seen.items()}), flush=True)


if __name__ == "__main__":
    main()
