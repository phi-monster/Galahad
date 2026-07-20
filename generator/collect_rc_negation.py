"""collect_rc_negation.py (/root/rc_venv) — RoboCasa PnP NEGATION-deconfounded oracle collector (Galahad G1 universal-deconf).

DESIGN (referent-type = NEGATION). Each scene has EXACTLY 2 objects {A, B}, A!=B, drawn from the B45 pool.
On the SAME scene (identical object positions, captured once and restored), two paired demos are produced:
  - NEGATION demo:  instruction "pick the object that is not the {A} ..."  -> oracle grasps B (the non-A object).
  - POSITIVE demo:  instruction "pick the {A} ..."                          -> oracle grasps A (the named object).
=> On one scene, positive and negation go to DIFFERENT objects. A model that IGNORES the word "not" (entity-matches
   the named noun {A}) grasps A in BOTH -> correct on positive, WRONG on negation -> at/below chance (2-AFC chance=50%).
   Position is randomized (uninformative); size/nearest heuristics -> chance. ONLY parsing "not {A}" passes both.
Role-balance: iterate ALL 20 ordered (A,B) category pairs equally => every category is the named/negated object and
   the negation-target equally often; positions randomized each reset => the INSTRUCTION is the ONLY predictor.

MOTOR = B45 collect_rc_deconf.py oracle, REUSED VERBATIM (hover->descend->grasp->lift->carry->lower->release->retract,
   OSC servo in base frame). rollout() takes a `target` obs-name so it can grasp either "obj"(=A) or "distr_0"(=B).
   Success = geometric in-sink check on the ACTUAL grasped object (env._check_success only tracks "obj"=A).

Run one shard/GPU:
  RENDER=0 RAW=/dev/shm/rc_neg_raw PAIRS=apple:banana,apple:lemon N_PER_PAIR=13 SHARD=0 \
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 /root/rc_venv/bin/python collect_rc_negation.py
"""
import os, sys, json, itertools, traceback
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
import robocasa.utils.object_utils as OU
from robosuite.utils.transform_utils import quat2axisangle, quat2mat

# can DROPPED: tall cylinder = motor trap (0/11) AND a §2 smoke-bomb (rolls/knocks into sink -> succ=True at obj_lift~0).
POOL = os.environ.get("POOL", "apple,banana,lemon,carrot").split(",")
# PAIRS = comma list of "A:B" ordered pairs THIS shard collects; default = all 20 ordered pairs over POOL.
_default_pairs = ["%s:%s" % (a, b) for a in POOL for b in POOL if a != b]
PAIRS = os.environ.get("PAIRS", ",".join(_default_pairs)).split(",")
N_PER_PAIR = int(os.environ.get("N_PER_PAIR", "13"))   # scenes per ordered pair; each scene -> 1 neg + 1 pos demo
RAW = os.environ.get("RAW", "/dev/shm/rc_neg_raw")
SHARD = int(os.environ.get("SHARD", "0"))
SEED = int(os.environ.get("SEED", str(2000 + SHARD)))
LAYOUT = int(os.environ.get("LAYOUT", "1"))
STYLE = int(os.environ.get("STYLE", "1"))
RENDER = os.environ.get("RENDER", "0") == "1"
CAMW = int(os.environ.get("CAMW", "256"))
MAXS = int(os.environ.get("MAXS", "520"))
os.makedirs(RAW, exist_ok=True)
VIZ = os.environ.get("VIZ", "/dev/shm/rc_neg_viz"); os.makedirs(VIZ, exist_ok=True)

# instruction templates — matched to RoboCasa native lang ("Pick the apple from the counter and place it in the sink.")
# so positive demos are in-distribution; negation differs MINIMALLY (only "object that is not the" vs "{a}").
POS_TMPL = os.environ.get("POS_TMPL", "Pick the {a} from the counter and place it in the sink.")
NEG_TMPL = os.environ.get("NEG_TMPL", "Pick the object that is not the {a} from the counter and place it in the sink.")

# ---- obs-key contract (VERIFY against probe_rc.py before trusting) ----
CAM_AGENT = os.environ.get("CAM_AGENT", "robot0_agentview_right")
CAM_WRIST = os.environ.get("CAM_WRIST", "robot0_eye_in_hand")
HOVER = float(os.environ.get("HOVER", "0.12"))
GRASP_DZ = float(os.environ.get("GRASP_DZ", "-0.01"))
LIFT_Z = float(os.environ.get("LIFT_Z", "1.15"))
SINK_DROP_Z = float(os.environ.get("SINK_DROP_Z", "0.98"))
SERVO = float(os.environ.get("SERVO", "0.04"))
GRIP_CLOSE, GRIP_OPEN = 1.0, -1.0

# geometric in-sink success gate (calibrated to B45 geometry: basin z~0.816, counter obj z~0.975; tune via probe/render)
SINK_R = float(os.environ.get("SINK_R", "0.15"))       # xy radius around sink center
SINK_Z_MAX = float(os.environ.get("SINK_Z_MAX", "0.92"))  # obj must have dropped below counter top into the basin
MIN_LIFT = float(os.environ.get("MIN_LIFT", "0.08"))   # §2: reject degenerate "knocked/rolled into sink" successes (obj never lifted)

REACH_X = tuple(float(v) for v in os.environ.get("REACH_X", "1.05,1.70").split(","))
REACH_Y = tuple(float(v) for v in os.environ.get("REACH_Y", "-0.55,-0.28").split(","))
SPACE = float(os.environ.get("SPACE", "0.10"))
DEBUG = os.environ.get("DEBUG", "0") == "1"


class DeconfPnP(PickPlaceCounterToSink):
    """2-object scene: 'obj'=target_cat (=A, the NAMED object) + 'distr_0'=other (=B, the NEGATION target)."""
    def __init__(self, pair, *a, **k):
        self._pair = list(pair)          # [A, B]
        self._target_cat = pair[0]       # A
        super().__init__(obj_groups=pair[0], *a, **k)

    def _get_obj_cfgs(self):
        A, B = self._pair
        reg = dict(fixture=self.counter, sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                   size=(0.55, 0.40), pos=("ref", -1.0))
        return [dict(name="obj", obj_groups=A, graspable=True, placement=dict(reg)),
                dict(name="distr_0", obj_groups=B, graspable=True, placement=dict(reg))]


def make_env(pair, seed):
    from robosuite.controllers import load_composite_controller_config
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return DeconfPnP(
        pair=pair, robots="PandaOmron", controller_configs=cc,
        camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=CAMW, camera_heights=CAMW,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
        ignore_done=True, seed=seed, layout_ids=LAYOUT, style_ids=STYLE,
        translucent_robot=False, obj_instance_split=None, generative_textures=None,
    )


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


def _settle(env, n=12):
    obs = None
    for _ in range(n):
        obs, _, _, _ = env.step(np.zeros(env.action_dim))
    return obs


def randomize_objs(env, rng):
    """Teleport all pool objects to random reachable (x,y); return obs + saved free-joint qpos (7dof each) for restore."""
    placed, saved = [], {}
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
    obs = _settle(env)
    # snapshot settled object poses for deterministic same-scene restore
    for name in list(env.objects.keys()):
        a0 = _obj_joint_addr(env, name)
        if a0 is not None:
            saved[name] = np.array(env.sim.data.qpos[a0:a0 + 7], np.float64)
    return obs, saved


def restore_objs(env, saved):
    """Teleport objects back to the SAVED settled poses (same-scene paired control) after a reset."""
    for name, q in saved.items():
        a0 = _obj_joint_addr(env, name)
        if a0 is not None:
            env.sim.data.qpos[a0:a0 + 7] = q
    env.sim.forward()
    return _settle(env)


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
    return env.step(robot.create_action_vector(ad))


def sink_xy(env):
    try:
        p = env.sink.pos
        return np.array([p[0], p[1]], np.float32)
    except Exception:
        return None


def in_sink(env, obs, target):
    """Geometric success on the ACTUAL grasped object (decoupled from env._check_success which tracks 'obj'=A)."""
    sxy = sink_xy(env)
    if sxy is None:
        return False
    p = objpos(obs, target)
    return bool(np.linalg.norm(p[:2] - sxy) < SINK_R and p[2] < SINK_Z_MAX)


def rollout(env, obs, target, record=True, tag=""):
    """B45 oracle VERBATIM; `target` = obs-name of the object to grasp/place ('obj'=A pos, 'distr_0'=B neg)."""
    robot = env.robots[0]
    tp0 = objpos(obs, target)
    sxy = sink_xy(env)
    ph, grip, hold, rel = "hover", GRIP_OPEN, 0, 0
    im, wr, st, ac = [], [], [], []
    frames = []
    dmin, dmin_xy, obj_zmax, last_ph = 1e9, 1e9, tp0[2], None
    base_R = quat2mat(np.asarray(obs["robot0_base_quat"], np.float64))
    _seen_ph = set()
    for s in range(MAXS):
        eef = np.asarray(obs["robot0_eef_pos"], np.float32)
        tp = objpos(obs, target)
        d = float(np.linalg.norm(eef - tp)); dmin = min(dmin, d)
        dxy = float(np.linalg.norm(eef[:2] - tp[:2])); dmin_xy = min(dmin_xy, dxy)
        obj_zmax = max(obj_zmax, float(tp[2]))
        if DEBUG and ph != last_ph:
            print("    [%s] ph=%-8s s=%3d eef=%s obj=%s dxy=%.3f dz=%.3f" % (
                tag, ph, s, np.round(eef, 3), np.round(tp, 3), dxy, float(eef[2] - tp[2])), flush=True)
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
        if RENDER and ph not in _seen_ph:  # one FULL-RES frame per phase transition (grasp moment etc. -> eyeball which obj)
            _seen_ph.add(ph)
            try:
                imageio.imwrite("%s/hi_%s_%s.png" % (VIZ, tag.replace(" ", "_").replace("!", "not"), ph),
                                np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1]))
            except Exception:
                pass
        obs, r, done, info = step_arm(env, robot, dpos, grip)
        if ph == "retract" and rel >= 22 and in_sink(env, obs, target):
            break
        if rel >= 30:
            break
    succ = in_sink(env, obs, target) and (obj_zmax - tp0[2] > MIN_LIFT)   # §2: must be genuinely lifted, not knocked in
    if DEBUG:
        print("  [%s] END succ=%s dmin=%.3f dmin_xy=%.3f obj_lift=%.3f last_ph=%s" % (
            tag, succ, dmin, dmin_xy, obj_zmax - tp0[2], ph), flush=True)
    return succ, (im, wr, st, ac), frames


def save_demo(kind, pair, scene_uid, data, lang, named, target_cat, paired):
    im, wr, st, ac = data
    np.savez_compressed(
        RAW + "/ep_%d_%s_%s_%s.npz" % (SHARD, kind, "%s-%s" % (pair[0], pair[1]), scene_uid),
        img=np.asarray(im, np.uint8), wrist=np.asarray(wr, np.uint8),
        state=np.asarray(st, np.float32), action=np.asarray(ac, np.float32),
        lang=lang, kind=kind, named=named, target_cat=target_cat, paired=bool(paired),
        pair="%s-%s" % (pair[0], pair[1]), shard=SHARD, scene_uid=scene_uid)


def main():
    rng = np.random.RandomState(SEED)
    n_neg = n_pos = n_paired = 0
    for pstr in PAIRS:
        A, B = pstr.split(":")
        env = make_env([A, B], SEED)
        got, tries = 0, 0
        while got < N_PER_PAIR and tries < N_PER_PAIR * 10:
            tries += 1
            scene_uid = "%04d" % tries   # per-(shard,pair); neg+pos of THIS try share it => reconstruct matched pairs
            env.reset()
            obs, saved = randomize_objs(env, rng)
            # --- NEGATION demo: grasp B (distr_0); instruction names A with "not" ---
            try:
                s_neg, d_neg, f_neg = rollout(env, obs, "distr_0", record=True, tag="NEG %s!=%s" % (B, A))
            except Exception as e:
                print("NEG_ERR", pstr, e, flush=True); traceback.print_exc(); continue
            # --- restore SAME scene, POSITIVE demo: grasp A (obj); instruction names A ---
            env.reset()
            obs2 = restore_objs(env, saved)
            try:
                s_pos, d_pos, f_pos = rollout(env, obs2, "obj", record=True, tag="POS %s" % A)
            except Exception as e:
                print("POS_ERR", pstr, e, flush=True); traceback.print_exc(); continue
            if DEBUG:  # CALIBRATION: on the positive demo obj=A IS env's designated target, so env._check_success is valid.
                try:
                    ec = bool(env._check_success())
                except Exception as e:
                    ec = "ERR:%s" % e
                print("  CALIB pair=%s in_sink(pos)=%s env_check=%s sink=%s obj_pos=%s distr_pos=%s" % (
                    pstr, s_pos, ec, np.round(sink_xy(env), 3),
                    np.round(objpos(_settle(env, 1), "obj"), 3), np.round(objpos(_settle(env, 1), "distr_0"), 3)),
                    flush=True)
            if RENDER:
                for f, tg in [(f_neg, "NEG"), (f_pos, "POS")]:
                    if f:
                        strip = np.concatenate([x for x in f[:16]], axis=1)
                        imageio.imwrite("%s/roll_%s_%s-%s_%d.png" % (VIZ, tg, A, B, tries), strip)
            # save each demo independently (max yield); flag same-scene both-succeeded as a matched control pair.
            paired = bool(s_neg and s_pos)
            if s_neg:
                save_demo("neg", [A, B], scene_uid, d_neg, NEG_TMPL.format(a=A), named=A, target_cat=B, paired=paired)
                n_neg += 1; got += 1   # gate the per-pair budget on NEGATION yield (the star output)
            if s_pos:
                save_demo("pos", [A, B], scene_uid, d_pos, POS_TMPL.format(a=A), named=A, target_cat=A, paired=paired)
                n_pos += 1
            if paired:
                n_paired += 1
        print("PAIR %s: neg_got=%d/%d (%d tries)" % (pstr, got, N_PER_PAIR, tries), flush=True)
        env.close()
    print("SHARD%d_DONE neg=%d pos=%d paired=%d" % (SHARD, n_neg, n_pos, n_paired), flush=True)


if __name__ == "__main__":
    main()
