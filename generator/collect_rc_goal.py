"""collect_rc_goal.py (/root/rc_venv) — RoboCasa GOAL/DESTINATION-referent DECONFOUNDED oracle collector (Galahad G1, type #7).

FORK of collect_ordinal.py (B52) / collect_rc_deconf.py (B45). MOTOR REUSED VERBATIM (servo / step_arm / phase machine /
base_R). The ONLY change vs B45: same object + same verb (pick-place), but the DESTINATION varies — several RECEPTACLES
are present in-scene at randomized positions, and the instruction's DESTINATION phrase ("... in the bowl / on the plate /
in the pan") is the ONLY predictor of which receptacle to place into. Confound to break: destination ⟂ scene-layout ⟂
object-identity ⇒ the model must READ the destination word, not go to a fixed slot nor the nearest/only container.

DECONF construction (per episode):
  - REC_POOL receptacles (default bowl,plate,pan) ALL present, teleported to random reachable (x,y) each reset (layout ⟂ dest).
  - ONE graspable target object, category ROTATES across OBJ_POOL (dest ⟂ target-identity), placed at a random reachable (x,y).
  - target destination r role-balanced across the receptacles (each is the goal equally often).
  - instruction = "put the {obj} {prep} the {receptacle}"  (the ONLY predictor of WHERE).
  => destination ⟂ layout (receptacle positions randomized), ⟂ identity (target rotates), role-balanced.

Success (INIT-FALSE, lift+delivery gated — B57 trap): target is delivered to the NAMED receptacle
  (nearest receptacle to the settled target == named AND target within DELIV_TOL of it) AND the target was LIFTED
  > MIN_LIFT during the episode AND gripper is far. A do-nothing policy => target never lifted, never near a receptacle => 0.

Records img/wrist/state(8)/action(7)/lang => LeRobot via npz_to_lerobot.py (B4 schema, unchanged).

Run one shard/GPU:
  RENDER=0 OBJ_POOL=apple,banana,lemon,carrot,bar_soap REC_POOL=bowl,plate,pan NDEMOS=64 RAW=/dev/shm/goal_raw SHARD=0 \\
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 /root/rc_venv/bin/python collect_rc_goal.py
"""
import os, sys, json, traceback, signal
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
from robosuite.utils.transform_utils import quat2axisangle, quat2mat

OBJ_POOL = os.environ.get("OBJ_POOL", "apple,banana,lemon,carrot,bar_soap").split(",")
REC_POOL = os.environ.get("REC_POOL", "bowl,cutting_board").split(",")  # 2 MAX-distinct receptacles (render-gated):
# bowl=white round-DEEP vs cutting_board=brown flat-RECTANGULAR. pan DROPPED (dark+slides to sink); plate DROPPED
# (round like bowl = similarity confound + a 3rd big receptacle makes RoboCasa's placement sampler ~2x slower).
# preposition per receptacle (neutralised where possible; plate is naturally "on"). Destination NOUN is the discriminator.
PREP = {"bowl": "in", "plate": "on", "pan": "in", "pot": "in", "basket": "in", "tray": "on", "cutting_board": "on"}
NDEMOS = int(os.environ.get("NDEMOS", "64"))
RAW = os.environ.get("RAW", "/dev/shm/goal_raw")
SHARD = int(os.environ.get("SHARD", "0"))
SEED = int(os.environ.get("SEED", str(9000 + SHARD)))
LAYOUT = int(os.environ.get("LAYOUT", "1"))
STYLE = int(os.environ.get("STYLE", "1"))
RENDER = os.environ.get("RENDER", "0") == "1"
CAMW = int(os.environ.get("CAMW", "256"))
MAXS = int(os.environ.get("MAXS", "540"))
DEBUG = os.environ.get("DEBUG", "0") == "1"
os.makedirs(RAW, exist_ok=True)
VIZ = os.environ.get("VIZ", "/dev/shm/goal_viz"); os.makedirs(VIZ, exist_ok=True)

CAM_AGENT = os.environ.get("CAM_AGENT", "robot0_agentview_right")
CAM_WRIST = os.environ.get("CAM_WRIST", "robot0_eye_in_hand")

# ---- MOTOR geometry (B45 — REUSED VERBATIM, do NOT re-tune) ----
HOVER = float(os.environ.get("HOVER", "0.12"))
GRASP_DZ = float(os.environ.get("GRASP_DZ", "-0.01"))
LIFT_Z = float(os.environ.get("LIFT_Z", "1.15"))
DROP_DZ = float(os.environ.get("DROP_DZ", "0.10"))    # lower to receptacle-top + this before release
SERVO = float(os.environ.get("SERVO", "0.04"))
GRIP_CLOSE, GRIP_OPEN = 1.0, -1.0
TALL_REST_Z = float(os.environ.get("TALL_REST_Z", "0.045"))  # B27 tall-object adaptive grasp threshold

# ---- placement region: FRONT COUNTER LIP that CLEARS the sink basin (B52: y>-0.48 drops items INTO the basin;
# -0.585 clears the front edge). All items (obj + 3 receptacles) sit in this front band, x-spread => layout ⟂ dest. ----
REACH_X = tuple(float(v) for v in os.environ.get("REACH_X", "0.88,1.66").split(","))
REACH_Y = tuple(float(v) for v in os.environ.get("REACH_Y", "-0.62,-0.52").split(","))
SPACE = float(os.environ.get("SPACE", "0.17"))     # min inter-item spacing (receptacles are large)
PARK = (float(os.environ.get("PARK_X", "3.5")), float(os.environ.get("PARK_Y", "3.5")))
DELIV_TOL = float(os.environ.get("DELIV_TOL", "0.13"))
MIN_LIFT = float(os.environ.get("MIN_LIFT", "0.04"))

ORD = ["obj"] + ["distr_%d" % i for i in range(8)]

# RoboCasa's native PlacementSampler can spin forever for some receptacle+seed combos (RESUME trap). Guard reset+
# randomize with SIGALRM; on timeout the caller rebuilds the env with a fresh seed and retries. Genuine reset ~1-3s.
# SIGALRM reset-guard: DISABLED by default (=0). It corrupts MuJoCo's C-level render context if it fires mid-render
# ('MjRenderContextOffscreen has no con'). With the 2-receptacle set the placement sampler no longer hangs, so no guard
# is needed. Kept (inert) for optional use on harder configs; alarm(0) never fires.
RESET_TIMEOUT = int(os.environ.get("RESET_TIMEOUT", "0"))
# native placement region (big -> the sampler fits obj+receptacles without rejection-looping; randomize() teleports
# everything anyway so the native positions are irrelevant, only that reset() SUCCEEDS quickly).
REG_W = float(os.environ.get("REG_W", "1.60"))
REG_H = float(os.environ.get("REG_H", "0.90"))


class _ResetTimeout(Exception):
    pass


def _alarm(sig, frm):
    raise _ResetTimeout()


signal.signal(signal.SIGALRM, _alarm)


class GoalPnP(PickPlaceCounterToSink):
    """target graspable 'obj' + REC_POOL receptacles as distractor objects (placed, not grasped)."""
    def __init__(self, obj_cat, rec_pool, *a, **k):
        self._obj_cat = obj_cat; self._rec_pool = list(rec_pool)
        super().__init__(obj_groups=obj_cat, *a, **k)

    def _get_obj_cfgs(self):
        # GENEROUS native region: everything is teleported by randomize() anyway, so the initial sample only needs to
        # SUCCEED (a tight region + 3 big receptacles makes the native PlacementSampler retry forever => reset hang).
        reg = dict(fixture=self.counter, sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                   size=(REG_W, REG_H), pos=("ref", -1.0))
        cfgs = [dict(name="obj", obj_groups=self._obj_cat, graspable=True, placement=dict(reg))]
        for i, c in enumerate(self._rec_pool):
            # receptacles: NOT required graspable (we never grasp them); heavy/stable placement targets
            cfgs.append(dict(name="distr_%d" % i, obj_groups=c, graspable=False, placement=dict(reg)))
        return cfgs


def make_env(obj_cat, seed):
    from robosuite.controllers import load_composite_controller_config
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return GoalPnP(obj_cat=obj_cat, rec_pool=REC_POOL, robots="PandaOmron", controller_configs=cc,
                   camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=CAMW, camera_heights=CAMW,
                   has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
                   ignore_done=True, seed=seed, layout_ids=LAYOUT, style_ids=STYLE,
                   translucent_robot=False, obj_instance_split=None, generative_textures=None)


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


def randomize(env, rng):
    """Teleport target + receptacles to random reachable (x,y), non-overlapping. Returns (obs, rec_names)."""
    names = list(env.objects.keys())
    placed = []
    for name in names:
        a0 = _obj_joint_addr(env, name)
        if a0 is None:
            continue
        z = float(env.sim.data.qpos[a0 + 2])
        x, y = None, None
        for _ in range(400):
            cx = rng.uniform(*REACH_X); cy = rng.uniform(*REACH_Y)
            if all((cx - px) ** 2 + (cy - py) ** 2 > SPACE ** 2 for px, py in placed):
                x, y = cx, cy; break
        if x is None:
            x, y = rng.uniform(*REACH_X), rng.uniform(*REACH_Y)
        env.sim.data.qpos[a0:a0 + 3] = [x, y, z + 0.02]
        placed.append((x, y))
    env.sim.forward()
    obs = None
    for _ in range(18):
        obs, _, _, _ = env.step(np.zeros(env.action_dim))
    return obs


def rec_name_for(env, key):
    cfg = next((c for c in env.object_cfgs if c.get("name") == key), None)
    g = cfg.get("obj_groups") if cfg else None
    return g if isinstance(g, str) else (g[0] if g else key)


def mkstate(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)


def servo(eef, wp, base_R):
    dw = np.asarray(wp, np.float32) - np.asarray(eef, np.float32)
    return np.clip((base_R.T @ dw) / SERVO, -1, 1).astype(np.float32)


def step_arm(env, robot, dpos, grip):
    ad = {"right": np.concatenate([np.asarray(dpos, np.float32), np.zeros(3, np.float32)]),
          "right_gripper": np.array([grip], np.float32)}
    return env.step(robot.create_action_vector(ad))


def gripper_far(env, target, thr=0.15):
    try:
        eef = np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]], np.float32)
        return float(np.linalg.norm(eef - objpos_env(env, target))) > thr
    except Exception:
        return True


def objpos_env(env, key):
    return np.asarray(env.sim.data.body_xpos[env.obj_body_id[key]], np.float32)


def delivered(env, obs, target, rec_keys, named_key, lifted):
    """INIT-FALSE success: target settled nearest the NAMED receptacle (& within tol) AND was lifted AND gripper far."""
    tp = objpos(obs, target)
    dists = {k: float(np.linalg.norm(tp[:2] - objpos(obs, k)[:2])) for k in rec_keys}
    nearest = min(dists, key=dists.get)
    return bool(lifted >= MIN_LIFT and nearest == named_key and dists[named_key] < DELIV_TOL and gripper_far(env, target))


def rollout(env, obs, target, named_key, rec_keys, record=True):
    robot = env.robots[0]
    tp0 = objpos(obs, target); z0 = tp0[2]
    rp = objpos(obs, named_key)                         # named receptacle world pos (destination)
    rec_top = float(rp[2]) + 0.03                       # approx top surface of receptacle
    tall = (z0 - 0.90) > TALL_REST_Z                    # crude tall-object flag (B27); most food is short
    hover = max(HOVER, (z0 - 0.90) + 0.12) if tall else HOVER
    ph, grip, hold, rel = "hover", GRIP_OPEN, 0, 0
    im, wr, st, ac = [], [], [], []
    frames = []
    obj_zmax, last_ph = z0, None
    base_R = quat2mat(np.asarray(obs["robot0_base_quat"], np.float64))
    for s in range(MAXS):
        eef = np.asarray(obs["robot0_eef_pos"], np.float32)
        tp = objpos(obs, target)
        obj_zmax = max(obj_zmax, float(tp[2]))
        if DEBUG and ph != last_ph:
            print("    ph=%-8s s=%3d dxy=%.3f dz=%.3f" % (ph, s, float(np.linalg.norm(eef[:2]-tp[:2])), float(eef[2]-tp[2])), flush=True); last_ph = ph
        if ph == "hover":
            wp = [tp[0], tp[1], tp[2] + hover]; grip = GRIP_OPEN
            if np.linalg.norm(eef[:2] - tp[:2]) < 0.02 and abs(eef[2] - (tp[2] + hover)) < 0.03:
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
            wp = [rp[0], rp[1], LIFT_Z]; grip = GRIP_CLOSE
            if np.linalg.norm(eef[:2] - rp[:2]) < 0.05: ph = "lower"
        elif ph == "lower":
            wp = [rp[0], rp[1], rec_top + DROP_DZ]; grip = GRIP_CLOSE
            if eef[2] < rec_top + DROP_DZ + 0.03: ph = "release"
        elif ph == "release":
            wp = [eef[0], eef[1], eef[2]]; grip = GRIP_OPEN; rel += 1
            if rel >= 6: ph = "retract"
        else:
            wp = [eef[0], eef[1], LIFT_Z + 0.20]; grip = GRIP_OPEN; rel += 1
        dpos = servo(eef, wp, base_R)
        if record:
            im.append(np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1]))
            wr.append(np.ascontiguousarray(np.asarray(obs[CAM_WRIST + "_image"])[::-1]))
            st.append(mkstate(obs))
            ac.append(np.concatenate([dpos, np.zeros(3, np.float32), [grip]]).astype(np.float32))
        if RENDER and s % 8 == 0:
            frames.append(np.asarray(obs[CAM_AGENT + "_image"])[::-1])
        obs, r, done, info = step_arm(env, robot, dpos, grip)
        lifted = obj_zmax - z0
        if ph == "retract" and rel >= 20 and delivered(env, obs, target, rec_keys, named_key, lifted):
            break
        if rel >= 28:
            break
    lifted = obj_zmax - z0
    succ = delivered(env, obs, target, rec_keys, named_key, lifted)
    if DEBUG:
        print("  END succ=%s lift=%.3f ph=%s" % (succ, lifted, ph), flush=True)
    return succ, (im, wr, st, ac), frames


def main():
    rng = np.random.RandomState(SEED)
    n_ok = 0; per_rec = {}
    # env is expensive to build (obj_groups fixed at construction) => one env per target-object block, reset per demo.
    # destinations are role-balanced GLOBALLY (dst_counter cycles, offset by shard) so layout ⟂ dest and each rec is goal ~equally.
    n_per_obj = -(-NDEMOS // len(OBJ_POOL))              # ceil
    dst_counter = SHARD
    got = 0
    for obj_cat in OBJ_POOL:
        if got >= NDEMOS:
            break
        env = make_env(obj_cat, SEED + got)
        block_got, tries = 0, 0
        while block_got < n_per_obj and got < NDEMOS and tries < n_per_obj * 8:
            tries += 1
            try:
                signal.alarm(RESET_TIMEOUT)
                env.reset()
                rec_keys = [k for k in env.objects.keys() if k != "obj"]
                obs = randomize(env, rng)
                signal.alarm(0)
            except _ResetTimeout:
                signal.alarm(0)
                print("RESET_TIMEOUT tries=%d — rebuilding env" % tries, flush=True)
                try:
                    env.close()
                except Exception:
                    pass
                env = make_env(obj_cat, SEED + got + tries * 1000)
                continue
            except Exception as e:
                signal.alarm(0)
                print("PLACE_ERR", e, flush=True); traceback.print_exc(); continue
            named_key = rec_keys[dst_counter % len(rec_keys)]
            rec_cat = rec_name_for(env, named_key)
            prep = PREP.get(rec_cat, "in")
            lang = "put the %s %s the %s" % (obj_cat.replace("_", " "), prep, rec_cat.replace("_", " "))
            if DEBUG:
                print("EP got=%d obj=%s dest=%s(%s) lang='%s'" % (got, obj_cat, named_key, rec_cat, lang), flush=True)
            try:
                succ, data, frames = rollout(env, obs, "obj", named_key, rec_keys, record=True)
            except Exception as e:
                print("ROLLOUT_ERR", e, flush=True); traceback.print_exc(); continue
            if RENDER and frames:
                strip = np.concatenate([f for f in frames[:16]], axis=1)
                imageio.imwrite("%s/roll_s%d_%s_%s_%d_%s.png" % (VIZ, SHARD, obj_cat, rec_cat, tries, "S" if succ else "F"), strip)
            if not succ:
                continue
            im, wr, st, ac = data
            np.savez_compressed(
                RAW + "/ep_%d_%04d.npz" % (SHARD, got),
                img=np.asarray(im, np.uint8), wrist=np.asarray(wr, np.uint8),
                state=np.asarray(st, np.float32), action=np.asarray(ac, np.float32),
                lang=lang, dest_cat=rec_cat, obj_cat=obj_cat, named_key=named_key,
                dest_xy=np.asarray(objpos(obs, named_key)[:2], np.float32),
            )
            per_rec[rec_cat] = per_rec.get(rec_cat, 0) + 1
            got += 1; block_got += 1; dst_counter += 1
        env.close()
    print("SHARD%d_DONE total=%d recs=%s" % (SHARD, n_ok if False else got, json.dumps(per_rec)), flush=True)


if __name__ == "__main__":
    main()
