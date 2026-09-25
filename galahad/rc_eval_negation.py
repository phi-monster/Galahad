"""rc_eval_negation.py (/root/rc_venv) — RoboCasa NEGATION-deconf GROUNDING eval; env-side driver, client to a policy server.

Generalises the CATEGORY template rc_eval.py to the NEGATION referent type (Galahad G1 universal battery).
NEGATION scene (from collect_rc_negation.py): EXACTLY 2 objects {named, target} on the counter.
  env = DeconfPnP(pair=[target, named])  =>  scene-key 'obj'=target (the object to grasp), 'distr_0'=named (the negated one).
  (The policy never sees scene-keys, so putting the TARGET at 'obj' is invisible to it — it only fixes bookkeeping/metric.)

B4 un-fakeable battery (mirrors rc_eval.py; NEGATION variant — the disambiguating cue is the word "not"):
  --mode task : SUBSTITUTION SR. instr = "Pick the object that is not the {named} ..." -> correct grasp = target (='obj').
                Headline negation-grounding number. A model that IGNORES "not" (entity-matches {named}) grasps the WRONG
                object => at/below chance. Success = target genuinely picked-and-placed in sink (see METRIC note below).
  --mode occ  : same as task but BOTH cameras blanked each step -> MUST collapse to ~0 (else not using pixels => void).
  --mode swap : SAME scene (deterministic per-pair seed => byte-identical placement to task), FLIP which object is negated:
                instr = "not the {target}" -> correct grasp switches to the OTHER real object = named (='distr_0').
                Metric = did the hand go to the flipped target? (argmin per-object min-eef-dist == 'distr_0'). Clean
                negation-parsing => hand follows the flip; a fixed motor habit (always same object) => it doesn't.

METRIC (load-bearing, DIFFERS from rc_eval.py on purpose — verified on box4 2026-07-16):
  env._check_success() is UNUSABLE here: = obj_inside_of(obj, sink, partial_check=True) AND gripper_obj_far. With
  collect_rc_negation's REACH region overlapping the sink, ~half the objects spawn IN the basin, so _check_success is
  TRUE at INIT in 75% (24/32) of scenes BEFORE any manipulation (§2 smoke-bomb — a do-nothing policy would "score" 75%).
  => TASK/OCC success uses the collect ORACLE's own definition, imported verbatim: in_sink(env,obs,'obj') AND the object
  was genuinely LIFTED (obj_zmax - obj_z_start > MIN_LIFT). This is init-False and rejects spawned-in / knocked-in
  degeneracy; it is also the SAME success gate the training demos were selected by (train==eval, LAB B47).

REACH inherited from collect_rc_negation module globals (single source of truth): REACH_X, REACH_Y, SPACE. Pass the SAME
REACH_X/REACH_Y env vars the collection used (defaults 1.05,1.70 / -0.55,-0.28 / SPACE 0.10) so placement matches training.
POOL = apple,banana,lemon,carrot (NO can — 0/11 cylinder motor-trap + roll-into-sink §2 smoke-bomb, dropped in collect).

Run (per mode; 12 ordered pairs x --n => n>=48 for Wilson):
  RENDER=0 REACH_X=1.05,1.70 REACH_Y=-0.55,-0.28 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
    /root/rc_venv/bin/python rc_eval_negation.py --mode task --pool apple,banana,lemon,carrot --n 4 --port 5600
"""
import os, sys, socket, struct, pickle, argparse, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robosuite.utils.transform_utils import quat2axisangle
from robosuite.controllers import load_composite_controller_config
sys.path.insert(0, "/root")
from collect_rc_negation import (DeconfPnP, randomize_objs, in_sink, objpos,
                                 CAM_AGENT, CAM_WRIST, NEG_TMPL, MIN_LIFT, REACH_X, REACH_Y, SPACE)

EVIZ = "/dev/shm/rc_neg_eval_viz"; os.makedirs(EVIZ, exist_ok=True)


def _recv_all(conn, n):
    buf = b""
    while len(buf) < n:
        d = conn.recv(n - len(buf))
        if not d:
            return None
        buf += d
    return buf


def recv_msg(conn):
    hdr = _recv_all(conn, 4)
    return None if hdr is None else pickle.loads(_recv_all(conn, struct.unpack(">I", hdr)[0]))


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(data)) + data)


def mkstate(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)


def eef_pos_obs(obs):
    return np.asarray(obs["robot0_eef_pos"], np.float32)


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))


def make_env(target, named, layout, style, camw, seed):
    # pair=[target, named] => 'obj'=target (grasp-me in TASK), 'distr_0'=named (the negated object).
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return DeconfPnP(pair=[target, named], robots="PandaOmron", controller_configs=cc,
                     camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=camw, camera_heights=camw,
                     has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
                     ignore_done=True, seed=seed, layout_ids=layout, style_ids=style,
                     translucent_robot=False, obj_instance_split=None, generative_textures=None)


def main(a):
    pool = a.pool.split(",")
    pairs = [(x, y) for x in pool for y in pool if x != y]   # (named, target): TASK negates x, grasp y
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(("127.0.0.1", a.port))
    send_msg(conn, {"cmd": "ping"}); assert recv_msg(conn).get("ok"), "server not ready"
    tot_s = tot = 0; swap_hit = swap_tot = 0
    for pidx, (named, target) in enumerate(pairs):
        env = make_env(target, named, a.layout, a.style, a.camw, a.seed)
        rng = np.random.RandomState(a.seed + pidx * 101)   # deterministic => task/occ/swap share the SAME scenes per (pair,j)
        s = 0      # task/occ: TARGET-in-sink successes this pair ; swap: hand->correct (obeyed) this pair
        for j in range(a.n):
            env.reset()
            obs, _ = randomize_objs(env, rng)   # SAME reachable placement as training (collect REACH globals); returns (obs, saved)
            send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
            keys = list(env.objects.keys())   # ['obj','distr_0']
            if a.mode == "swap":
                neg_word = target       # FLIP: negate the target -> correct grasp = the OTHER (named) object
                correct_key = "distr_0"
            else:                       # task / occ
                neg_word = named
                correct_key = "obj"     # = target
            instr = NEG_TMPL.format(a=neg_word)
            tp0 = float(objpos(obs, "obj")[2])   # target ('obj') z at start (for MIN_LIFT genuine-lift gate)
            objzmax = tp0
            mind = {k: 1e9 for k in keys}
            frames = []
            succ = False
            for step in range(a.max_steps):
                img = np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1])
                wr = np.ascontiguousarray(np.asarray(obs[CAM_WRIST + "_image"])[::-1])
                if a.render and step % 10 == 0:
                    frames.append(img.copy())   # UN-blanked, so render shows the real scene even in occ mode
                if a.mode == "occ":
                    img = np.zeros_like(img); wr = np.zeros_like(wr)
                send_msg(conn, {"cmd": "act", "blank": a.mode == "occ",
                                "obs": {"image_b": img.tobytes(), "image_shape": list(img.shape), "image_dtype": str(img.dtype),
                                        "wrist_b": wr.tobytes(), "wrist_shape": list(wr.shape), "wrist_dtype": str(wr.dtype),
                                        "instruction": instr, "state": mkstate(obs).tolist()}})
                act = np.asarray(recv_msg(conn)["action"], np.float32).ravel()
                ad = {"right": np.concatenate([act[:6], np.zeros(max(0, 6 - len(act)))])[:6],
                      "right_gripper": np.array([act[6] if len(act) > 6 else -1.0], np.float32)}
                obs, r, done, info = env.step(env.robots[0].create_action_vector(ad))
                ep = eef_pos_obs(obs)
                for k in keys:
                    mind[k] = min(mind[k], float(np.linalg.norm(ep - objpos(obs, k))))
                objzmax = max(objzmax, float(objpos(obs, "obj")[2]))
                if a.mode != "swap" and in_sink(env, obs, "obj") and (objzmax - tp0 > MIN_LIFT):
                    succ = True; break   # genuine pick-and-place of the TARGET into the sink (init-False, lift-gated)
            hand_key = min(mind, key=mind.get)   # object the EEF got closest to = what the policy targeted
            if a.mode == "swap":
                obeyed = (hand_key == correct_key)
                swap_hit += int(obeyed); swap_tot += 1; s += int(obeyed)
                tag = "OBEY" if obeyed else ("DISOBEY_hand-%s" % hand_key)
            else:
                s += int(succ); tot += 1
                tag = ("S" if succ else "F") + "_hand-%s" % hand_key   # expose WHICH object was targeted
            if a.render and frames:
                strip = np.concatenate(frames[:16], axis=1)
                imageio.imwrite("%s/%s_neg-%s_tgt-%s_%d_%s.png" % (EVIZ, a.mode, named, target, j, tag), strip)
        if a.mode == "swap":
            print("PAIR swap neg-%s correct-%s: obeyed %d/%d" % (target, named, s, a.n), flush=True)
        else:
            print("PAIR neg-%s grasp-%s: %d/%d" % (named, target, s, a.n), flush=True)
        tot_s += s
        env.close()
    if a.mode == "swap":
        lo = wilson(swap_hit, swap_tot)
        print("[SWAP] hand->named %d/%d = %.1f%% CI[%.1f,%.1f]" % (swap_hit, swap_tot, *lo), flush=True)
    else:
        lo = wilson(tot_s, tot)
        print("[%s] SR %d/%d = %.1f%% CI[%.1f,%.1f]" % (a.mode.upper(), tot_s, tot, *lo), flush=True)
    print("RC_EVAL_NEGATION_DONE", flush=True)
    print("RC_EVAL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="task", choices=["task", "occ", "swap"])
    ap.add_argument("--pool", default="apple,banana,lemon,carrot")   # NO can (motor-trap / smoke-bomb, dropped in collect)
    ap.add_argument("--n", type=int, default=4)                      # PER ordered pair; 12 pairs x 4 = 48 (Wilson floor)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--camw", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--render", action="store_true")
    main(ap.parse_args())
