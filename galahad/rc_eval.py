"""rc_eval.py (/root/rc_venv) — RoboCasa deconf GROUNDING eval; env-side driver, client to a policy server.

B4 un-fakeable battery (mirrors libero_pro_eval + occ_test):
  --mode task : SUBSTITUTION SR. Each pool object is the NAMED target (correct instruction). Success = named obj in
                sink + gripper far. This is the headline grounding number (must beat base MolmoAct2 zero-shot).
  --mode occ  : same as task but BOTH cameras blanked each step. Success MUST collapse to ~0 (else the policy isn't
                using pixels => grounding claim void).
  --mode swap : SAME scene, instruction NAMES A DIFFERENT present object (a distractor). Metric = did the hand go to
                the newly-named object? (argmin over per-object min gripper distance == named object's scene key).
                Clean grounding => hand follows the name; a fixed motor habit => it doesn't.

Object positions + eef via env.sim (body_xpos / eef site) — robust to obs-key naming. n>=48, Wilson CI.
Run: RENDER=0 /root/rc_venv/bin/python rc_eval.py --mode task --pool apple,banana,lemon,carrot,can --n 10 --port 5600
"""
import os, sys, socket, struct, pickle, argparse, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robosuite.utils.transform_utils import quat2axisangle
from robosuite.controllers import load_composite_controller_config
sys.path.insert(0, "/root")
from collect_rc_deconf import DeconfPnP, CAM_AGENT, CAM_WRIST, randomize_objs

EVIZ = "/dev/shm/rc_eval_viz"; os.makedirs(EVIZ, exist_ok=True)

# §2 smoke-bomb fix (LAB B47-negation): raw env._check_success() = obj_inside_of(obj,sink,partial_check=True) is TRUE at
# INIT wherever the collect REACH overlaps the sink basin (measured 77.8% init in the negation region). Gate TASK/OCC
# success on a GENUINE lift (obj_zmax - obj_z_start > MIN_LIFT) so spawned-in / knocked-in degeneracy is rejected and the
# metric is init-False by construction (region-independent). Genuine oracle demos lift >MIN_LIFT so train==eval holds.
MIN_LIFT = float(os.environ.get("MIN_LIFT", "0.08"))   # matches collect_rc_negation.MIN_LIFT default


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


def readable(cat):
    return cat.replace("_", " ")


def eef_pos(env):
    return np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]], np.float32)


def obj_pos(env, key):
    return np.asarray(env.sim.data.body_xpos[env.obj_body_id[key]], np.float32)


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))


def make_env(pool, target, layout, style, camw, seed):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return DeconfPnP(pool=pool, target_cat=target, robots="PandaOmron", controller_configs=cc,
                     camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=camw, camera_heights=camw,
                     has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
                     ignore_done=True, seed=seed, layout_ids=layout, style_ids=style,
                     translucent_robot=False, obj_instance_split=None, generative_textures=None)


def cat_of(env, key):
    # scene object key ("obj"/"distr_i") -> its category name (for matching the named object in swap)
    cfg = next((c for c in env.object_cfgs if c.get("name") == key), None)
    g = cfg.get("obj_groups") if cfg else None
    return g if isinstance(g, str) else (g[0] if g else key)


def main(a):
    pool = a.pool.split(",")
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(("127.0.0.1", a.port))
    send_msg(conn, {"cmd": "ping"}); assert recv_msg(conn).get("ok"), "server not ready"
    tot_s = tot = 0; swap_hit = swap_tot = 0
    per = []
    for target in pool:
        env = make_env(pool, target, a.layout, a.style, a.camw, a.seed)
        rng = np.random.RandomState(a.seed + (hash(target) % 1000))
        s = 0
        for j in range(a.n):
            env.reset()
            obs = randomize_objs(env, rng)   # SAME reachable-counter placement as training (needs matching REACH_X/Y env)
            send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
            keys = list(env.objects.keys())
            frames = []
            if a.mode == "swap":
                distr = [k for k in keys if k != "obj"]
                named_key = distr[j % len(distr)]
                named_cat = cat_of(env, named_key)
            else:
                named_key = "obj"; named_cat = cat_of(env, "obj")
            instr = "Pick the %s from the counter and place it in the sink." % readable(named_cat)
            mind = {k: 1e9 for k in keys}
            obj_z0 = float(obj_pos(env, "obj")[2]); obj_zmax = obj_z0   # lift-gate anchor: target ('obj') z at init
            for step in range(a.max_steps):
                img = np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1])
                wr = np.ascontiguousarray(np.asarray(obs[CAM_WRIST + "_image"])[::-1])
                if a.render and step % 10 == 0:
                    frames.append(img.copy())   # UN-blanked, so the render shows the real scene even in occ mode
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
                ep = eef_pos(env)
                for k in keys:
                    mind[k] = min(mind[k], float(np.linalg.norm(ep - obj_pos(env, k))))
                obj_zmax = max(obj_zmax, float(obj_pos(env, "obj")[2]))   # track genuine lift of the target
                if a.mode != "swap" and bool(env._check_success()) and (obj_zmax - obj_z0 > MIN_LIFT):
                    break
            hand_key = min(mind, key=mind.get)   # object the EEF got closest to = what the policy targeted
            if a.mode == "swap":
                obeyed = (hand_key == named_key)
                swap_hit += int(obeyed); swap_tot += 1
                tag = "OBEY" if obeyed else ("DISOBEY_hand-%s" % hand_key)
            else:
                succ = bool(env._check_success()) and (obj_zmax - obj_z0 > MIN_LIFT)   # §2 lift-gated (init-False)
                s += int(succ); tot += 1
                tag = ("S" if succ else "F") + "_hand-%s" % hand_key   # note3: expose WHICH object was targeted
            if a.render and frames:
                strip = np.concatenate(frames[:16], axis=1)
                imageio.imwrite("%s/%s_%s_%d_%s.png" % (EVIZ, a.mode, target, j, tag), strip)
        per.append((target, s, a.n)); tot_s += s
        print("TARGET %s: %d/%d" % (target, s, a.n), flush=True)
        env.close()
    if a.mode == "swap":
        lo = wilson(swap_hit, swap_tot)
        print("[SWAP] hand->named %d/%d = %.1f%% CI[%.1f,%.1f]" % (swap_hit, swap_tot, *lo), flush=True)
    else:
        lo = wilson(tot_s, tot)
        print("[%s] SR %d/%d = %.1f%% CI[%.1f,%.1f]" % (a.mode.upper(), tot_s, tot, *lo), flush=True)
    print("RC_EVAL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="task", choices=["task", "occ", "swap"])
    ap.add_argument("--pool", default="apple,banana,lemon,carrot,can")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--camw", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--render", action="store_true")   # save eval-rollout filmstrips (note3: see WHICH object grasped)
    main(ap.parse_args())
