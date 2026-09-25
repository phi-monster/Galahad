"""rc_eval_colour.py (/root/rc_venv) — RoboCasa COLOUR-deconf grounding eval (env-side driver, client to a policy server).

Un-fakeable battery (mirrors rc_eval.py; COLOUR variant — the disambiguating cue is COLOUR, not category):
  --mode task : SUBSTITUTION SR. obj is recoloured to a rotating target colour; instruction names that colour.
                Success = obj in sink + gripper far. Headline grounding number.
  --mode occ  : same but BOTH cameras blanked -> MUST collapse to ~0 (else not using pixels => grounding void).
  --mode swap : SAME scene, instruction names a DIFFERENT present colour (a distractor's). Did the hand go to the
                newly-named COLOUR's object? Clean colour-grounding => hand follows the named colour.

INSTR format is IDENTICAL to collect_rc_colour.py (train==eval; LAB B47 mismatch lesson).
Run: /root/rc_venv/bin/python rc_eval_colour.py --mode task --base_cat can --ncol 3 --n 12 --port 5600
"""
import os, sys, socket, struct, pickle, argparse, math, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robosuite.utils.transform_utils import quat2axisangle
from robosuite.controllers import load_composite_controller_config
sys.path.insert(0, "/root")
from collect_rc_colour import (DeconfPnPColour, CAM_AGENT, CAM_WRIST, randomize_objs,
                               recolour, assign_colours, ALL_COLOURS, OBJ_WORD)

EVIZ = "/root/collect_colour/eval_viz"; os.makedirs(EVIZ, exist_ok=True)

# §2 smoke-bomb fix (LAB B47-negation): gate TASK/OCC success on a GENUINE lift so raw env._check_success()
# (obj_inside_of partial_check, TRUE at init wherever REACH overlaps the sink) can't fake a do-nothing success.
# Init-False by construction; genuine oracle demos lift >MIN_LIFT so train==eval holds.
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


def make_env(base_cat, ncol, layout, style, camw, seed):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return DeconfPnPColour(base_cat=base_cat, n_obj=ncol, robots="PandaOmron", controller_configs=cc,
                           camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=camw, camera_heights=camw,
                           has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
                           ignore_done=True, seed=seed, layout_ids=layout, style_ids=style,
                           translucent_robot=False, obj_instance_split=None, generative_textures=None)


def main(a):
    colour_names = list(ALL_COLOURS.keys())[:a.ncol]
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(("127.0.0.1", a.port))
    send_msg(conn, {"cmd": "ping"}); assert recv_msg(conn).get("ok"), "server not ready"
    tot_s = tot = 0; swap_hit = swap_tot = 0
    env = make_env(a.base_cat, a.ncol, a.layout, a.style, a.camw, a.seed)
    for ti, target_colour in enumerate(colour_names):
        rng = np.random.RandomState(a.seed + ti * 101)
        s = 0
        for j in range(a.n):
            env.reset()
            obs = randomize_objs(env, rng)
            obj_names = list(env.objects.keys())
            slot_colour = assign_colours(target_colour, rng, obj_names)  # obj=target_colour
            recolour(env, slot_colour)
            env.sim.forward(); obs, _, _, _ = env.step(np.zeros(env.action_dim))
            send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
            colour_to_key = {v: k for k, v in slot_colour.items()}
            if a.mode == "swap":
                distr_colours = [c for c in slot_colour.values() if c != target_colour]
                named_colour = distr_colours[j % len(distr_colours)]
                named_key = colour_to_key[named_colour]
            else:
                named_colour = target_colour; named_key = "obj"
            instr = "pick the %s %s from the counter and place it in the sink" % (named_colour, OBJ_WORD)
            keys = obj_names
            mind = {k: 1e9 for k in keys}
            obj_z0 = float(obj_pos(env, "obj")[2]); obj_zmax = obj_z0   # lift-gate anchor: target ('obj') z at init
            frames = []
            for step in range(a.max_steps):
                img = np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1])
                wr = np.ascontiguousarray(np.asarray(obs[CAM_WRIST + "_image"])[::-1])
                if a.render and step % 10 == 0:
                    frames.append(img.copy())
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
            hand_key = min(mind, key=mind.get)
            hand_colour = slot_colour.get(hand_key, "?")
            if a.mode == "swap":
                obeyed = (hand_key == named_key)
                swap_hit += int(obeyed); swap_tot += 1
                tag = "OBEY" if obeyed else ("DISOBEY_hand-%s" % hand_colour)
            else:
                succ = bool(env._check_success()) and (obj_zmax - obj_z0 > MIN_LIFT)   # §2 lift-gated (init-False)
                s += int(succ); tot += 1
                tag = ("S" if succ else "F") + "_hand-%s" % hand_colour
            if a.render and frames:
                strip = np.concatenate(frames[:16], axis=1)
                imageio.imwrite("%s/%s_%s_%d_%s.png" % (EVIZ, a.mode, target_colour, j, tag), strip)
        print("TARGET_COLOUR %s: %d/%d" % (target_colour, s, a.n), flush=True)
    env.close()
    if a.mode == "swap":
        lo = wilson(swap_hit, swap_tot)
        print("[SWAP] hand->named-colour %d/%d = %.1f%% CI[%.1f,%.1f]" % (swap_hit, swap_tot, *lo), flush=True)
    else:
        lo = wilson(tot_s, tot)
        print("[%s] SR %d/%d = %.1f%% CI[%.1f,%.1f]" % (a.mode.upper(), tot_s, tot, *lo), flush=True)
    print("RC_EVAL_COLOUR_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="task", choices=["task", "occ", "swap"])
    ap.add_argument("--base_cat", default="can")
    ap.add_argument("--ncol", type=int, default=3)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--camw", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--render", action="store_true")
    main(ap.parse_args())
