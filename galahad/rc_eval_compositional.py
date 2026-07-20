"""rc_eval_compositional.py (/root/rc_venv) — RoboCasa COMPOSITIONAL (COLOUR x SIZE conjunction) grounding eval.
Env-side driver, client to a policy server. Generalizes rc_eval.py / rc_eval_colour.py to a 2-attribute conjunction.

SCENE (byte-identical construction to collect_rc_cs.py — train==eval; LAB B47 mismatch lesson): a 2x2 factorial of
{target_colour,other_colour} x {target_size,other_size} on a SINGLE base category (shape constant). Drop-one:
  obj     = (target_size, target_colour)   <- ONLY object matching BOTH  => the conjunction target
  distr_0 = (other_size , target_colour)   <- COLOUR-match (flip size)   => unique (other_size,target_colour)
  distr_1 = (target_size, other_colour )   <- SIZE-match   (flip colour) => unique (target_size,other_colour)
  distr_2 = (other_size , other_colour )   <- neutral
"the {colour}"->{obj,distr_0} ambiguous ; "the {size}"->{obj,distr_1} ambiguous ; only "{size} {colour}"=obj unique.
A single-attribute-only model MUST fail => un-fakeable conjunction grounding.

Un-fakeable battery (ALL 3 modes welded — TASK-only is a §2 smoke-bomb):
  --mode task : conjunction SR. instr NAMES BOTH attributes; target = obj (the unique conjunction);
                Success = env._check_success() (obj in sink + gripper far). Headline grounding number.
  --mode occ  : identical scene/instr but BOTH cameras blanked each step -> MUST collapse ~0 (else not using pixels
                => conjunction-grounding claim void).
  --mode swap : SAME scene, FLIP exactly ONE attribute so the named target moves to a DIFFERENT present distractor
                that EXISTS in the 2x2 (flip size -> distr_0 ; flip colour -> distr_1). OBEYED = EEF's
                closest-approached object == the newly-named conjunction's key. Rotates flip-size/flip-colour over j
                so BOTH the size-word and the colour-word are tested (a model that follows only one attribute is
                caught). Clean conjunction-grounding => hand follows the flipped name; a fixed motor habit => it doesn't.

SIZE is build-time (object_scale) => env is REBUILT per target_size (mirrors collect_rc_cs). COLOUR is runtime recolour.
REACH_X/REACH_Y/SPACE + SS/SL/SIZES/COLOUR_POOL/CAT are read from ENV VARS by collect_rc_cs at import => set the SAME
env the collector used so placement/sizes/colours match training.

Socket contract byte-identical to rc_eval.py (ping/reset/act; obs=image_b+wrist_b+instruction+state[8]; action[7];
images [::-1]). Object positions + eef via env.sim (obj_body_id / eef site) — robust to obs-key naming. n>=48, Wilson CI.

Run (task; set the SAME env vars run_cs.sh used for collection — CAT=carrot [elongated=graspable; round can/lemon
= 0-grasp], SS=0.55 SL=1.0). occ/swap identical but --mode occ / --mode swap:
  CAT=carrot COLOUR_POOL=red,green,blue SIZES=small,large SS=0.55 SL=1.0 REACH_X=1.72,1.90 REACH_Y=-0.42,-0.14 SPACE=0.09 \\
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 \\
  /root/rc_venv/bin/python rc_eval_compositional.py --mode task --n 8 --port 5600
"""
import os, sys, socket, struct, pickle, argparse, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
from robosuite.utils.transform_utils import quat2axisangle
from robosuite.controllers import load_composite_controller_config
sys.path.insert(0, "/root")
# collect_rc_cs reads REACH_X/Y/SPACE/SS/SL/SIZES/COLOUR_POOL/CAT from ENV at import => set the collector's env to match.
from collect_rc_cs import (DeconfPnPCS, CAM_AGENT, CAM_WRIST, randomize_objs, recolour,
                           ALL_COLOURS, OBJ_WORD, SIZE_SCALE, SIZE_POOL, COLOUR_POOL, CAT)

EVIZ = os.environ.get("EVIZ", "/root/collect_compositional/eval_viz"); os.makedirs(EVIZ, exist_ok=True)

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


def make_env(base_cat, slot_scale, layout, style, camw, seed):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return DeconfPnPCS(cat=base_cat, slot_scale=slot_scale, robots="PandaOmron", controller_configs=cc,
                       camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=camw, camera_heights=camw,
                       has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
                       ignore_done=True, seed=seed, layout_ids=layout, style_ids=style,
                       translucent_robot=False, obj_instance_split=None, generative_textures=None)


def instr_of(size, colour):
    # BYTE-IDENTICAL to collect_rc_cs.py lang: "pick the {size} {colour} {OBJ_WORD} and place it in the sink"
    return "pick the %s %s %s and place it in the sink" % (size, colour, OBJ_WORD)


def main(a):
    colours = COLOUR_POOL[:a.ncol]
    sizes = list(SIZE_POOL)
    assert len(sizes) == 2, "compositional size axis needs exactly 2 sizes, got %r" % (sizes,)
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(("127.0.0.1", a.port))
    send_msg(conn, {"cmd": "ping"}); assert recv_msg(conn).get("ok"), "server not ready"

    tot_s = tot = 0
    # swap tracked overall + split by which attribute was flipped (diagnostic: does it follow size AND colour?)
    sh = {"all": 0, "size": 0, "colour": 0}; st = {"all": 0, "size": 0, "colour": 0}
    per = []

    for target_size in sizes:                                   # env REBUILT per size (object_scale is build-time)
        other_size = [s for s in sizes if s != target_size][0]
        slot_scale = {"obj": SIZE_SCALE[target_size], "distr_1": SIZE_SCALE[target_size],
                      "distr_0": SIZE_SCALE[other_size], "distr_2": SIZE_SCALE[other_size]}
        env = make_env(a.base_cat, slot_scale, a.layout, a.style, a.camw, a.seed)
        for ci, target_colour in enumerate(colours):            # COLOUR via runtime recolour (no rebuild)
            rng = np.random.RandomState(a.seed + (hash((target_size, target_colour)) % 100000))
            other_colour = str(rng.choice([c for c in colours if c != target_colour]))
            slot_colour = {"obj": target_colour, "distr_0": target_colour,
                           "distr_1": other_colour, "distr_2": other_colour}
            s = 0
            for j in range(a.n):
                env.reset()
                obs = randomize_objs(env, rng)                  # SAME reachable placement as training (matching REACH env)
                recolour(env, slot_colour)
                env.sim.forward(); obs, _, _, _ = env.step(np.zeros(env.action_dim))
                send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
                keys = list(env.objects.keys())

                if a.mode == "swap":
                    if j % 2 == 0:                              # FLIP SIZE -> distr_0 = (other_size, target_colour)
                        named_key = "distr_0"; flip = "size"
                        instr = instr_of(other_size, target_colour)
                    else:                                       # FLIP COLOUR -> distr_1 = (target_size, other_colour)
                        named_key = "distr_1"; flip = "colour"
                        instr = instr_of(target_size, other_colour)
                else:                                           # task / occ: named = conjunction obj
                    named_key = "obj"; flip = None
                    instr = instr_of(target_size, target_colour)

                mind = {k: 1e9 for k in keys}
                obj_z0 = float(obj_pos(env, "obj")[2]); obj_zmax = obj_z0   # lift-gate anchor: target ('obj') z at init
                frames = []
                for step in range(a.max_steps):
                    img = np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1])
                    wr = np.ascontiguousarray(np.asarray(obs[CAM_WRIST + "_image"])[::-1])
                    if a.render and step % 10 == 0:
                        frames.append(img.copy())               # UN-blanked, so render shows real scene even in occ
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
                hand_key = min(mind, key=mind.get)              # object the EEF got closest to = what the policy targeted
                if a.mode == "swap":
                    obeyed = (hand_key == named_key)
                    sh["all"] += int(obeyed); st["all"] += 1
                    sh[flip] += int(obeyed); st[flip] += 1
                    tag = ("OBEY_%s" % flip) if obeyed else ("DISOBEY_%s_hand-%s" % (flip, hand_key))
                else:
                    succ = bool(env._check_success()) and (obj_zmax - obj_z0 > MIN_LIFT)   # §2 lift-gated (init-False)
                    s += int(succ); tot += 1
                    tag = ("S" if succ else "F") + "_hand-%s" % hand_key
                if a.render and frames:
                    strip = np.concatenate(frames[:16], axis=1)
                    imageio.imwrite("%s/%s_%s_%s_%d_%s.png" % (EVIZ, a.mode, target_size, target_colour, j, tag), strip)
            cell = "%s_%s" % (target_size, target_colour)
            if a.mode != "swap":
                tot_s += s
                per.append((cell, s, a.n))
                print("CELL %s: %d/%d" % (cell, s, a.n), flush=True)
            else:
                print("CELL %s: swap done (see split below)" % cell, flush=True)
        env.close()

    if a.mode == "swap":
        lo = wilson(sh["all"], st["all"])
        losz = wilson(sh["size"], st["size"]); locol = wilson(sh["colour"], st["colour"])
        print("[SWAP] flip_size  hand->named %d/%d = %.1f%% CI[%.1f,%.1f]" % (sh["size"], st["size"], *losz), flush=True)
        print("[SWAP] flip_colour hand->named %d/%d = %.1f%% CI[%.1f,%.1f]" % (sh["colour"], st["colour"], *locol), flush=True)
        print("[SWAP] hand->named %d/%d = %.1f%% CI[%.1f,%.1f]" % (sh["all"], st["all"], *lo), flush=True)
    else:
        lo = wilson(tot_s, tot)
        print("[%s] SR %d/%d = %.1f%% CI[%.1f,%.1f]" % (a.mode.upper(), tot_s, tot, *lo), flush=True)
    print("RC_EVAL_CS_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="task", choices=["task", "occ", "swap"])
    ap.add_argument("--base_cat", default=CAT)          # single base category (shape constant); defaults to collector CAT
    ap.add_argument("--ncol", type=int, default=len(COLOUR_POOL))  # #colours to rotate as target (from COLOUR_POOL)
    ap.add_argument("--n", type=int, default=8)         # episodes per (size,colour) cell => n * 2 sizes * ncol total
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--camw", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--render", action="store_true")    # save eval-rollout filmstrips (see WHICH object grasped)
    main(ap.parse_args())
