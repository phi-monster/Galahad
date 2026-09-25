"""libero_goal_vary_eval.py — DISCONFIRMING CONTROL for the "base goal-100% = memorization vs grounding" question.

libero_goal = ONE fixed kitchen scene, 10 goal instructions, SAME 4 free objects every task
(akita_black_bowl_1, cream_cheese_1, wine_bottle_1, plate_1; fixtures wooden_cabinet_1/stove/wine_rack never move).
Canonical init jitter is ~2-3cm. If base's 100% is MEMORIZATION (canned instruction->trajectory tuned to canonical
positions), displacing the free objects >>jitter while keeping the (:language) goal instruction FIXED must break it.
If base's 100% is GROUNDING (reads current object positions), it survives the displacement.

Same validated env contract as libero_pro_eval.py (180deg agentview+wrist flip, state=[eef_pos,quat2axisangle,
gripper_qpos] 8-dim, 7-DoF OSC action, set_init_state + settle, done=goal-predicate-satisfied). The ONLY addition is
`--vary`: after set_init_state+settle, re-place the 4 free objects to new (x,y) inside a validated manipuland zone
(the bounding hull of the 4 canonical slots, clear of all fixtures), min-displacement DMIN from canonical, min inter-
object sep SEP, z+quaternion+instruction unchanged. This IS the reusable G1 goal-deconf generator (`randomize_goal`).

Modes: default(canonical) | --vary | --occ (blank cams) | --render (dump frames) | --render_only (no policy server;
build+randomize+render+assert invariants only, for the §3.6 validity pre-check).

Run (env worker; policy served by base_eval_server.py / arm_eval_server.py on --port):
  PYTHONPATH=/root/LIBERO-PRO MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G CUDA_VISIBLE_DEVICES=$G LIBERO_CONFIG_PATH=/root/.libero \
    /root/libplusvenv/bin/python libero_goal_vary_eval.py --bddl_dir <goal_bddl> --init_dir <goal_init> \
      --n_trials N --port P [--vary] [--occ] [--shard k/M] [--render]
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import socket, struct, pickle, argparse, glob, re, json
import numpy as np
import torch
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle
from PIL import Image

# ---- validated manipuland zone (sim world xy, from the goal bddl regions + probed canonical live positions) ----
# canonical: bowl(-0.09,0.00) cream_cheese(-0.045,+0.135) wine_bottle(-0.20,-0.05) plate(+0.05,-0.02); jitter~0.02-0.03
# fixtures (avoid): cabinet(0.03,-0.24) stove(-0.41,0.21) wine_rack(-0.26,-0.26). Zone = manipuland bounding hull, padded.
ZONE_X = (-0.21, 0.04)     # manipuland band (canonical spans -0.20..+0.05); stays left of the cabinet body
ZONE_Y = (-0.05, 0.15)     # >= -0.05 clears the cabinet (anchor y=-0.24); <= 0.15 clears the stove (y=0.21)
Z_TABLE = 0.90              # table-top z of a settled free object (probed ~0.898-0.909); |z-Z_TABLE|<Z_TOL invariant
Z_TOL = 0.06


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
    conn.sendall(struct.pack(">I", len(pickle.dumps(obj, protocol=4))) + pickle.dumps(obj, protocol=4))


def make_state(o):
    return np.concatenate([o["robot0_eef_pos"], quat2axisangle(o["robot0_eef_quat"]),
                           o["robot0_gripper_qpos"]]).astype(np.float32)


def parse_lang(bddl_path):
    t = open(bddl_path).read()
    m = re.search(r"\(:language\s*(.*?)\)", t, re.S)
    return m.group(1).strip() if m else ""


def free_objs(env):
    """the movable manipulands = free joints '<obj>_joint0' (excludes fixtures/basket)."""
    names = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
    return [n[:-7] for n in names if n and n.endswith("_joint0") and "basket" not in n]


def randomize_goal(env, rng, dmin=0.06, dmax=0.16, sep=0.08, tries=400, layouts=8):
    """REUSABLE G1 GOAL-DECONF GENERATOR (self-validating). After a canonical set_init_state+settle, re-place every free
    manipuland to a new (x,y): dmin<=|new-canonical|<=dmax (dmin>>~0.025 jitter breaks a memorized reach; dmax keeps
    objects in the visible band so a GROUNDED motor can still grasp), pairwise sep>=sep, z+quaternion untouched. Settle,
    then REQUIRE every object on-table (assert_valid) — if fixture-penetration ejected one, restore canonical qpos and
    resample (up to `layouts` full attempts). Returns (obs_after_settle, moved) only for a VALID layout; else raises."""
    sim = env.sim
    objs = free_objs(env)
    addr = {o: sim.model.get_joint_qpos_addr(o + "_joint0")[0] for o in objs}
    canon_qpos = sim.data.qpos.copy(); canon_qvel = sim.data.qvel.copy()
    canon = {o: (float(canon_qpos[addr[o]]), float(canon_qpos[addr[o] + 1])) for o in objs}
    for _lay in range(layouts):
        sim.data.qpos[:] = canon_qpos; sim.data.qvel[:] = 0.0; sim.forward()   # restore clean canonical each attempt
        placed, newpos, ok = [], {}, True
        for o in objs:
            cx, cy = canon[o]
            for _ in range(tries):
                x = rng.uniform(*ZONE_X); y = rng.uniform(*ZONE_Y)
                d2 = (x - cx) ** 2 + (y - cy) ** 2
                if d2 < dmin ** 2 or d2 > dmax ** 2:
                    continue
                if all((x - px) ** 2 + (y - py) ** 2 > sep ** 2 for px, py in placed):
                    sim.data.qpos[addr[o]:addr[o] + 2] = [x, y]; placed.append((x, y)); newpos[o] = (x, y)
                    break
            else:
                ok = False; break
        if not ok:
            continue
        sim.forward()
        obs = None
        for _ in range(30):                  # settle the re-placed objects
            obs, _, _, _ = env.step(np.zeros(7))
        if assert_valid(obs, objs):          # a free object was ejected off-table -> discard this layout, resample
            continue
        moved = {o: (canon[o][0], canon[o][1], newpos[o][0], newpos[o][1],
                     float(np.hypot(newpos[o][0] - canon[o][0], newpos[o][1] - canon[o][1]))) for o in objs}
        return obs, moved
    raise RuntimeError("randomize_goal: no valid on-table layout after retries")


def assert_valid(obs, objs):
    """numeric invariant (§2 reduce-to-invariant): every free object stayed on the table after re-place+settle."""
    bad = []
    for o in objs:
        p = np.asarray(obs.get(o + "_pos", [9, 9, 9]), np.float32)
        onx = ZONE_X[0] - 0.06 <= p[0] <= ZONE_X[1] + 0.06
        ony = ZONE_Y[0] - 0.06 <= p[1] <= ZONE_Y[1] + 0.06
        onz = abs(p[2] - Z_TABLE) < Z_TOL
        if not (onx and ony and onz):
            bad.append((o, tuple(round(float(v), 3) for v in p[:3])))
    return bad


def task_files(bddl_dir, init_dir):
    out = []
    for b in sorted(glob.glob(os.path.join(bddl_dir, "*.bddl"))):
        stem = os.path.splitext(os.path.basename(b))[0]
        ip = os.path.join(init_dir, stem + ".pruned_init")
        if os.path.exists(ip):
            out.append((stem, b, ip))
    return out


# the 2 goals whose target is a FIXED FIXTURE (drawer/stove-knob) — position-variation of free objects can't test
# memorization for these (target never moves). Reported separately from the 8 free-manipuland goals.
ARTICULATED = {"open_the_middle_drawer_of_the_cabinet", "turn_on_the_stove"}


def render_only(args, tasks):
    """§3.6 validity pre-check: build+canonical+vary+render+assert, NO policy server."""
    rng = np.random.RandomState(args.seed)
    for stem, bddl, ip in tasks:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        inits = np.asarray(torch.load(ip, weights_only=False))
        env.reset(); obs = env.set_init_state(inits[0])
        for _ in range(20): obs, _, _, _ = env.step(np.zeros(7))
        canon_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        objs = free_objs(env)
        try:
            obs, moved = randomize_goal(env, rng, dmin=args.dmin, dmax=args.dmax, sep=args.sep)
        except RuntimeError as e:
            print(f"[{stem[:34]:34s}] RANDOMIZE_FAIL {e}", flush=True); env.close(); continue
        bad = assert_valid(obs, objs)
        vary_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        pair = np.concatenate([canon_img, vary_img], 1)
        Image.fromarray(pair.astype(np.uint8)).save(f"/root/goalvary_{stem[:26]}.png")
        dz = " ".join(f"{o.split('_')[0]}:{m[4]:.2f}" for o, m in moved.items())
        print(f"[{stem[:34]:34s}] {parse_lang(bddl)[:34]:34s} disp={{{dz}}} bad={bad} RENDERED", flush=True)
        env.close()
    print("RENDER_ONLY_DONE", flush=True)


def main(args):
    tasks = task_files(args.bddl_dir, args.init_dir)
    if args.only:
        tasks = [t for t in tasks if args.only in t[0]]
    if args.shard:
        k, M = [int(x) for x in args.shard.split("/")]
        tasks = [t for i, t in enumerate(tasks) if i % M == k]
    if args.render_only:
        render_only(args, tasks); return

    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(("127.0.0.1", args.port))
    send_msg(conn, {"cmd": "ping"}); assert recv_msg(conn).get("ok"), "server not ready"
    rng = np.random.RandomState(args.seed)
    tot_s = tot = 0
    man_s = man_t = art_s = art_t = 0            # manipuland-goal vs articulated-goal tallies
    for stem, bddl, ip in tasks:
        language = parse_lang(bddl)
        inits = np.asarray(torch.load(ip, weights_only=False))
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        objs = free_objs(env)
        s = 0; nbad = 0
        for j in range(args.n_trials):
            env.reset(); obs = env.set_init_state(inits[j % len(inits)])
            for _ in range(50): obs, _, _, _ = env.step(np.zeros(7))   # official settle (libero_pro_eval: 5 = under-settled)
            moved = None
            if args.vary:
                for _att in range(3):
                    try:
                        obs, moved = randomize_goal(env, rng, dmin=args.dmin, dmax=args.dmax, sep=args.sep); break
                    except RuntimeError:
                        env.reset(); obs = env.set_init_state(inits[j % len(inits)])
                        for _ in range(50): obs, _, _, _ = env.step(np.zeros(7))
                bad = assert_valid(obs, objs)
                if bad:
                    nbad += 1
                    print(f"  [INVALID-SKIP] {stem[:30]} tr{j} off-table={bad}", flush=True); continue
            send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
            frames = []; done = False
            for step in range(args.max_steps):
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wr = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                if args.occ:
                    img = np.zeros_like(img); wr = np.zeros_like(wr)
                if args.render and step % 20 == 0:
                    frames.append(img)
                send_msg(conn, {"cmd": "act", "blank": args.occ,
                                "obs": {"image_b": img.tobytes(), "image_shape": list(img.shape), "image_dtype": str(img.dtype),
                                        "wrist_b": wr.tobytes(), "wrist_shape": list(wr.shape), "wrist_dtype": str(wr.dtype),
                                        "instruction": language, "state": make_state(obs).tolist()}})
                action = np.asarray(recv_msg(conn)["action"], np.float32)
                obs, reward, done, info = env.step(action)
                if done:
                    break
            s += int(done); tot += 1; tot_s += int(done)
            if stem in ARTICULATED: art_t += 1; art_s += int(done)
            else: man_t += 1; man_s += int(done)
            if args.render and frames:
                dtag = "V" if args.vary else "C"
                Image.fromarray(np.concatenate(frames, 1).astype(np.uint8)).save(
                    f"/root/goalroll_{dtag}_{stem[:20]}_tr{j}_{'S' if done else 'F'}.png")
        mv = ""
        if moved is not None:
            mv = " disp=" + ",".join(f"{o.split('_')[0]}:{m[4]:.2f}" for o, m in moved.items())
        print(f"TASK {stem[:40]:40s} {s}/{args.n_trials} art={stem in ARTICULATED} badskip={nbad}{mv} :: {language}", flush=True)
        env.close()
    tag = ("OCC" if args.occ else ("VARY" if args.vary else "CANON"))
    print(f"[{tag}] libero_goal SR = {tot_s}/{tot} = {100*tot_s/max(tot,1):.1f}%  "
          f"(manipuland {man_s}/{man_t}={100*man_s/max(man_t,1):.1f}%  articulated {art_s}/{art_t}={100*art_s/max(art_t,1):.1f}%)", flush=True)
    print("LIBERO_GOAL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl_dir", default="/root/LIBERO-PRO/libero/libero/bddl_files/libero_goal")
    ap.add_argument("--init_dir", default="/root/LIBERO-PRO/libero/libero/init_files/libero_goal")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--port", type=int, default=5980)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dmin", type=float, default=0.06)   # min displacement from canonical (>> ~0.025 jitter)
    ap.add_argument("--dmax", type=float, default=0.13)   # max displacement (keep in visible central band, clear of cabinet)
    ap.add_argument("--sep", type=float, default=0.08)    # min inter-object separation
    ap.add_argument("--shard", default="")                # k/M
    ap.add_argument("--only", default="")                 # substring filter on task stem (e.g. cream_cheese)
    ap.add_argument("--vary", action="store_true")
    ap.add_argument("--occ", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render_only", action="store_true")
    main(ap.parse_args())
