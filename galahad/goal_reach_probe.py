"""goal_reach_probe.py — §3.6 MECHANISM disambiguator for a base_vary DROP. For each varied manipuland goal, track the
min EEF-xy distance to (a) the target manipuland's OLD canonical spot vs (b) its NEW (moved) live position. A MEMORIZED
policy reaches the OLD spot (open-loop to the remembered location -> grasps air); a GROUNDED policy reaches the NEW
object. Renders each rollout so the destination is eyeball-verifiable. Run against a FREE base_eval_server port.
  PYTHONPATH=/root/LIBERO-PRO MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G CUDA_VISIBLE_DEVICES=$G LIBERO_CONFIG_PATH=/root/.libero \
    /root/libplusvenv/bin/python goal_reach_probe.py --tasks 3,6,7 --n 3 --port 5980
"""
import os, sys, argparse, glob, re
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
sys.path.insert(0, "/root")
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from libero_goal_vary_eval import (randomize_goal, free_objs, assert_valid, make_state,
                                   send_msg, recv_msg, parse_lang, task_files)
import socket


def parse_target(bddl):
    t = open(bddl).read(); g = t[t.find("(:goal"):]
    m = re.search(r"\((?:In|On)\s+([A-Za-z0-9_]+)\s", g)
    return m.group(1) if m else None


def main(a):
    tasks = task_files("/root/LIBERO-PRO/libero/libero/bddl_files/libero_goal",
                       "/root/LIBERO-PRO/libero/libero/init_files/libero_goal")
    idx = [int(x) for x in a.tasks.split(",")]
    tasks = [tasks[i] for i in idx]
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(("127.0.0.1", a.port))
    send_msg(conn, {"cmd": "ping"}); assert recv_msg(conn).get("ok")
    rng = np.random.RandomState(a.seed)
    old_wins = new_wins = succ = 0
    for stem, bddl, ip in tasks:
        lang = parse_lang(bddl); tgt = parse_target(bddl)
        inits = np.asarray(torch.load(ip, weights_only=False))
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        objs = free_objs(env)
        for j in range(a.n):
            env.reset(); obs = env.set_init_state(inits[j % len(inits)])
            for _ in range(50): obs, _, _, _ = env.step(np.zeros(7))
            try:
                obs, moved = randomize_goal(env, rng)
            except RuntimeError:
                print(f"  {stem[:24]} tr{j} randomize-fail", flush=True); continue
            cx, cy = moved[tgt][0], moved[tgt][1]          # OLD canonical spot of the target manipuland
            send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
            frames = []; done = False; mn_old = mn_new = 1e9
            for step in range(a.max_steps):
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wr = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                if step % 20 == 0: frames.append(img)
                send_msg(conn, {"cmd": "act", "blank": False,
                                "obs": {"image_b": img.tobytes(), "image_shape": list(img.shape), "image_dtype": str(img.dtype),
                                        "wrist_b": wr.tobytes(), "wrist_shape": list(wr.shape), "wrist_dtype": str(wr.dtype),
                                        "instruction": lang, "state": make_state(obs).tolist()}})
                action = np.asarray(recv_msg(conn)["action"], np.float32)
                obs, reward, done, info = env.step(action)
                eef = np.asarray(obs["robot0_eef_pos"], np.float32)[:2]
                newp = np.asarray(obs.get(tgt + "_pos", np.zeros(3)), np.float32)[:2]   # LIVE moved pos
                mn_old = min(mn_old, float(np.hypot(eef[0] - cx, eef[1] - cy)))
                mn_new = min(mn_new, float(np.hypot(eef[0] - newp[0], eef[1] - newp[1])))
                if done: break
            verdict = "REACHED_OLD(memorize)" if mn_old < mn_new else "REACHED_NEW(ground)"
            old_wins += int(mn_old < mn_new); new_wins += int(mn_new <= mn_old); succ += int(done)
            Image.fromarray(np.concatenate(frames, 1).astype(np.uint8)).save(
                f"/root/mech_{stem[:18]}_tr{j}_{'S' if done else 'F'}.png")
            print(f"MECH {stem[:26]:26s} tr{j} tgt={tgt:18s} d_old={mn_old:.3f} d_new={mn_new:.3f} {verdict} done={done}"
                  f" disp={moved[tgt][4]:.2f} :: {lang}", flush=True)
        env.close()
    print(f"[MECH-SUMMARY] reached_OLD(memorize)={old_wins} reached_NEW(ground)={new_wins} success={succ}", flush=True)
    print("MECH_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="3,6,7")   # manipuland goals: bowl->plate, cream_cheese->bowl, wine->rack
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--port", type=int, default=5980)
    ap.add_argument("--seed", type=int, default=11)
    main(ap.parse_args())
