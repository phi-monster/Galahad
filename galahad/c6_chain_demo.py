"""c6_chain_demo.py — THE FIGURE. One set of weights, one scene, one word changed — and BOTH faces move together.

This is the only picture that can carry the word "unified". The tables prove each face separately (the action face
obeys a new name 60/60 at 2 mm; the prediction face relocates its motion region 58.3% vs a 25% chance floor). The
figure proves they are the SAME model doing it AT THE SAME TIME, driven by the SAME word:

    row 1:  "Pick the <TRUE target>"  ->  [predicted region sits on the true target] | [the arm goes to the true target]
    row 2:  "Pick the <DECOY>"        ->  [predicted region MOVES to the decoy]      | [the arm goes to the decoy]

Same pixels at t=0 in both rows. Same weights. Nothing changes but the noun.

A frozen-pointer pipeline cannot produce this (its predictor is a separate model). A bolt-on video predictor
(V-JEPA2-AC) cannot produce this (the predictor is not the policy). A 3-model platform (Genie Envisioner) cannot
produce this (three sets of weights). That is the whole argument, in one image.

Run: CKPT=/root/jfs_local PORT=5000 python c6_chain_demo.py     (server must have GALAHAD_FS_CAPTURE=1)
"""
import os, sys, socket, argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/research/phi-arena/scripts/grounding_ladder")
os.environ.setdefault("MUJOCO_GL", "egl")
from libero.libero.envs import OffScreenRenderEnv                    # noqa: E402
from libero_pro_eval import make_state, send_msg, recv_msg, parse_bddl, task_list   # noqa: E402

G = 16
DS = 2
OUT = "/root/c6_fig"


def block_max(m, g):
    h = m.shape[0] // g
    return m.reshape(g, h, g, h).max(axis=(1, 3))


def obj_masks(env, obs):
    inst = np.asarray(obs["agentview_segmentation_instance"]).squeeze()[::-1, ::-1]
    elem = np.asarray(obs["agentview_segmentation_element"]).squeeze()[::-1, ::-1]
    sim = env.sim
    g2b = {g_: str(sim.model.body_id2name(sim.model.geom_bodyid[g_])) for g_ in range(sim.model.ngeom)}
    names = [n[:-7] for n in (sim.model.joint_id2name(i) for i in range(sim.model.njnt))
             if n and n.endswith("_joint0") and "basket" not in n]
    out = {}
    for nm in names:
        gids = [g_ for g_, b in g2b.items() if nm in b]
        px = np.isin(elem, gids) if gids else np.zeros_like(elem, bool)
        if px.sum() < 5:
            continue
        ids, cnt = np.unique(inst[px], return_counts=True)
        out[nm] = block_max((inst[::DS, ::DS] == int(ids[np.argmax(cnt)])).astype(np.float32), G)
    return out


def region_panel(img, region, gt):
    """the PREDICTION face: red contour = what the model says will move; green = the object the instruction named."""
    up = np.kron(1 / (1 + np.exp(-region)), np.ones((256 // G, 256 // G)))
    hot = up > np.quantile(up, 0.88)
    edge = hot ^ np.pad(hot, 1, mode="edge")[2:, 2:]
    gm = np.kron(gt, np.ones((256 // G, 256 // G))) > 0.5
    ge = gm ^ np.pad(gm, 1, mode="edge")[2:, 2:]
    v = img.astype(np.float32).copy()
    v[hot] = 0.72 * v[hot] + 0.28 * np.array([255, 0, 0])
    v[edge] = [255, 0, 0]
    v[ge] = [0, 255, 0]
    return np.clip(v, 0, 255).astype(np.uint8)


def run(conn, env, init_state, lang, masks, named, max_steps=280, every=35):
    """one instruction: capture the prediction at t=0, then roll the SAME model out and film where the arm goes."""
    env.reset()
    obs = env.set_init_state(np.asarray(init_state))
    for _ in range(50):
        obs, _, _, _ = env.step(np.zeros(7))
    img0 = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wr0 = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
    send_msg(conn, {"cmd": "act", "blank": False,
                    "obs": {"image_b": img0.tobytes(), "image_shape": list(img0.shape), "image_dtype": str(img0.dtype),
                            "wrist_b": wr0.tobytes(), "wrist_shape": list(wr0.shape), "wrist_dtype": str(wr0.dtype),
                            "instruction": lang, "state": make_state(obs).tolist(), "gt_pointer": [0.0, 0.0, 0.0]}})
    rep = recv_msg(conn)
    if "region" not in rep:
        raise SystemExit("server returned no region map — set GALAHAD_FS_CAPTURE=1 on the SERVER.")
    reg = np.asarray(rep["region"], np.float32).squeeze()   # server forwards (1,G,G) -> drop batch dim
    if reg.ndim == 1:
        reg = reg.reshape(G, G)
    pred = region_panel(img0, reg, masks[named])

    send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
    frames, mind = [img0], 1e9
    for step in range(max_steps):
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wr = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        send_msg(conn, {"cmd": "act", "blank": False,
                        "obs": {"image_b": img.tobytes(), "image_shape": list(img.shape), "image_dtype": str(img.dtype),
                                "wrist_b": wr.tobytes(), "wrist_shape": list(wr.shape), "wrist_dtype": str(wr.dtype),
                                "instruction": lang, "state": make_state(obs).tolist(), "gt_pointer": [0.0, 0.0, 0.0]}})
        obs, _, done, _ = env.step(np.asarray(recv_msg(conn)["action"], np.float32))
        e = np.asarray(obs["robot0_eef_pos"], np.float32)[:2]
        p = np.asarray(obs.get(named + "_pos", np.zeros(3)), np.float32)[:2]
        mind = min(mind, float(np.linalg.norm(e - p)))
        if step % every == 0 and len(frames) < 5:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        if done:
            break
    frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
    while len(frames) < 6:
        frames.append(frames[-1])
    return pred, np.concatenate(frames[:6], 1), mind


def main(a):
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(("127.0.0.1", a.port))
    send_msg(conn, {"cmd": "ping"}); assert recv_msg(conn).get("ok"), "server not ready"
    os.makedirs(OUT, exist_ok=True)
    for bp in task_list(a.bddl_dir, a.tasks):
        stem = os.path.splitext(os.path.basename(bp))[0]
        lang, tgt = parse_bddl(bp)
        ip = os.path.join(a.init_dir, stem + ".pruned_init")
        if not os.path.exists(ip):
            continue
        env = OffScreenRenderEnv(bddl_file_name=bp, camera_heights=256, camera_widths=256,
                                 camera_segmentations=["instance", "element"])
        init = torch.load(ip, weights_only=False)[a.trial]
        env.reset(); obs = env.set_init_state(np.asarray(init))
        for _ in range(50):
            obs, _, _, _ = env.step(np.zeros(7))
        masks = obj_masks(env, obs)
        if tgt not in masks:
            env.close(); continue
        pool = sorted([k for k in masks if k != tgt])
        dec = pool[a.decoy % len(pool)]
        ldec = f"Pick the {dec.rsplit('_', 1)[0].replace('_', ' ')} and place it in the basket"

        p1, f1, d1 = run(conn, env, init, lang, masks, tgt)          # the TRUE name
        p2, f2, d2 = run(conn, env, init, ldec, masks, dec)          # the DECOY name — IDENTICAL scene, one word changed
        env.close()

        row1 = np.concatenate([p1, f1], 1)
        row2 = np.concatenate([p2, f2], 1)
        w = min(row1.shape[1], row2.shape[1])
        fig = np.concatenate([row1[:, :w], np.zeros((6, w, 3), np.uint8), row2[:, :w]], 0)
        Image.fromarray(fig).save(f"{OUT}/{stem[:22]}.png")
        print(f"C6 {stem[:26]:<26} row1='{lang}' -> EEF got {d1*100:.1f}cm from {tgt}", flush=True)
        print(f"C6 {'':<26} row2='{ldec}' -> EEF got {d2*100:.1f}cm from {dec}", flush=True)
        print(f"   wrote {OUT}/{stem[:22]}.png   (left panel = the PREDICTION face; right strip = the ACTION face)", flush=True)
    print("C6_DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bddl_dir", default="/root/LIBERO-PRO/libero/libero/bddl_files/libero_object_task")
    p.add_argument("--init_dir", default="/root/LIBERO-PRO/libero/libero/init_files/libero_object")
    p.add_argument("--tasks", default="1,3,5")
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--decoy", type=int, default=0)
    p.add_argument("--port", type=int, default=5000)
    main(p.parse_args())
