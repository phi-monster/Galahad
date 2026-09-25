"""libero_pro_eval.py — DECOUPLE eval on the REAL LIBERO-PRO benchmark (arxiv 2510.03827).

Same validated contract as libero_eval.py (180deg agentview flip, state=[eef_pos, quat2axisangle, gripper_qpos],
7-DoF OSC_POSE action, set_init_state + 5 settle steps, done=success), but reads PERTURBED bddl+init from explicit
LIBERO-PRO directories instead of a base benchmark suite. Instruction + target-object are parsed from EACH perturbed
BDDL (so TASK perturbation feeds the NEW instruction + points at the NEW target).

Perturbation axes (this driver):
  position: --bddl_dir .../libero_object_temp_x0.2  --init_dir .../init_files/libero_object_temp_x0.2
            (shipped; objects translated, SAME instruction — tests the position-pointer's designed strength)
  task    : --bddl_dir .../libero_object_task        --init_dir .../init_files/libero_object   (base init reused —
            same scene geometry, TaskPerturbator changed language+goal+obj_of_interest → the target object CHANGED,
            the real language-grounding test where the field scores ~0)

Pointer for GT_PTR (B1) = the (perturbed) target object's live _pos (parsed from the perturbed :goal).
Pointer for NATIVE_PTR (b2n) = frozen Qwen3-VL reads the (perturbed) instruction → bbox → ray-plane world pt.

  yes N | PYTHONPATH=/root/LIBERO-PRO MUJOCO_GL=egl /root/libvenv/bin/python libero_pro_eval.py \
      --bddl_dir <dir> --init_dir <dir> --tasks <csv> --n_trials N --port P
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import socket, fcntl, struct, pickle, argparse, glob, re
import numpy as np
import torch
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle
from PIL import Image


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


def make_state(o):
    return np.concatenate([o["robot0_eef_pos"], quat2axisangle(o["robot0_eef_quat"]),
                           o["robot0_gripper_qpos"]]).astype(np.float32)


# Image flip fed to the POLICY. Default "vh" = [::-1,::-1] (MolmoAct2 / openvla-oft convention, LAB #7e).
# pi05_libero (HuggingFaceVLA/libero) trains on RAW camera images (official lerobot/envs/libero.py feeds raw;
# the [::-1,::-1] there is render/viz-only) -> use PI_FLIP=raw for pi05.  v=[::-1], h=[:,::-1].
_PI_FLIP = os.environ.get("PI_FLIP", "vh")
def _flip_img(a):
    if _PI_FLIP == "vh":
        return a[::-1, ::-1]
    if _PI_FLIP == "v":
        return a[::-1]
    if _PI_FLIP == "h":
        return a[:, ::-1]
    return a  # raw


def parse_bddl(bddl_path):
    """Return (language, target_obj) from a (possibly perturbed) BDDL."""
    t = open(bddl_path).read()
    lang = re.search(r"\(:language\s*(.*?)\)", t, re.S)
    language = lang.group(1).strip() if lang else ""
    g = t[t.find("(:goal"):]
    m = re.search(r"\((?:In|On)\s+([A-Za-z0-9_]+)\s", g)  # (In cream_cheese_1 basket_...) -> cream_cheese_1
    target = m.group(1) if m else "akita_black_bowl_1"
    return language, target


# exclude round bottles (oracle 0% grasp, motor not trained on them) — the 8-task set used for B1/B2/B2n
DROP = {"salad_dressing", "ketchup"}


def task_list(bddl_dir, tasks_arg):
    files = sorted(glob.glob(os.path.join(bddl_dir, "*.bddl")))
    if tasks_arg:  # explicit indices into the sorted list
        idx = [int(x) for x in tasks_arg.split(",")]
        files = [files[i] for i in idx]
    else:  # all, minus the round bottles
        files = [f for f in files if not any(d in os.path.basename(f) for d in DROP)]
    return files


def main(args):
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(("127.0.0.1", args.port))
    send_msg(conn, {"cmd": "ping"}); assert recv_msg(conn).get("ok"), "server not ready"
    bddls = task_list(args.bddl_dir, args.tasks)
    _swaplangs = [parse_bddl(b)[0] for b in bddls]  # --swap_instr source
    # --spatial_swap (B1 grounding): the two akita_black_bowls are IDENTICAL, distinguished ONLY by the spatial phrase.
    # Feed the ALTERNATE bowl's phrase in the SAME fixed scene and read (via PROBE) whether the EEF goes to the
    # NEWLY-NAMED bowl (OBEYED) vs the memorised/canonical one. Direct analog of SWAP2/OBEYED_NAME (LAB B32) for spatial.
    _deconf = {}
    if args.spatial_swap:
        import json
        _dp = os.environ.get("SPATIAL_DECONF", "/root/spatial_deconf.json")
        _deconf = {d["bddl"]: d for d in json.load(open(_dp))}
    tot_s = tot = 0
    per = []
    for _bi, bddl_path in enumerate(bddls):
        stem = os.path.splitext(os.path.basename(bddl_path))[0]
        language, target_obj = parse_bddl(bddl_path)
        if args.swap_instr and len(_swaplangs) > 1:
            language = _swaplangs[(_bi + 1) % len(_swaplangs)]  # WRONG instruction; target_obj (scoring) stays TRUE
        if args.nonsense:
            language = "xxx"                                    # MEANINGLESS instruction; target_obj (scoring) stays TRUE
        init_path = os.path.join(args.init_dir, stem + ".pruned_init")
        if not os.path.exists(init_path):
            print(f"[SKIP] no init for {stem}: {init_path}", flush=True); continue
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
        init_states = torch.load(init_path, weights_only=False)
        non_target = []
        _PROBE = bool(os.environ.get('PROBE')) or args.spatial_swap
        if args.wrong_obj or _PROBE or args.swap_scene or args.spatial_swap:
            _jn = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
            non_target = [n[:-7] for n in _jn if n and n.endswith("_joint0") and "basket" not in n and n[:-7] != target_obj]
        decoy = None
        _dpool = sorted(non_target) if args.swap_scene else []
        if args.spatial_swap:
            _dc = _deconf.get(os.path.basename(bddl_path))
            if _dc is None:
                print(f"[SKIP] spatial_swap: {os.path.basename(bddl_path)} not in deconf", flush=True); env.close(); continue
            _tnum = "2" if str(target_obj).endswith("_1") else "1"     # name the OTHER identical bowl
            decoy = "akita_black_bowl_" + _tnum
            language = _dc["b%s_phrase" % _tnum]                       # the decoy bowl's OWN spatial phrase (exact training phrasing)
            if decoy not in non_target:
                non_target.append(decoy)                               # ensure PROBE tracks the named bowl's distance
        if args.swap_scene and not _dpool:
            print(f"[SKIP] swap_scene: no in-scene decoy for {stem}", flush=True); env.close(); continue
        s = 0
        n_obey = 0
        for j in range(args.n_trials):
            if args.swap_scene:
                # ROTATE so every in-scene non-target gets named across the trials (5 decoys, 6 trials = all 5).
                decoy = _dpool[(_bi + j) % len(_dpool)]
                _dn = re.sub(r"_[0-9]+$", "", decoy).replace("_", " ")   # `alphabet_soup_1` -> `alphabet soup`
                language = f"Pick the {_dn} and place it in the basket"  # EXACT training phrasing (no trailing id)
            env.reset()
            obs = env.set_init_state(np.asarray(init_states[j % len(init_states)]))
            for _ in range(50):                         # official settle (was 5 = under-settled)
                obs, _, _, _ = env.step(np.zeros(7))
            send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
            frames = []; done = False
            _mn_t = 1e9; _mn_d = {o: 1e9 for o in non_target}
            for step in range(args.max_steps):
                img = np.ascontiguousarray(_flip_img(obs["agentview_image"]))          # PI_FLIP env (vh default; raw for pi05)
                wr = np.ascontiguousarray(_flip_img(obs["robot0_eye_in_hand_image"]))
                if args.occ:
                    img = np.zeros_like(img); wr = np.zeros_like(wr)
                if args.render and step % 20 == 0:
                    frames.append(img)
                if args.wrong_obj and non_target:
                    _tp = np.asarray(obs.get(target_obj + "_pos", np.zeros(3)), np.float32)
                    _w = max(non_target, key=lambda o: float(np.linalg.norm(np.asarray(obs.get(o + "_pos", np.zeros(3)), np.float32)[:2] - _tp[:2])))
                    ptr_val = np.asarray(obs.get(_w + "_pos", np.zeros(3)), np.float32)
                else:
                    ptr_val = np.asarray(obs.get(target_obj + "_pos", np.zeros(3)), np.float32)  # (perturbed) target's LIVE pos
                send_msg(conn, {"cmd": "act", "blank": args.occ,
                                "obs": {"image_b": img.tobytes(), "image_shape": list(img.shape), "image_dtype": str(img.dtype),
                                        "wrist_b": wr.tobytes(), "wrist_shape": list(wr.shape), "wrist_dtype": str(wr.dtype),
                                        "instruction": language, "state": make_state(obs).tolist(),
                                        "gt_pointer": ptr_val.tolist()}})
                action = np.asarray(recv_msg(conn)["action"], np.float32)
                if os.environ.get("DBG_ACT") and step < 8:
                    print("ACT step%d %s" % (step, np.round(action, 4).tolist()), flush=True)
                obs, reward, done, info = env.step(action)
                if _PROBE:
                    _e = np.asarray(obs['robot0_eef_pos'], np.float32)[:2]
                    _mn_t = min(_mn_t, float(np.linalg.norm(_e - np.asarray(obs.get(target_obj+'_pos', np.zeros(3)), np.float32)[:2])))
                    for _o in non_target:
                        _mn_d[_o] = min(_mn_d[_o], float(np.linalg.norm(_e - np.asarray(obs.get(_o+'_pos', np.zeros(3)), np.float32)[:2])))
                if done:
                    break
            _eef = np.asarray(obs["robot0_eef_pos"], np.float32)[:2]
            _tp = np.asarray(obs.get(target_obj + "_pos", np.zeros(3)), np.float32)[:2]
            print("REACH to_ptr=%.3f to_tgt=%.3f" % (float(np.linalg.norm(_eef - np.asarray(ptr_val, np.float32)[:2])), float(np.linalg.norm(_eef - _tp))), flush=True)
            if _PROBE and non_target:
                _nd = min(_mn_d, key=_mn_d.get); _ndv = _mn_d[_nd]
                _v = "REACHED_TGT" if _mn_t < _ndv else "WENT_DISTRACTOR"
                print("PROBE target=%s min_tgt=%.3f nearest=%s min_dist=%.3f %s" % (target_obj, _mn_t, _nd, _ndv, _v), flush=True)
            if (args.swap_scene or args.spatial_swap) and decoy is not None:
                _dd = _mn_d.get(decoy, 1e9)
                _obey = _dd < _mn_t and _dd < 0.05          # went to the NAMED object, not the memorised one
                n_obey += int(_obey)
                print("SWAP2 named=%s d_named=%.3f d_true=%.3f %s" % (decoy, _dd, _mn_t,
                      "OBEYED_NAME" if _obey else "IGNORED_NAME"), flush=True)
            s += int(done); tot += 1; tot_s += int(done)
            if args.render and frames:
                Image.fromarray(np.concatenate(frames, 1).astype(np.uint8)).save(f"/root/pro_leval_{stem[:20]}_tr{j}_{'S' if done else 'F'}.png")
        per.append((stem, s, args.n_trials, target_obj, language))
        _ob = f" OBEY={n_obey}/{args.n_trials} decoys={len(_dpool)}" if args.swap_scene else (
              f" OBEY={n_obey}/{args.n_trials} named=akita_black_bowl_{'2' if str(target_obj).endswith('_1') else '1'}" if args.spatial_swap else "")
        print(f"TASK {stem} {s}/{args.n_trials} target={target_obj}{_ob} :: {language}", flush=True)
        env.close()
    tag = "OCC" if args.occ else ("WOBJ" if args.wrong_obj else ("SPATIAL_SWAP" if args.spatial_swap else ("SWAP2" if args.swap_scene else ("SWAP" if args.swap_instr else ("NONSENSE" if args.nonsense else "EVAL")))))
    print(f"[{tag}] LIBERO-PRO({os.path.basename(args.bddl_dir)}) SR = {tot_s}/{tot} = {100*tot_s/max(tot,1):.1f}%", flush=True)
    print("LIBERO_EVAL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl_dir", required=True)
    ap.add_argument("--init_dir", required=True)
    ap.add_argument("--tasks", default="")                 # "" = all (minus round bottles); else indices into sorted bddl list
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=280)  # LIBERO-PRO TASK_MAX_STEPS libero_object = 280
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--occ", action="store_true")
    ap.add_argument("--wrong_obj", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--swap_scene", action="store_true")   # WELL-POSED swap: decoy is IN the scene; scores TRUE target (~0) AND reports whether the EEF obeyed the new name (needs PROBE=1)
    ap.add_argument("--swap_instr", action="store_true")  # feed a WRONG-object instruction, score TRUE target (Arm A grounding control)
    ap.add_argument("--nonsense", action="store_true")    # feed a MEANINGLESS instruction ("xxx"), score TRUE target (BATTERY face: kills the paraphrase smoke-bomb; a language-driven policy MUST collapse)
    ap.add_argument("--spatial_swap", action="store_true")  # B1 GROUNDING: name the OTHER identical bowl (its b1/b2 spatial phrase, from $SPATIAL_DECONF); PROBE reads OBEYED_NAME (EEF->named bowl) vs the memorised one. Scores TRUE (canonical) target too (stays low if grounded).
    main(ap.parse_args())
