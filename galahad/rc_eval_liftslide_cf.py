"""rc_eval_liftslide_cf.py (/root/rc_venv) — VERB axis (LIFT vs SLIDE) GROUNDING battery + 🔴 2x2 CONFUSION logging.
Superset of rc_eval_liftslide.py: identical env/gate/contract, PLUS per-episode JSONL of (told_verb x classified_motion)
so the FULL 2x2 confusion off-diagonal (the grounded-vs-null arbiter) can be computed. The VERB word is the ONLY cue
(SAME object+scene appears with BOTH verbs, role-balanced) => object CANNOT leak the verb.

Disjoint gate (no sink dependency): lift = end_h>H_HI ; slide = moved>=PUSH_MIN AND end_h<H_LO AND peak<H_PEAK AND on_counter.
  did_lift and did_slide are MUTUALLY EXCLUSIVE (lift needs end_h>0.12, slide needs end_h<0.06) => classified motion is
  exactly one of {lift, slide, noop(neither)}.

Modes: task (name a verb, balanced) / occ (blank cams + zero state, MUST collapse) / nonsense (garbage verb).
Per-episode JSONL (OUT_JSONL): {mode,told_verb,target,j,seed,end_h,moved,peak,on_counter,did_lift,did_slide,motion,succ}.
Sharding: run K workers with distinct --seed, each its own OUT_JSONL shard; pool + analyze_confusion.py.
Run: OUT_JSONL=/dev/shm/ls_cf/task_0.jsonl REACH_X=1.72,1.88 REACH_Y=-0.45,-0.28 \
     /root/rc_venv/bin/python rc_eval_liftslide_cf.py --mode task --n 16 --port 5710 --seed 8600
"""
import os, sys, socket, struct, pickle, argparse, math, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
sys.path.insert(0, "/root")
from collect_rc_deconf import randomize_objs, CAM_AGENT, CAM_WRIST, mkstate
from collect_rc_verb import make_counter_env, objpos
import robocasa.utils.object_utils as OU

EVIZ = "/dev/shm/ls_eval_viz"; os.makedirs(EVIZ, exist_ok=True)
POOL = os.environ.get("POOL", "apple,lemon,orange").split(",")
VERBS = ["lift", "slide"]
PUSH_MIN = float(os.environ.get("PUSH_MIN", "0.10")); H_HI = float(os.environ.get("H_HI", "0.12")); H_LO = float(os.environ.get("H_LO", "0.06"))
H_PEAK = float(os.environ.get("H_PEAK", "0.10"))   # slide must NEVER go high (peak guard) => a failed-lift (peak>0.12) FAILS slide
EXTRA_SETTLE = int(os.environ.get("EXTRA_SETTLE", "15"))
LANG = {"lift": "lift the %s", "slide": "slide the %s"}   # MATCHED: verb is the ONLY difference
OUT_JSONL = os.environ.get("OUT_JSONL", "")


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


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z = 1.96; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))


def did_lift(end_h):
    return bool(end_h > H_HI)                          # ended HELD-UP high


def did_slide(end_h, moved, on_counter, peak):
    return bool(moved >= PUSH_MIN and end_h < H_LO and peak < H_PEAK and on_counter)


def main(a):
    jf = open(OUT_JSONL, "a", buffering=1) if OUT_JSONL else None
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(("127.0.0.1", a.port))
    send_msg(conn, {"cmd": "ping"}); assert recv_msg(conn).get("ok"), "server not ready"
    tot_s = tot = 0; per_verb = {v: [0, 0] for v in VERBS}
    # confusion accumulators: cf[told][motion]  motion in {lift,slide,noop}
    cf = {"lift": {"lift": 0, "slide": 0, "noop": 0}, "slide": {"lift": 0, "slide": 0, "noop": 0}}
    for target_cat in POOL:
        env = make_counter_env(target_cat, a.seed + (hash(target_cat) % 1000))
        rng = np.random.RandomState(a.seed + (hash(target_cat) % 1000))
        for j in range(a.n):
            env.reset(); obs = randomize_objs(env, rng)
            for _ in range(EXTRA_SETTLE):
                obs, _, _, _ = env.step(np.zeros(env.action_dim))
            verb = VERBS[j % 2]
            instr = ("fribble the %s" % target_cat.replace("_", " ")) if a.mode == "nonsense" else (LANG[verb] % target_cat.replace("_", " "))
            send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
            tp0 = objpos(obs, "obj"); z0 = float(tp0[2]); obj_zmax = z0; frames = []
            for step in range(a.max_steps):
                img = np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1])
                wr = np.ascontiguousarray(np.asarray(obs[CAM_WRIST + "_image"])[::-1])
                if a.render and step % 10 == 0:
                    frames.append(img.copy())
                if a.mode == "occ":
                    img = np.zeros_like(img); wr = np.zeros_like(wr)
                st = (np.zeros(8, np.float32) if (a.mode == "occ" or os.environ.get("FORCE_ZERO_STATE") == "1") else mkstate(obs)).tolist()
                send_msg(conn, {"cmd": "act", "blank": a.mode == "occ",
                                "obs": {"image_b": img.tobytes(), "image_shape": list(img.shape), "image_dtype": str(img.dtype),
                                        "wrist_b": wr.tobytes(), "wrist_shape": list(wr.shape), "wrist_dtype": str(wr.dtype),
                                        "instruction": instr, "state": st}})
                act = np.asarray(recv_msg(conn)["action"], np.float32).ravel()
                ad = {"right": np.concatenate([act[:6], np.zeros(max(0, 6 - len(act)))])[:6],
                      "right_gripper": np.array([act[6] if len(act) > 6 else -1.0], np.float32)}
                obs, r, done, info = env.step(env.robots[0].create_action_vector(ad))
                obj_zmax = max(obj_zmax, float(objpos(obs, "obj")[2]))
            tp = objpos(obs, "obj"); end_h = float(tp[2] - z0); moved = float(np.linalg.norm(tp[:2] - tp0[:2]))
            on_counter = bool(tp[2] > z0 - 0.10); peak = float(obj_zmax - z0)
            dl = did_lift(end_h); ds = did_slide(end_h, moved, on_counter, peak)
            motion = "lift" if dl else ("slide" if ds else "noop")
            cf[verb][motion] += 1
            if a.mode == "nonsense":
                succ = bool(dl or ds)                    # any clean verb-motion (grounded => LOW)
            else:
                succ = dl if verb == "lift" else ds      # did the NAMED verb (task + un-fakeable obey)
            s = int(succ); tot_s += s; tot += 1
            if a.mode != "nonsense":
                per_verb[verb][0] += s; per_verb[verb][1] += 1
            if jf:
                jf.write(json.dumps({"mode": a.mode, "told_verb": verb, "target": target_cat, "j": j, "seed": a.seed,
                                     "end_h": round(end_h, 4), "moved": round(moved, 4), "peak": round(peak, 4),
                                     "on_counter": on_counter, "did_lift": dl, "did_slide": ds, "motion": motion,
                                     "succ": bool(succ)}) + "\n")
            if a.render and frames:
                imageio.imwrite("%s/%s_%s_%s_%d_%s.png" % (EVIZ, a.mode, verb, target_cat, j, motion),
                                np.concatenate(frames[:16], axis=1))
        env.close()
    lo = wilson(tot_s, tot)
    label = {"task": "TASK/OBEY(verb-follow)", "occ": "OCC", "nonsense": "NONSENSE(any-verb-motion)"}[a.mode]
    print("[%s] %d/%d = %.1f%% CI[%.1f,%.1f]" % (label, tot_s, tot, *lo), flush=True)
    for vb in VERBS:
        k, nn = per_verb[vb]
        if nn:
            print("  %-5s %d/%d = %.1f%%" % (vb, k, nn, 100 * k / nn), flush=True)
    # 2x2 CONFUSION (this shard only; pool shards with analyze_confusion.py for the definitive table)
    print("CONFUSION_SHARD " + json.dumps(cf), flush=True)
    for told in VERBS:
        n_told = sum(cf[told].values())
        if n_told:
            pl = wilson(cf[told]["lift"], n_told); psl = wilson(cf[told]["slide"], n_told)
            print("  told-%-5s n=%d : did_lift %d/%d=%.1f%%[%.1f,%.1f]  did_slide %d/%d=%.1f%%[%.1f,%.1f]  noop=%d" % (
                told, n_told, cf[told]["lift"], n_told, *pl, cf[told]["slide"], n_told, *psl, cf[told]["noop"]), flush=True)
    print("RC_EVAL_LS_DONE", flush=True)
    if jf:
        jf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="task", choices=["task", "occ", "nonsense"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--port", type=int, default=5710)
    ap.add_argument("--seed", type=int, default=8600)
    ap.add_argument("--render", action="store_true")
    main(ap.parse_args())
