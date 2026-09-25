"""collect_liftslide.py (/root/rc_venv) — RoboCasa VERB axis (LIFT vs SLIDE) DECONFOUNDED oracle collector.
Galahad G1 type #8, graspable family. The instruction's VERB selects the ACTION (lift-through-air vs drag-on-surface),
NOT the object. Same object + scene appears with BOTH verbs (role-balanced) => object CANNOT leak the verb.

Placement: CLEAN FLAT COUNTER right of the sink (REACH_X~[1.72,1.88], REACH_Y~[-0.45,-0.28]) => reliable grasp (oracle
lift/slide 18/18 each, disjoint gate, noop 0/0) + object x/y VARIES => vision required => OCC will collapse.
Oracle (both verbs share the proven grasp): hover->descend->grasp; then verb picks lift(raise+hold) vs drag(low+release).
Success (disjoint gate, no sink dependency): lift = end_h>0.12 ; slide = moved>=PUSH_MIN AND end_h<0.06 AND on_counter.
Records img/wrist/state(8)/action(7=dpos3+0,0,0+grip) + lang => npz (img,wrist,state,action,lang,verb,obj_cat,family).

Run one shard/GPU:
  SHARD=0 N_PER=32 POOL=apple,lemon,orange RAW=/dev/shm/verb_ls_raw REACH_X=1.72,1.88 REACH_Y=-0.45,-0.28 \\
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 /root/rc_venv/bin/python collect_liftslide.py
"""
import os, sys, json, traceback
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
sys.path.insert(0, "/root")
from collect_rc_deconf import (CAM_AGENT, CAM_WRIST, servo, step_arm, sink_xy, HOVER, GRASP_DZ, LIFT_Z,
                               GRIP_CLOSE, GRIP_OPEN, randomize_objs, mkstate)
from collect_rc_verb import make_counter_env, objpos, gripper_far_obj
import robocasa.utils.object_utils as OU
from robosuite.utils.transform_utils import quat2mat

POOL = os.environ.get("POOL", "apple,lemon,orange").split(",")
N_PER = int(os.environ.get("N_PER", "32"))          # successes per (object, verb); total = N_PER*len(POOL)*2
RAW = os.environ.get("RAW", "/dev/shm/verb_ls_raw"); os.makedirs(RAW, exist_ok=True)
SHARD = int(os.environ.get("SHARD", "0"))
SEED = int(os.environ.get("SEED", str(7300 + SHARD * 1000)))
MAXS = int(os.environ.get("MAXS", "480"))
CAMW_PIX = int(os.environ.get("CAMW_PIX", "256"))
DRAG_DIST = float(os.environ.get("DRAG_DIST", "0.14"))
HOLDN = int(os.environ.get("HOLDN", "14")); EXTRA_SETTLE = int(os.environ.get("EXTRA_SETTLE", "15"))
PUSH_MIN = float(os.environ.get("PUSH_MIN", "0.10")); H_HI = float(os.environ.get("H_HI", "0.12")); H_LO = float(os.environ.get("H_LO", "0.06"))
H_PEAK = float(os.environ.get("H_PEAK", "0.10"))   # slide demos must stay LOW throughout (peak guard) => train/eval-consistent + a failed-lift can't score slide
RENDER = os.environ.get("RENDER", "0") == "1"
VIZ = "/dev/shm/verb_ls_viz"; os.makedirs(VIZ, exist_ok=True)
# MATCHED instructions: the VERB is the ONLY linguistic difference (like the 7 axes) — no "across the counter" suffix
# that a model could key on instead of the verb lemma (red-team RANK 4).
LANG = {"lift": "lift the %s", "slide": "slide the %s"}


def rollout(env, obs, verb, record=True):
    robot = env.robots[0]; tp0 = objpos(obs, "obj"); z0 = float(tp0[2]); obj_zmax = z0
    sxy = sink_xy(env); base_R = quat2mat(np.asarray(obs["robot0_base_quat"], np.float64))
    dirx = 1.0 if tp0[0] >= sxy[0] else -1.0; d = np.array([dirx, 0.0], np.float32)
    ph, grip, hold, rel, dstep = "hover", GRIP_OPEN, 0, 0, 0; grasp_z = z0
    im, wr, st, ac, frames = [], [], [], [], []
    for s in range(MAXS):
        eef = np.asarray(obs["robot0_eef_pos"], np.float32); tp = objpos(obs, "obj")
        obj_zmax = max(obj_zmax, float(tp[2])); moved = float(np.linalg.norm(tp[:2] - tp0[:2]))
        if ph == "hover":
            wp = [tp[0], tp[1], tp[2] + HOVER]; grip = GRIP_OPEN
            if np.linalg.norm(eef[:2] - tp[:2]) < 0.02 and abs(eef[2] - (tp[2] + HOVER)) < 0.03: ph = "descend"
        elif ph == "descend":
            gz = tp[2] + GRASP_DZ; wp = [tp[0], tp[1], gz]; grip = GRIP_OPEN
            if eef[2] < gz + 0.02: ph = "grasp"; hold = 0
        elif ph == "grasp":
            wp = [tp[0], tp[1], eef[2]]; grip = GRIP_CLOSE; hold += 1
            if hold >= HOLDN: ph = ("lift" if verb == "lift" else "drag"); grasp_z = min(float(eef[2]), z0 + 0.005)
        elif ph == "lift":
            wp = [tp0[0], tp0[1], LIFT_Z]; grip = GRIP_CLOSE
            if eef[2] > LIFT_Z - 0.04: ph = "holdup"; hold = 0
        elif ph == "holdup":
            wp = [tp0[0], tp0[1], LIFT_Z]; grip = GRIP_CLOSE; hold += 1
            if hold >= 20: ph = "done"
        elif ph == "drag":
            dstep += 1; tgt = tp0[:2] + d * DRAG_DIST; wp = [tgt[0], tgt[1], grasp_z]; grip = GRIP_CLOSE
            if moved >= DRAG_DIST * 0.9 or dstep > 120: ph = "release"; rel = 0
        elif ph == "release":
            wp = [eef[0], eef[1], eef[2]]; grip = GRIP_OPEN; rel += 1
            if rel >= 8: ph = "done"
        else:
            wp = [eef[0], eef[1], eef[2]]
        dpos = servo(eef, wp, base_R)
        if record:
            im.append(np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1]))
            wr.append(np.ascontiguousarray(np.asarray(obs[CAM_WRIST + "_image"])[::-1]))
            st.append(mkstate(obs)); ac.append(np.concatenate([dpos, np.zeros(3, np.float32), [grip]]).astype(np.float32))
        if RENDER and s % 10 == 0: frames.append(np.asarray(obs[CAM_AGENT + "_image"])[::-1])
        obs, _, _, _ = step_arm(env, robot, dpos, grip)
        if ph == "done" and s > 30: break
    tp = objpos(obs, "obj"); end_h = float(tp[2] - z0); moved = float(np.linalg.norm(tp[:2] - tp0[:2]))
    on_counter = bool(tp[2] > z0 - 0.10)
    peak = float(obj_zmax - z0)
    if verb == "lift":
        succ = bool(end_h > H_HI)
    else:
        succ = bool(moved >= PUSH_MIN and end_h < H_LO and peak < H_PEAK and on_counter)
    return succ, (im, wr, st, ac), frames, dict(end_h=round(end_h, 3), moved=round(moved, 3), peak=round(peak, 3))


def main():
    rng_global = np.random.RandomState(SEED)
    got = 0; per_verb = {"lift": 0, "slide": 0}
    for cat in POOL:
        env = make_counter_env(cat, SEED + hash(cat) % 1000)
        for verb in ["lift", "slide"]:
            ok, tries = 0, 0
            while ok < N_PER and tries < N_PER * 6:
                tries += 1
                env.reset()
                obs = randomize_objs(env, rng_global)
                for _ in range(EXTRA_SETTLE):
                    obs, _, _, _ = env.step(np.zeros(env.action_dim))
                try:
                    succ, data, frames, sig = rollout(env, obs, verb, record=True)
                except Exception as e:
                    print("ROLL_ERR", verb, cat, e, flush=True); traceback.print_exc(); continue
                if RENDER and frames:
                    imageio.imwrite("%s/%s_%s_%d_%s.png" % (VIZ, verb, cat, tries, "S" if succ else "F"),
                                    np.concatenate([f for f in frames[:14]], axis=1))
                if not succ:
                    continue
                im, wri, stt, act = data
                np.savez_compressed("%s/ep_%d_%s_%s_%04d.npz" % (RAW, SHARD, verb, cat, ok),
                                    img=np.asarray(im, np.uint8), wrist=np.asarray(wri, np.uint8),
                                    state=np.asarray(stt, np.float32), action=np.asarray(act, np.float32),
                                    lang=LANG[verb] % cat.replace("_", " "), verb=verb, obj_cat=cat, family="graspable")
                ok += 1; got += 1; per_verb[verb] += 1
            print("  %s/%s: %d/%d ok (%d tries)" % (cat, verb, ok, N_PER, tries), flush=True)
        env.close()
    print("SHARD%d_DONE total=%d per_verb=%s" % (SHARD, got, json.dumps(per_verb)), flush=True)


if __name__ == "__main__":
    main()
