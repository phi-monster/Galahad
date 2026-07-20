"""collect_confounded_task_c1.py — the L3 spike's CONFOUNDED anchor data (the DISEASE / "original un-deconfounded").

Matched pair with the deconf collector: SAME 10 libero_object objects, SAME height-adaptive oracle motor (B27), SAME
foresight channels, SAME obs contract. The ONE difference is the CONFOUND:

  deconf (c1v2)    : randomize() puts every object at a uniform-random (x,y) AND every object is collected as target
                     (role-balanced) => scene/position ⟂ identity => the INSTRUCTION is the only cue => grounding forced.
  confounded (this): standard-LIBERO structure. For task `tid`, collect ONLY that task's CANONICAL target (the object
                     named in the original bddl), at the CANONICAL init positions (NO randomize). So scene-layout-tid
                     ALWAYS -> object-tid (scene→target correlation) and object-tid sits at its canonical spot
                     (position→identity). A BC policy learns layout→grasp and IGNORES the instruction (gradient
                     starvation / B47-B50 position-memorization) => at eval (libero_object_task perturbs the NAMED
                     object in the SAME canonical scene) it grasps the canonical object, NOT the newly-named one =>
                     SWAP fails, while TASK on the canonical target is fine (positions match eval) => CLEAN disease
                     anchor with clean attribution (SWAP-fail is grounding, not a position mismatch).

Why NOT arbitrary "home" positions: eval uses the CANONICAL libero_object init positions. Training the target at
off-canonical positions would make the confounded anchor fail TASK too (position mismatch), muddying attribution.
Canonical positions => TASK works, only SWAP fails => the failure is unambiguously grounding.

Verify with verify_confound.py: target xy -> identity classifier >> chance; per-identity target-position std small.
Run (matched to deconf): SHARD=0 N_PER_OBJ=30 TASKS=0 RAW=/dev/shm/dct_conf_raw PYTHONPATH=/dev/shm/LIBERO \
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 /root/libvenv/bin/python collect_confounded_task_c1.py
"""
import os, numpy as np, torch, re
os.environ.setdefault("MUJOCO_GL", "egl")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

SHARD = int(os.environ.get("SHARD", "0"))
N_PER_OBJ = int(os.environ.get("N_PER_OBJ", "30"))     # episodes of THIS task's canonical target on this shard
TASKS = [int(x) for x in os.environ.get("TASKS", "0,1,2,3,4,5,6,7,8,9").split(",")]
RAW = os.environ.get("RAW", "/dev/shm/dct_conf_raw")
SEED = int(os.environ.get("SEED", str(1000 + SHARD)))
DS = int(os.environ.get("C1_DS", "2"))
HZ, GDZ, LZ = 0.15, 0.012, 0.32
TALL_Z = float(os.environ.get("TALL_Z", "0.045"))
NECK_GDZ = float(os.environ.get("NECK_GDZ", "0.045"))
CLEAR_H = float(os.environ.get("CLEAR_H", "0.12"))
os.makedirs(RAW, exist_ok=True)


def readable(obj):
    return re.sub(r"_\d+$", "", obj).replace("_", " ")


def base_name(o):
    return re.sub(r"_\d+$", "", str(o))


def scene_objs(env):
    sim = env.sim
    names = [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]
    return [n[:-7] for n in names if n and n.endswith("_joint0") and "basket" not in n]


def canonical_target_name(bddl_path):
    """The object this task is about, from 'pick_up_the_<X>_and_place_it_in_the_basket.bddl'."""
    b = os.path.basename(bddl_path)
    m = re.search(r"pick_up_the_(.+?)_and_place", b)
    return m.group(1) if m else None


def servo(eef, wp):
    return np.clip((np.asarray(wp, np.float32) - np.asarray(eef, np.float32)) / 0.05, -1, 1)


def mkstate(o):
    return np.concatenate([o["robot0_eef_pos"], quat2axisangle(o["robot0_eef_quat"]), o["robot0_gripper_qpos"]]).astype(np.float32)


def allobj_pos(obs, objs):
    return np.asarray([obs[o + "_pos"] for o in objs], np.float32)


def rollout(env, obs, target, objs, basket="basket_1", maxs=340):
    ph, grip, hold, rel = "hover", -1.0, 0, 0
    im, wr, st, ac, tpos, dep, seg, apos = [], [], [], [], [], [], [], []
    z0 = float(np.asarray(obs[target + "_pos"])[2])
    tall = z0 > TALL_Z
    hz = max(HZ, z0 + CLEAR_H) if tall else HZ
    gdz = NECK_GDZ if tall else GDZ
    for s in range(maxs):
        eef, tp, bp = obs["robot0_eef_pos"], obs[target + "_pos"], obs[basket + "_pos"]
        if ph == "hover":
            wp = [tp[0], tp[1], hz]; grip = -1
            if np.linalg.norm(eef[:2] - tp[:2]) < 0.015 and abs(eef[2] - hz) < 0.03: ph = "descend"
        elif ph == "descend":
            gz = tp[2] + gdz; wp = [tp[0], tp[1], gz]; grip = -1
            if eef[2] < gz + 0.02: ph = "grasp"; hold = 0
        elif ph == "grasp":
            gz = tp[2] + gdz; wp = [tp[0], tp[1], gz]; grip = 1; hold += 1
            if hold >= 8: ph = "lift"
        elif ph == "lift":
            wp = [tp[0], tp[1], LZ]; grip = 1
            if eef[2] > LZ - 0.03: ph = "carry"
        elif ph == "carry":
            wp = [bp[0], bp[1], LZ]; grip = 1
            if np.linalg.norm(eef[:2] - bp[:2]) < 0.03: ph = "release"
        else:
            wp = [bp[0], bp[1], 0.22]; grip = -1; rel += 1
        a = np.zeros(7, np.float32); a[:3] = servo(eef, wp); a[6] = grip
        im.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        wr.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
        d = np.asarray(obs["agentview_depth"]).squeeze()[::-1, ::-1][::DS, ::DS].astype(np.float16)
        g = np.asarray(obs["agentview_segmentation_instance"]).squeeze()[::-1, ::-1][::DS, ::DS].astype(np.uint8)
        dep.append(d); seg.append(g)
        st.append(mkstate(obs)); ac.append(a.copy()); tpos.append(np.asarray(tp, np.float32))
        apos.append(allobj_pos(obs, objs))
        obs, r, done, info = env.step(a)
        if rel >= 25:
            break
    tpf, bpf = np.asarray(obs[target + "_pos"]), np.asarray(obs[basket + "_pos"])
    in_basket = float(np.linalg.norm(tpf[:2] - bpf[:2])) < 0.075
    return in_basket, (im, wr, st, ac, tpos, dep, seg, apos)


def main():
    bd = benchmark.get_benchmark_dict()["libero_object"]()
    n_ok = 0
    for tid in TASKS:
        t = bd.get_task(tid)
        tb = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=tb, camera_heights=256, camera_widths=256,
                                 camera_depths=True, camera_segmentations="instance")
        inits = torch.load(os.path.join(get_libero_path("init_states"), t.problem_folder, t.init_states_file))
        objs = scene_objs(env)
        cbase = canonical_target_name(tb)                       # e.g. "alphabet_soup"
        target = next((o for o in objs if base_name(o) == cbase), None)
        if target is None:
            print(f"task{tid} NO_CANONICAL_TARGET base={cbase} objs={objs}", flush=True); env.close(); continue
        lang = "pick up the " + readable(target) + " and place it in the basket"
        got, tries = 0, 0
        while got < N_PER_OBJ and tries < N_PER_OBJ * 6:
            tries += 1
            env.reset()
            obs = env.set_init_state(np.asarray(inits[tries % len(inits)]))
            for _ in range(13):                                 # settle at CANONICAL positions (no randomize)
                obs, _, _, _ = env.step(np.zeros(7))
            ok, data = rollout(env, obs, target, objs)
            if not ok:
                continue
            im, wr, st, ac, tpos, dep, seg, apos = data
            np.savez_compressed(RAW + "/ep_%02d_%d_%s_%04d.npz" % (SHARD, tid, target, got),
                                img=np.asarray(im, np.uint8), wrist=np.asarray(wr, np.uint8),
                                state=np.asarray(st, np.float32), action=np.asarray(ac, np.float32),
                                role=np.asarray(tpos, np.float32),
                                depth=np.asarray(dep, np.float16), seg=np.asarray(seg, np.uint8),
                                allpos=np.asarray(apos, np.float32), objnames=np.asarray(objs),
                                target_name=target, lang=lang)
            got += 1; n_ok += 1
        print("task%d canonical=%s: %d/%d (%d)" % (tid, target, got, N_PER_OBJ, tries), flush=True)
        env.close()
    print("SHARD%d_CONF_DONE total=%d" % (SHARD, n_ok), flush=True)


if __name__ == "__main__":
    main()
