"""collect_deconf_task_c1.py (py3.8 sim venv) — C1 UPGRADE of collect_deconf_task.py.
IDENTICAL deconfounded oracle (motor 87%, untouched) + records the FORESIGHT channels the unification (C2/C3/C4) needs:
  depth (agentview, 128x128 float16, flipped to match img) + instance-seg (128x128 uint8, flipped) per frame,
  target_name, and all-object positions per frame (allpos/objnames) so name->seg-id is recoverable offline via depth-deproject
  (convention-robust — no P-matrix flip). Future-frame(t+k) + motion-region GT are DERIVED at convert time from the frame seq.
Depth/seg render + flip-alignment VISUALLY validated 2026-07-14 (c1_probe: seg/depth regions sit ON objects in the flipped av).
Run: SHARD=0 N_PER_OBJ=20 TASKS=0,1 RAW=/root/dct_c1_raw PYTHONPATH=/root/LIBERO MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 /root/libvenv/bin/python collect_deconf_task_c1.py
"""
import os, numpy as np, torch, re
os.environ.setdefault("MUJOCO_GL", "egl")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

SHARD = int(os.environ.get("SHARD", "0"))
N_PER_OBJ = int(os.environ.get("N_PER_OBJ", "20"))
TASKS = [int(x) for x in os.environ.get("TASKS", "0,1,2,3,4,5,6,7,8,9").split(",")]
RAW = os.environ.get("RAW", "/root/dct_c1_raw")
SEED = int(os.environ.get("SEED", str(1000 + SHARD)))
DS = int(os.environ.get("C1_DS", "2"))          # downsample stride for depth/seg -> 128x128 (foresight targets are coarse)
HZ, GDZ, LZ = 0.15, 0.012, 0.32
DROP = set(os.environ.get("DROP", "").split(",")) - {""}   # A3(b): EMPTY by default — the round bottles are now graspable
# A3(d) NEVER-SEEN arm: an object listed here is BANISHED FROM THE SCENE (not merely dropped as a target) — it is
# teleported far out of the camera's view, so it appears NEITHER as a target NOR as a distractor. Train on this and the
# eval's slot for that object becomes a true out-of-vocabulary test (the model has literally never seen it).
EXCLUDE = set(os.environ.get("EXCLUDE", "").split(",")) - {""}
# --- A3(b) HEIGHT-ADAPTIVE GRASP (LAB B27; measured + render-verified 2026-07-14) ---------------------------------
# The old fixed hover (0.15) BULLDOZED the tall bottles: ketchup/salad_dressing stand with their tops AT ~0.15, so the
# horizontal approach rammed them (obj shoved 0.109 m, knocked over, then the oracle chased the fallen object and closed
# on AIR: grip span 0.0414 -> 0.0010). And even with a clear hover, a top-down grasp at the BODY centre cannot descend —
# the bottle itself blocks the jaws (span never closes). Both are fixed by: hover ABOVE the object, grasp at its NECK.
#   tall (rest z > 0.045): hover = max(0.15, z0 + 0.12) ; grasp = z + 0.045   -> ketchup 2/2, salad 2/2 (were 0/2)
#   short(rest z <= 0.045): unchanged 0.15 / +0.012                            -> alphabet_soup FAILS 0/2 with the neck grasp
# Safe for the tall boxes too (milk/oj/bbq 2/2, no regression). Render-verified: bottle stays upright, jaws close on the
# neck, it lifts clean off the table (NOT a scoop / knock-over / empty close).
TALL_Z = float(os.environ.get("TALL_Z", "0.045"))
NECK_GDZ = float(os.environ.get("NECK_GDZ", "0.045"))
CLEAR_H = float(os.environ.get("CLEAR_H", "0.12"))
os.makedirs(RAW, exist_ok=True)


def readable(obj):
    return re.sub(r"_\d+$", "", obj).replace("_", " ")


def scene_objs(env):
    sim = env.sim
    names = [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]
    objs = [n[:-7] for n in names if n and n.endswith("_joint0") and "basket" not in n]
    gone = DROP | EXCLUDE
    return [o for o in objs if re.sub(r"_\d+$", "", o) not in gone and o not in gone]


def randomize(env, rng):
    sim = env.sim
    names = [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]
    objs = [n[:-7] for n in names if n and n.endswith("_joint0") and "basket" not in n]
    addr = {o: sim.model.get_joint_qpos_addr(o + "_joint0")[0] for o in objs}
    placed = []
    for o in objs:
        if re.sub(r"_\d+$", "", o) in EXCLUDE or o in EXCLUDE:
            sim.data.qpos[addr[o]:addr[o] + 3] = [2.5, 2.5, 0.5]      # banish: far outside the agentview frustum
            continue
        for _ in range(200):
            x, y = rng.uniform(-0.18, 0.13), rng.uniform(-0.28, 0.08)
            if all((x - px) ** 2 + (y - py) ** 2 > 0.075 ** 2 for px, py in placed):
                break
        sim.data.qpos[addr[o]:addr[o] + 2] = [x, y]; placed.append((x, y))
    sim.forward()
    obs = None
    for _ in range(8):
        obs, _, _, _ = env.step(np.zeros(7))
    return obs


def servo(eef, wp):
    return np.clip((np.asarray(wp, np.float32) - np.asarray(eef, np.float32)) / 0.05, -1, 1)


def mkstate(o):
    return np.concatenate([o["robot0_eef_pos"], quat2axisangle(o["robot0_eef_quat"]), o["robot0_gripper_qpos"]]).astype(np.float32)


def allobj_pos(obs, objs):
    return np.asarray([obs[o + "_pos"] for o in objs], np.float32)   # (n_obj,3)


def rollout(env, obs, target, objs, basket="basket_1", maxs=340):
    ph, grip, hold, rel = "hover", -1.0, 0, 0
    im, wr, st, ac, tpos, dep, seg, apos = [], [], [], [], [], [], [], []
    z0 = float(np.asarray(obs[target + "_pos"])[2])          # rest height => tall (bottle) vs short (can/box)
    tall = z0 > TALL_Z
    hz = max(HZ, z0 + CLEAR_H) if tall else HZ               # clear the object before the horizontal approach
    gdz = NECK_GDZ if tall else GDZ                          # grab the NECK of a tall bottle, not its blocked body
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
        # C1 foresight channels — SAME [::-1,::-1] flip as img (validated aligned), then stride-downsample to 128
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
    rng = np.random.RandomState(SEED)
    n_ok = 0
    for tid in TASKS:
        t = bd.get_task(tid)
        tb = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=tb, camera_heights=256, camera_widths=256,
                                 camera_depths=True, camera_segmentations="instance")   # C1: request depth + instance seg
        inits = torch.load(os.path.join(get_libero_path("init_states"), t.problem_folder, t.init_states_file))
        objs = scene_objs(env)
        for target in objs:
            lang = "pick up the " + readable(target) + " and place it in the basket"
            got, tries = 0, 0
            while got < N_PER_OBJ and tries < N_PER_OBJ * 5:
                tries += 1
                env.reset(); obs = env.set_init_state(np.asarray(inits[tries % len(inits)]))
                for _ in range(5): obs, _, _, _ = env.step(np.zeros(7))
                obs = randomize(env, rng)
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
            print("task%d %s: %d/%d (%d)" % (tid, target, got, N_PER_OBJ, tries), flush=True)
        env.close()
    print("SHARD%d_C1_DONE total=%d" % (SHARD, n_ok), flush=True)


if __name__ == "__main__":
    main()
