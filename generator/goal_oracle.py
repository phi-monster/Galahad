"""goal_oracle.py — GT-xyz reach-grasp-place oracle that SOLVES varied-scene libero_goal "put X on/in Y" tasks by
reading LIVE object positions (⇒ handles displaced objects by construction). Displaces the free objects via
randomize_goal (from libero_goal_vary_eval), holds (:language) fixed → varied-scene goal-deconf demos. Only the 6
pick-place goals (On/In X Y). Saves ONLY successful episodes as npz (img/wrist/state/action/lang), [::-1,::-1] contract.

  --render_only : §3.6 de-risk — solve N varied scenes per task, render, report success + oracle SR (no npz).
  collect mode  : SHARD/N_PER_TASK/RAW envs, save successful varied demos.
Run: PYTHONPATH=/root/LIBERO-PRO MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G CUDA_VISIBLE_DEVICES=$G LIBERO_CONFIG_PATH=/root/.libero \
       /root/libvenv/bin/python goal_oracle.py [--render_only] [--tasks bowl_on_plate,...]
"""
import os, sys, re, glob, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
sys.path.insert(0, "/root")
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle
from PIL import Image
from libero_goal_vary_eval import randomize_goal, free_objs

BDDL = "/root/LIBERO-PRO/libero/libero/bddl_files/libero_goal"
INIT = "/root/LIBERO-PRO/libero/libero/init_files/libero_goal"
# libero_goal WORLD frame: table/objects at z~0.90, arm reaches ~1.18 (NOT the libero_object frame — collect_varied's
# 0.15/0.30 were below-table here and made the arm press-and-stall). Hover/lift are ABSOLUTE goal-frame z; descend/place
# offsets are object-RELATIVE.
HZ, GDZ, LZ = 1.05, 0.012, 1.15   # hover-z / grasp descend-offset (rel obj z, collect_varied-proven) / lift+carry-z


def parse_goal(bddl):
    t = open(bddl).read(); g = t[t.find("(:goal"):]
    m = re.search(r"\((?:On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", g)
    lang = re.search(r"\(:language\s*(.*?)\)", t, re.S)
    return m.group(1), m.group(2), (lang.group(1).strip() if lang else "")


def y_xyz(env, obs, yref):
    """world (x,y,z) of the destination Y: free-object live pos, else fixture site/body pos."""
    p = obs.get(yref + "_pos")
    if p is not None:
        return np.asarray(p, np.float32)
    sim = env.sim
    for getter in ("site", "body", "geom"):
        try:
            if getter == "site":
                return np.asarray(sim.data.get_site_xpos(yref), np.float32)
            if getter == "body":
                return np.asarray(sim.data.get_body_xpos(yref), np.float32)
            if getter == "geom":
                return np.asarray(sim.data.get_geom_xpos(yref), np.float32)
        except Exception:
            continue
    return None


def servo(eef, wp):
    return np.clip((np.asarray(wp, np.float32) - np.asarray(eef, np.float32)) / 0.05, -1, 1)


def mkstate(o):
    return np.concatenate([o["robot0_eef_pos"], quat2axisangle(o["robot0_eef_quat"]), o["robot0_gripper_qpos"]]).astype(np.float32)


def rollout(env, obs, X, Yref, rel, maxs=360, rec=False):
    ph, grip, hold = "hover", -1.0, 0
    goff = 0.10 if "wine_bottle" in X else GDZ   # tall bottle: grasp the BODY (base+0.10), not the base (blocks descent)
    im, wr, st, ac = [], [], [], []
    frames = []
    for s in range(maxs):
        eef = obs["robot0_eef_pos"]; xp = np.asarray(obs[X + "_pos"], np.float32)
        yp = y_xyz(env, obs, Yref)
        if yp is None:
            return False, None, frames
        place_z = yp[2] + (0.06 if rel == "On" else 0.03) + 0.02   # drop just above Y top
        if ph == "hover":
            wp = [xp[0], xp[1], HZ]; grip = -1
            if np.linalg.norm(eef[:2] - xp[:2]) < 0.015 and abs(eef[2] - HZ) < 0.03: ph = "descend"
        elif ph == "descend":
            gz = xp[2] + goff; wp = [xp[0], xp[1], gz]; grip = -1          # descend OPEN to grasp z (obj-aware)
            if eef[2] < gz + 0.008: ph = "grasp"; hold = 0                # only grasp once actually AT the object
        elif ph == "grasp":
            wp = [xp[0], xp[1], xp[2] + goff]; grip = 1; hold += 1         # hold at grasp z, close, wait for full close
            if hold >= 14: ph = "lift"
        elif ph == "lift":
            wp = [xp[0], xp[1], LZ]; grip = 1
            if eef[2] > LZ - 0.03: ph = "carry"
        elif ph == "carry":
            wp = [yp[0], yp[1], LZ]; grip = 1
            if np.linalg.norm(eef[:2] - yp[:2]) < 0.025: ph = "lower"
        elif ph == "lower":
            wp = [yp[0], yp[1], place_z]; grip = 1
            if eef[2] < place_z + 0.02: ph = "release"; hold = 0
        else:
            wp = [yp[0], yp[1], place_z]; grip = -1; hold += 1
        a = np.zeros(7, np.float32); a[:3] = servo(eef, wp); a[6] = grip
        if os.environ.get("TRACE") and s % 15 == 0:
            print(f"  s{s:3d} ph={ph:8s} eef=({eef[0]:+.3f},{eef[1]:+.3f},{eef[2]:.3f}) X=({xp[0]:+.3f},{xp[1]:+.3f},{xp[2]:.3f}) Y=({yp[0]:+.3f},{yp[1]:+.3f},{yp[2]:.3f}) grip={grip:+.0f}", flush=True)
        if rec:
            im.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            wr.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
            st.append(mkstate(obs)); ac.append(a.copy())
        if s % 20 == 0:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        obs, r, done, info = env.step(a)
        if done:
            return True, (im, wr, st, ac), frames
    return False, None, frames


PICKPLACE = ["put_the_bowl_on_the_plate", "put_the_cream_cheese_in_the_bowl", "put_the_bowl_on_the_stove",
             "put_the_wine_bottle_on_the_rack", "put_the_bowl_on_top_of_the_cabinet",
             "put_the_wine_bottle_on_top_of_the_cabinet"]


def main(a):
    tasks = a.tasks.split(",") if a.tasks else PICKPLACE
    rng = np.random.RandomState(a.seed)
    RAW = os.environ.get("RAW", "/root/goal_deconf_raw"); os.makedirs(RAW, exist_ok=True)
    SHARD = int(os.environ.get("SHARD", "0")); NPT = int(os.environ.get("N_PER_TASK", "3"))
    grand_s = grand_n = 0
    for stem in tasks:
        bddl = os.path.join(BDDL, stem + ".bddl"); ip = os.path.join(INIT, stem + ".pruned_init")
        X, Yref, lang = parse_goal(bddl)
        inits = np.asarray(torch.load(ip, weights_only=False))
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        got = tries = s_cnt = 0
        target = NPT if not a.render_only else a.n
        while got < target and tries < target * 4:
            tries += 1
            env.reset(); obs = env.set_init_state(inits[tries % len(inits)])
            for _ in range(50): obs, _, _, _ = env.step(np.zeros(7))
            try:
                obs, moved = randomize_goal(env, rng)
            except RuntimeError:
                continue
            ok, data, frames = rollout(env, obs, X, Yref, "On", rec=not a.render_only)
            grand_n += 1; s_cnt += int(ok)
            if a.render_only:
                if frames:
                    Image.fromarray(np.concatenate(frames[::max(1,len(frames)//8)], 1).astype(np.uint8)).save(
                        f"/root/oracle_{stem[:22]}_{tries}_{'S' if ok else 'F'}.png")
                got += 1
                print(f"ORACLE {stem[:30]:30s} try{tries} X={X} Y={Yref} {'SOLVED' if ok else 'FAIL'}", flush=True)
            else:
                if ok:
                    im, wr, st, ac = data
                    np.savez_compressed(f"{RAW}/ep_{SHARD:02d}_{stem[:16]}_{got:04d}.npz",
                                        img=np.asarray(im, np.uint8), wrist=np.asarray(wr, np.uint8),
                                        state=np.asarray(st, np.float32), action=np.asarray(ac, np.float32), lang=lang)
                    got += 1; grand_s += 1
        print(f"[{stem[:30]}] oracle_SR={s_cnt}/{tries} saved={got}", flush=True)
        env.close()
    print(f"GOAL_ORACLE_DONE total_saved={grand_s} solved={grand_s}/{grand_n} shard{SHARD}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_only", action="store_true")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
