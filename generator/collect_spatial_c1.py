"""collect_spatial_c1.py (py3.8 sim venv) — B1 SPATIAL deconfounded collector + C1 foresight recording.

Extends the collect_spatial.py offset rim-pinch oracle to ALL 10 libero_spatial scenes (4 both-table + 6 with an
elevated bowl: cabinet-top / stove / cookies-box / ramekin). The drawer bowl (wooden_cabinet_1_top_region) is
DROPPED — recessed under the cabinet overhang, ungraspable top-down (hand stalls ~6cm above, closes on air;
verified across 3 offset variants 2026-07-15). Records the C1 foresight channels (img/wrist/state/action/depth/
seg/allpos/objnames/role/target_name/lang), pattern copied from collect_deconf_task_c1.py.

DECONFOUNDING (driven by /root/spatial_deconf.json): the two akita_black_bowls are IDENTICAL; per scene we collect
BOTH — (GRASP_OBJ=akita_black_bowl_1, lang=b1_phrase) AND (GRASP_OBJ=akita_black_bowl_2, lang=b2_phrase) — so the
dataset is role-balanced and the ONLY cue distinguishing the two targets is the benchmark's own SPATIAL phrase.
=> we DO NOT randomize object positions (unlike the C1 object task): the phrase is bound to the fixed spatial
layout; we iterate the benchmark's OWN init_states, which keep each bowl in its region across inits.

GRASP MECHANICS (measured + render-verified 2026-07-15, LAB spatial):
  Bowl dia 0.107 > gripper span 0.096 => centered top-down slides into the interior (no purchase) => OFFSET
  rim-pinch straddle (one finger inside a wall, one outside). Grasp z is RELATIVE to the bowl rest z, so an
  elevated bowl (cabinet-top +23cm) works with the SAME code.
    mode by BASE-DISTANCE (base@x=-0.66): base_dist>=FARDIST(0.78) => reach-limited FAR (toward-robot X offset,
      grasp at the reach floor via descend-timeout); else NEAR Y-straddle. (base-dist, NOT bowl-x>=0.06 — the old
      x rule mis-flagged elevated CENTRAL bowls (cookies/cabinet) as FAR and broke them.)
    PEDESTAL bowls (support==ramekin): the straddle nudges the bowl ~2cm on the narrow pedestal; a DEEPER grab
      (DESZ_PED=0.020, vs 0.038 flat) still catches the wall and lifts it clean. render-verified.
  Success = STRICT on_plate: bowl xy within 0.06 of plate AND lifted >=0.08 clear of rest AND resting at plate
  level (rejects the reach-margin "pushed along the surface" false-positive).

Run (fan-out across GPUs; ONE gpu each): SHARD=0 N_PER_BOWL=20 TID=0,3,5,7,9 OUT_DIR=/root/spatial_raw \\
  PYTHONPATH=/root/LIBERO-PRO MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \\
  /root/libvenv/bin/python collect_spatial_c1.py
Env: OUT_DIR(/root/spatial_raw) N_PER_BOWL(20) SHARD(0) TID(0..9 csv) INIT0(0) MAXTRY_MULT(6) DRY(0: skip savez,
     just report success table) VIZ(0: also dump a keyframe strip per (scene,bowl) for the §3.6 look).
"""
import os, re, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/root/research/phi-arena/scripts/grounding_ladder")
import numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle, quat2mat
import collect_deconf_task_c1 as C   # reuse servo()

DECONF = os.environ.get("DECONF", "/root/spatial_deconf.json")
OUT_DIR = os.environ.get("OUT_DIR", "/root/spatial_raw")
N_PER_BOWL = int(os.environ.get("N_PER_BOWL", "20"))
SHARD = int(os.environ.get("SHARD", "0"))
TID = [int(x) for x in os.environ.get("TID", "0,1,2,3,4,5,6,7,8,9").split(",") if x != ""]
INIT0 = int(os.environ.get("INIT0", "0"))
MAXTRY_MULT = int(os.environ.get("MAXTRY_MULT", "6"))
BOWLS = [b for b in os.environ.get("BOWLS", "1,2").split(",") if b]   # select which bowl(s) to run (e.g. BOWLS=2 = cabinet only)
DRY = os.environ.get("DRY", "0") == "1"
VIZ = os.environ.get("VIZ", "0") == "1"
FORCE = os.environ.get("FORCE_GRASP", "0") == "1"   # render/attempt DROPPED bowls (for §3.6 diagnosis + fix)
DS = int(os.environ.get("C1_DS", "2"))               # depth/seg stride-downsample 256->128
VIZDIR = "/root/spatial_smoke"; os.makedirs(VIZDIR, exist_ok=True)

# --- oracle params (defaults = render-verified) ---
HOVER = float(os.environ.get("HOVER", "0.14"))
DESZ_NEAR = float(os.environ.get("DESZ_NEAR", "0.038"))   # flat NEAR grab depth (rel to bowl rest z)
DESZ_FAR = float(os.environ.get("DESZ_FAR", "0.030"))
DESZ_PED = float(os.environ.get("DESZ_PED", "0.020"))     # ramekin pedestal: deeper -> catches despite the nudge
LIFT = float(os.environ.get("LIFT", "0.18"))
DROP = float(os.environ.get("DROP", "0.10"))
OFFY = float(os.environ.get("OFFY", "-0.044"))
OFFX_FAR = float(os.environ.get("OFFX_FAR", "-0.062"))
FARDIST = float(os.environ.get("FARDIST", "0.78"))        # horizontal dist from base >= this => reach-limited FAR
BASE = np.array([-0.66, 0.0])
HOLD_NEAR = int(os.environ.get("HOLD_NEAR", "18"))
HOLD_FAR = int(os.environ.get("HOLD_FAR", "30"))
DESC_TIMEOUT = int(os.environ.get("DESC_TIMEOUT", "110"))

# Per-hard-bowl overrides — the 3 reach-limited bowls the general oracle can't grasp (drawer / cabinet-in-drawer-
# scene / far-edge-table). Keyed (tid, bowl-str). The other 17 bowls hit HARD.get()->None and use the general path
# BYTE-IDENTICAL (no downgrade — verified tid3/7/9 cabinets at the SAME gz0 succeed unchanged). Values keep the arm
# inside the Panda reach envelope so the diff-IK doesn't fold the arm back (eef->-1.57). Tuned by render (§3.6).
HARD = {
    # drawer tid4-b1 (STANDARD LIBERO open-top drawer, NOT a wall — the earlier "geometric wall" misread the cabinet
    # body panels behind the open drawer as walls around the bowl). Root cause of the old 1.2%: a[3:6] orientation was
    # left 0 => the finger straddle stayed on world-Y and its outer finger hit the drawer tray/back wall, popping the
    # bowl during carry. FIX = gripper-YAW servo (a[5] closed-loop holds the finger straddle to world-X, off the walls)
    # + X-straddle offx=-0.03 + pin-down (desz<0 past the reach-floor) + safe_carry level transit + pull=0. n=12
    # render-verified real lift-out-of-drawer + place. (Y-straddle also works IF the yaw servo holds it steady: the
    # lever is the ACTIVE orientation-hold, not the specific angle; X-straddle offx=-0.03 was most efficient, 8/8 in 8.)
    (4, "1"): dict(yaw=0.0, offx=-0.03, offy=0.0, desz=-0.06, dto=200, freeze=1, hold=30, lifth=0.16, hov=0.12, hov_xy=0.04, safe_carry=1, pull=0.0),
    (4, "2"): dict(offy=0.044, lifth=0.15, hov_xy=0.03, safe_carry=1),   # cabinet-top (elevated, exposed): +y straddle + LEVEL carry (no diagonal down-swing) = 10/10 verified
    (6, "1"): dict(offx=-0.03, offy=0.044, desz=-0.06, hov=0.12, hov_xy=0.04, hold=30, lifth=0.16, freeze=1, dto=200),  # far-edge table (bdist~0.80): pin-down past the reach-floor + +y Y-straddle + freeze = 8/8 verified
}


def parse_ooi(b):
    m = re.search(r"\(:obj_of_interest\s*(.*?)\)", open(b).read(), re.S); return m.group(1).split() if m else []


def mkstate(o):
    return np.concatenate([o["robot0_eef_pos"], quat2axisangle(o["robot0_eef_quat"]),
                           o["robot0_gripper_qpos"]]).astype(np.float32)


def scene_objnames(obs):
    # WORLD positions of real free objects only: obs has both "{obj}_pos" and the RELATIVE "{obj}_to_robot0_eef_pos"
    # (both end in _pos) — keep only the world ones so allpos deprojects correctly for offline name->seg recovery.
    return sorted(k[:-4] for k in obs if k.endswith("_pos") and not k.startswith("robot0") and "_to_robot0_eef" not in k)


def rollout(env, obs, grasp, place, support, objnames, tid=-1, bowl="0", maxs=460):
    """Offset rim-pinch oracle + per-step C1 recording. Returns (on_plate, channels_tuple, keyframes)."""
    gz0 = float(np.asarray(obs[grasp + "_pos"])[2]); pz0 = float(np.asarray(obs[place + "_pos"])[2])
    bx0 = float(np.asarray(obs[grasp + "_pos"])[0]); by0 = float(np.asarray(obs[grasp + "_pos"])[1])
    bdist = float(np.hypot(bx0 - BASE[0], by0 - BASE[1]))
    far = bdist >= FARDIST
    pedestal = (support == "ramekin")
    offx, offy = (OFFX_FAR, 0.0) if far else (0.0, OFFY)
    desz = DESZ_FAR if far else (DESZ_PED if pedestal else DESZ_NEAR)
    hold_max = HOLD_FAR if far else HOLD_NEAR
    hov_xy, hov_z, car_tol = (0.02, 0.04, 0.025) if far else (0.015, 0.03, 0.02)
    hov_h, lift_h, pull = HOVER, LIFT, 0.0
    hb = HARD.get((tid, str(bowl)))     # 3 hard bowls only; None for the other 17 (general path byte-identical)
    _ov = os.environ.get("HB_OVERRIDE", "")   # per-GPU param sweep: '{"4,2":{"offy":0.044,...}}' merged OVER the HARD entry
    if _ov:
        _o = json.loads(_ov).get("%d,%s" % (tid, str(bowl)))
        if _o:
            hb = {**(hb or {}), **_o}
    if hb:
        offx = hb.get("offx", offx); offy = hb.get("offy", offy)
        hov_h = hb.get("hov", hov_h); lift_h = hb.get("lifth", lift_h); desz = hb.get("desz", desz)
        hov_xy = hb.get("hov_xy", hov_xy); hold_max = hb.get("hold", hold_max)
        hov_z = hb.get("hov_z", hov_z); pull = hb.get("pull", pull)
    car_tol = hb.get("car_tol", car_tol) if hb else car_tol
    safe_carry = bool(hb.get("safe_carry", 0)) if hb else False   # elevated bowl: level transit + straight-down place (no diagonal down-swing)
    place_drop = hb.get("placez", DROP) if hb else DROP
    freeze = bool(hb.get("freeze", 0)) if hb else False   # commit to a FIXED grasp xy at descend-start (stops chasing a nudged/receding bowl)
    dto = int(hb.get("dto", DESC_TIMEOUT)) if hb else DESC_TIMEOUT   # per-bowl descend timeout (pin-down bowls need longer to reach the reach-floor)
    hook = bool(hb.get("hook", 0)) if hb else False   # drawer 2-phase: PUSH the recessed bowl OUT to open table (no grasp), THEN grasp it as an exposed bowl
    hooky = hb.get("hooky", 0.06) if hb else 0.06     # push target y (toward the open +Y side / plate)
    hookx = hb.get("hookx", 0.0) if hb else 0.0       # push target x offset from bowl start
    hookz = hb.get("hookz", 0.02) if hb else 0.02     # push height above bowl rest (closed gripper shoves the bowl body)
    yaw = hb.get("yaw", None) if hb else None   # target finger-straddle angle (world-XY rad); None=no yaw (default Y-straddle). drawer: rotate straddle off the tray side-walls
    ph, grip, hold, rel, dstep, lstep, lowstep = ("hookdown" if hook else "hover"), -1.0, 0, 0, 0, 0, 0
    bx_start = None
    fgx = fgy = None
    maxz = gz0
    im, wr, st, ac, role, dep, seg, apos = [], [], [], [], [], [], [], []
    keyframes = {}; last_ph = None
    for s in range(maxs):
        eef = np.asarray(obs["robot0_eef_pos"]); tp = np.asarray(obs[grasp + "_pos"]); bp = np.asarray(obs[place + "_pos"])
        gx, gy = tp[0] + offx, tp[1] + offy
        if freeze and fgx is not None and ph in ("descend", "grasp", "extract", "lift"):
            gx, gy = fgx, fgy   # use the frozen straddle point (don't chase the receding bowl)
        if bx_start is None: bx_start = tp[0]
        if ph == "hookdown":     # descend CLOSED gripper into the bowl to push height
            wp = [tp[0], tp[1], gz0 + hookz]; grip = 1
            if eef[2] < gz0 + hookz + 0.03 and np.linalg.norm(eef[:2] - tp[:2]) < 0.04: ph = "hookpush"
        elif ph == "hookpush":   # shove the bowl to the open table position (bx_start+hookx, hooky)
            wp = [bx_start + hookx, hooky, gz0 + hookz]; grip = 1
            if tp[1] > hooky - 0.05 or (abs(tp[1] - hooky) < 0.05 and abs(tp[0] - (bx_start + hookx)) < 0.05): ph = "hookup"
        elif ph == "hookup":     # lift + open, clear of the relocated bowl, then normal grasp
            wp = [eef[0], eef[1], gz0 + 0.24]; grip = -1
            if eef[2] > gz0 + 0.20: ph = "hover"
        elif ph == "hover":
            wp = [gx, gy, gz0 + hov_h]; grip = -1
            if np.linalg.norm(eef[:2] - [gx, gy]) < hov_xy and abs(eef[2] - (gz0 + hov_h)) < hov_z:
                ph = "descend"; dstep = 0; fgx, fgy = gx, gy
        elif ph == "descend":
            wp = [gx, gy, gz0 + desz]; grip = -1; dstep += 1
            if eef[2] < gz0 + desz + 0.012 or dstep > dto:
                ph = "grasp"; hold = 0
        elif ph == "grasp":
            wp = [gx, gy, eef[2]]; grip = 1; hold += 1
            if hold >= hold_max: ph = "lift"
        elif ph == "lift":
            wp = [gx - pull, gy, gz0 + lift_h]; grip = 1; lstep += 1   # pull=0 for the 17 (byte-identical); >0 drags a reach-edge bowl toward base while lifting
            if eef[2] > gz0 + lift_h - 0.03 or (lstep > 40 and maxz >= gz0 + 0.09): ph = "carry"
        elif ph == "carry":
            off = tp[:2] - eef[:2]; tgt = bp[:2] - off
            cz = (gz0 + lift_h) if safe_carry else (pz0 + LIFT)   # elevated bowl: keep the lift height for a LEVEL lateral transit
            wp = [tgt[0], tgt[1], cz]; grip = 1
            if np.linalg.norm(eef[:2] - tgt) < car_tol: ph = ("lower" if safe_carry else "release")
        elif ph == "lower":   # elevated-bowl only: straight DOWN over the plate, still gripping, before release
            off = tp[:2] - eef[:2]; tgt = bp[:2] - off
            wp = [tgt[0], tgt[1], pz0 + place_drop]; grip = 1; lowstep += 1
            if eef[2] < pz0 + place_drop + 0.02 or lowstep > 80: ph = "release"
        else:
            wp = [bp[0], bp[1], pz0 + place_drop]; grip = -1; rel += 1
        a = np.zeros(7, np.float32); a[:3] = C.servo(eef, wp); a[6] = grip
        if yaw is not None:
            Rm = quat2mat(np.asarray(obs["robot0_eef_quat"], np.float64))
            sv = Rm[:, 1]   # gripper local-Y = finger-separation axis (verify col0 vs col1 via DBG_YAW)
            cur = float(np.arctan2(sv[1], sv[0]))
            err = ((float(yaw) - cur + np.pi / 2) % np.pi) - np.pi / 2   # straddle line is mod-pi
            if ph in ("hover", "descend", "grasp", "lift", "carry"):
                a[5] = float(np.clip(err * 2.5, -1, 1))
            if FORCE and s % 20 == 0:
                print("DBG_YAW s=%d ph=%s cur=%.3f tgt=%.3f col0=(%.2f,%.2f) col1=(%.2f,%.2f)" % (
                    s, ph, cur, float(yaw), Rm[0, 0], Rm[1, 0], Rm[0, 1], Rm[1, 1]), flush=True)
        # --- C1 recording (per step, SAME [::-1,::-1] flip as collect_deconf_task_c1) ---
        # DRY (success-table only): skip renders — the oracle reads OBJECT POSITIONS, not pixels, so the success
        # rate is render-independent; env is created minimal-camera in DRY. Only render when we'll save (or VIZ).
        if not DRY:
            im.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            wr.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
            d = np.asarray(obs["agentview_depth"]).squeeze()[::-1, ::-1][::DS, ::DS].astype(np.float16)
            g = np.asarray(obs["agentview_segmentation_instance"]).squeeze()[::-1, ::-1][::DS, ::DS].astype(np.uint8)
            dep.append(d); seg.append(g)
            st.append(mkstate(obs)); ac.append(a.copy()); role.append(np.asarray(tp, np.float32))
            apos.append(np.asarray([obs[o + "_pos"] for o in objnames], np.float32))
        if VIZ and ph != last_ph:
            keyframes[ph] = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy(); last_ph = ph
        maxz = max(maxz, float(tp[2]))
        obs, r, done, info = env.step(a)
        if rel >= 22: break
    tpf = np.asarray(obs[grasp + "_pos"]); bpf = np.asarray(obs[place + "_pos"])
    on_plate = (float(np.linalg.norm(tpf[:2] - bpf[:2])) < 0.06 and maxz >= gz0 + 0.08 and tpf[2] >= pz0 - 0.005)
    if FORCE:
        eff = np.asarray(obs["robot0_eef_pos"])
        print("DBG %s sup=%s bdist=%.3f far=%s ped=%s gz0=%.3f pz0=%.3f | eef=(%.3f,%.3f,%.3f) tpf=(%.3f,%.3f,%.3f) maxz=%.3f dxy_plate=%.3f lifted=%s onp=%s" % (
            grasp, support, bdist, far, pedestal, gz0, pz0, eff[0], eff[1], eff[2],
            tpf[0], tpf[1], tpf[2], maxz, float(np.linalg.norm(tpf[:2] - bpf[:2])), maxz >= gz0 + 0.08, on_plate), flush=True)
    if VIZ:
        keyframes["final"] = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy()
    return on_plate, (im, wr, st, ac, role, dep, seg, apos), keyframes


def main():
    deconf = {d["bddl"]: d for d in json.load(open(DECONF))}
    bd = benchmark.get_benchmark_dict()["libero_spatial"]()
    os.makedirs(OUT_DIR, exist_ok=True)
    table = []   # (tid, bowl, support, phrase, got, tries)
    n_ok = 0
    for tid in TID:
        t = bd.get_task(tid)
        if t.bddl_file not in deconf:
            print(f"[skip] tid{tid} {t.bddl_file} not in deconf map", flush=True); continue
        dc = deconf[t.bddl_file]
        tb = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        ooi = parse_ooi(tb); place = ooi[1]
        if DRY and not VIZ:   # success-table only: minimal camera (oracle is position-driven, cameras don't affect it)
            env = OffScreenRenderEnv(bddl_file_name=tb, camera_heights=84, camera_widths=84)
        else:
            env = OffScreenRenderEnv(bddl_file_name=tb, camera_heights=256, camera_widths=256,
                                     camera_depths=True, camera_segmentations="instance")
        inits = torch.load(os.path.join(get_libero_path("init_states"), t.problem_folder, t.init_states_file))
        for bowl in BOWLS:
            gr = "akita_black_bowl_" + bowl
            phrase = dc["b%s_phrase" % bowl]; support = dc["b%s_support" % bowl]; graspable = dc["b%s_graspable" % bowl]
            if not graspable and not FORCE:
                print(f"tid{tid} {gr} [{support}]: DROPPED (ungraspable), phrase={phrase!r}", flush=True)
                table.append((tid, bowl, support, phrase, -1, 0)); continue
            got, tries = 0, 0
            while got < N_PER_BOWL and tries < N_PER_BOWL * MAXTRY_MULT:
                env.reset()
                obs = env.set_init_state(np.asarray(inits[(INIT0 + tries) % len(inits)]))
                for _ in range(5): obs, _, _, _ = env.step(np.zeros(7))
                objnames = scene_objnames(obs)
                tries += 1
                ok, ch, kf = rollout(env, obs, gr, place, support, objnames, tid=tid, bowl=bowl)
                if VIZ and (got == 0 or (tries == 1)):
                    from PIL import Image
                    order = ["hover", "descend", "grasp", "lift", "carry", "lower", "release", "final"]
                    imgs = [kf[k] for k in order if k in kf]
                    Image.fromarray(np.concatenate(imgs, 1).astype(np.uint8)).save(
                        f"{VIZDIR}/coll_t{tid}_b{bowl}_{support}_{'S' if ok else 'F'}.png")
                if not ok:
                    continue
                im, wr, st, ac, role, dep, seg, apos = ch
                if not DRY:
                    np.savez_compressed(
                        OUT_DIR + "/ep_%02d_t%d_%s_%04d.npz" % (SHARD, tid, gr, got),
                        img=np.asarray(im, np.uint8), wrist=np.asarray(wr, np.uint8),
                        state=np.asarray(st, np.float32), action=np.asarray(ac, np.float32),
                        role=np.asarray(role, np.float32),
                        depth=np.asarray(dep, np.float16), seg=np.asarray(seg, np.uint8),
                        allpos=np.asarray(apos, np.float32), objnames=np.asarray(objnames),
                        target_name=gr, lang=phrase)
                got += 1; n_ok += 1
            print(f"tid{tid} {gr} [{support}]: {got}/{N_PER_BOWL} ({tries} tries) :: {phrase!r}", flush=True)
            table.append((tid, bowl, support, phrase, got, tries))
        env.close()
    print("\n=== SUCCESS TABLE (tid, bowl, support, got/N, tries) ===", flush=True)
    for tid, bowl, support, phrase, got, tries in table:
        rate = "DROP" if got < 0 else f"{got}/{N_PER_BOWL}({tries}t)"
        print(f"  tid{tid} b{bowl} {support:<8} {rate:<12} {phrase[9:]!r}", flush=True)
    print(f"SHARD{SHARD}_SPATIAL_C1_DONE total_saved={n_ok} dry={DRY}", flush=True)


if __name__ == "__main__":
    main()
