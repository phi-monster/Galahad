"""rc_eval_ordcol.py (/root/rc_venv) — ORDINAL x COLOUR compositional ZERO-SHOT grounding eval (E2 / LAB B62).

NEVER-SEEN composition test. The G1 universal LoRA was trained on 6 referent types SEPARATELY (colour, category,
ordinal, relation, negation, compositional=colour x CATEGORY) — NEVER on ordinal x colour, and NEVER on the
"filter-by-colour then count-in-the-subset" structure. This eval asks: does it zero-shot "the second red object from
the left", or fall back to a single-type heuristic (pure-ordinal / pure-colour)?

COMPOSITION = collect_ordinal.place_row (validated L->R row, ROW_AXIS=x ROW_SIGN=1 REF=left) + collect_rc_colour.recolour
(flat vivid hue-separated rgba). SINGLE category (all cans => shape constant, only colour+position vary). K objects,
2 colours (cf x2 + contrast x2). Target = the m-th cf-coloured object in L->R order.

THREE READINGS separated by construction (scene arranged so all 3 land on DIFFERENT objects):
  COMP  = the m-th cf object                (compositional-correct: filter by colour THEN count)
  ORD   = the m-th object overall            (pure-ordinal heuristic: ignore colour, just count)
  COL   = the 1st (leftmost) cf object       (pure-colour heuristic: ignore ordinal, grab a cf)
A scored instruction is "separating" iff COMP is distinct from BOTH ORD and COL (recorded per episode).

METRIC (rc_eval's existing SWAP-OBEY argmin): hand_key = argmin over the K row objects of per-object min EEF distance.
Per episode classify hand_key -> COMP / ORD / COL / OTHER. Headline = COMP-rate over separating episodes vs the two
heuristic anchors (ORD-rate, COL-rate) vs chance (1/K). Counterfactual TRIPLE per scene: "2nd cf" / "1st cf" /
"2nd contrast" (3 DIFFERENT correct objects) tests hand-tracking.

Modes:
  gate  : NO server. RENDER-GATE (§3.6). Build scenes, recolour, annotate pos:colour at each projected pixel, dump the
          agentview frame + an oracle filmstrip driven to COMP. scp + LOOK: colours eye-distinguishable at 256px?
          separation holds? gripper home L/centre of row (so nearest-cf == leftmost-cf == COL, not COMP)?
  swap  : compositional SWAP-OBEY + 4-way tally (headline). Runs the counterfactual triple per scene.
  occ   : identical but BOTH cameras blanked -> MUST collapse COMP to ~chance (else not using pixels => claim void).
  para  : PARAPHRASE control. PURE ordinal SWAP (multi-cat pool, NO recolour) with a REWORDED ordinal instruction.
          If the universal obeys the paraphrase (~B60 ordinal SWAP 81%) a composition null is "composition" not "phrasing".

Socket contract BYTE-IDENTICAL to rc_eval_ordinal.py (ping/reset/act; obs=image_b+wrist_b+instruction+state[8]; action[7]).
Launch env MUST match ordinal production (uwl_type.sh ordinal): ROW_AXIS=x ROW_SIGN=1 REF=left REACH_X=0.95,1.65
REACH_Y=-0.60,-0.57 SPACE_MIN=0.10 SPACE_MAX=0.14 JITTER=0.008 SINK_XY_TOL=0.15 SINK_Z_MAX=0.95 USE_OFFICIAL_SUCC=1.
"""
import os, sys, socket, struct, pickle, argparse, math, json, itertools
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio.v2 as imageio
sys.path.insert(0, "/root")
import collect_ordinal as CO
from collect_ordinal import (OrdinalPnP, place_row, rollout, target_in_sink, sink_xy,
                             objpos, mkstate, CAM_AGENT, CAM_WRIST, ORDW, REF)
from robosuite.controllers import load_composite_controller_config

EVIZ = os.environ.get("EVIZ", "/dev/shm/ordcol_eval_viz"); os.makedirs(EVIZ, exist_ok=True)
MIN_LIFT = float(os.environ.get("MIN_LIFT", "0.08"))

# vivid, hue-separated, high-saturation (verbatim from collect_rc_colour.py — the render-gate-validated palette)
ALL_COLOURS = {
    "red":    [0.85, 0.08, 0.08, 1.0],
    "green":  [0.10, 0.65, 0.12, 1.0],
    "blue":   [0.10, 0.22, 0.88, 1.0],
    "yellow": [0.90, 0.80, 0.10, 1.0],
}


# ---------- socket helpers (byte-identical to rc_eval_ordinal.py) ----------
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
    if hdr is None:
        raise ConnectionError("server closed connection")   # was: return None -> made per-step crash catchable
    body = _recv_all(conn, struct.unpack(">I", hdr)[0])
    if body is None:
        raise ConnectionError("server closed connection mid-message")
    return pickle.loads(body)


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(data)) + data)


CONN_ERRS = (ConnectionError, BrokenPipeError, ConnectionResetError, OSError, EOFError)


def rconnect(port, wait=600):
    """(Re)connect to the policy server, retrying while the supervisor restarts it. Robustness: a server
    death (rare — B60 ran n=96 clean) then costs one reconnect wait, not the whole sweep."""
    import time
    t0 = time.time()
    while time.time() - t0 < wait:
        c = None
        try:
            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            c.settimeout(120)
            c.connect(("127.0.0.1", port))
            send_msg(c, {"cmd": "ping"})
            if recv_msg(c).get("ok"):
                c.settimeout(None)
                return c
        except Exception:
            try:
                if c is not None:
                    c.close()
            except Exception:
                pass
            time.sleep(4)
    raise ConnectionError("could not reconnect to port %d within %ds" % (port, wait))


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))


def eef_pos_obs(obs):
    return np.asarray(obs["robot0_eef_pos"], np.float32)


# ---------- recolour (verbatim from collect_rc_colour.py; env-class-agnostic, keyed by object name) ----------
def _obj_geom_ids(env, name):
    obj = env.objects[name]
    ids = set(); names = []
    for attr in ("visual_geoms", "contact_geoms"):
        v = getattr(obj, attr, None)
        if v:
            names += list(v)
    for gn in names:
        try:
            ids.add(env.sim.model.geom_name2id(gn))
        except Exception:
            pass
    if ids:
        return ids
    try:
        bid = env.sim.model.body_name2id(obj.root_body)
    except Exception:
        return ids
    for gid in range(env.sim.model.ngeom):
        b = int(env.sim.model.geom_bodyid[gid])
        while b > 0:
            if b == bid:
                ids.add(gid); break
            b = int(env.sim.model.body_parentid[b])
    return ids


def recolour(env, slot_colour):
    for name, cname in slot_colour.items():
        rgba = np.asarray(ALL_COLOURS[cname], np.float32)
        for gid in _obj_geom_ids(env, name):
            env.sim.model.geom_matid[gid] = -1
            env.sim.model.geom_rgba[gid] = rgba
    env.sim.forward()


def make_env(pool, layout, style, camw, seed):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return OrdinalPnP(pool=pool, robots="PandaOmron", controller_configs=cc,
                      camera_names=[CAM_AGENT, CAM_WRIST], camera_widths=camw, camera_heights=camw,
                      has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True,
                      ignore_done=True, seed=seed, layout_ids=layout, style_ids=style,
                      translucent_robot=False, obj_instance_split=None, generative_textures=None)


# ---------- ordinal x colour scene templates (K=4, m=2) ----------
# P = 0-indexed L->R positions that get the cf (filter) colour; the rest get the contrast colour.
# Separation for the PRIMARY "2nd cf" instruction (m=2): COMP=lr[P[1]], ORD=lr[1], COL=lr[P[0]] all distinct
# <=> P[1]!=1 and P[0]!=1. Valid K=4 templates: (0,2)->[cf,cc,cf,cc]; (0,3)->[cf,cc,cc,cf]; (2,3)->[cc,cc,cf,cf].
P_TEMPLATES_K4 = [(0, 2), (0, 3), (2, 3)]
COLOUR_POOL = ["red", "green", "blue"]


def ordcol_scene(rng, K):
    """Pick a template + a (cf, contrast) colour pair. Returns (P, cf, cc)."""
    P = P_TEMPLATES_K4[rng.randint(len(P_TEMPLATES_K4))]
    two = list(rng.choice(COLOUR_POOL, size=2, replace=False))
    cf, cc = two[0], two[1]
    return P, cf, cc


def assign_ordcol(lr, P, cf, cc):
    """lr = L->R object keys. Positions in P get cf, the rest get cc. Returns slot_colour dict."""
    Pset = set(P)
    return {lr[p]: (cf if p in Pset else cc) for p in range(len(lr))}


def nth_of_colour(lr, slot_colour, colour, m):
    """the m-th (1-indexed) object of `colour` in L->R order; None if fewer than m present."""
    matches = [k for k in lr if slot_colour[k] == colour]
    return matches[m - 1] if len(matches) >= m else None


def anchors_for(lr, slot_colour, colour, m):
    """COMP=m-th `colour`; ORD=m-th overall; COL=1st `colour`. Returns dict of keys (or None)."""
    return {
        "comp": nth_of_colour(lr, slot_colour, colour, m),
        "ord":  lr[m - 1] if len(lr) >= m else None,
        "col":  nth_of_colour(lr, slot_colour, colour, 1),
    }


def nearest_of_colour(obs, lr, slot_colour, colour, eef):
    """the `colour` object closest to the EEF home (the target a 'grab nearest {colour}' heuristic would pick)."""
    ks = [k for k in lr if slot_colour[k] == colour]
    return min(ks, key=lambda k: float(np.linalg.norm(eef - objpos(obs, k)))) if ks else None


def classify(hand_key, anc):
    if hand_key == anc["comp"]:
        return "COMP"
    if hand_key == anc["ord"]:
        return "ORD"
    if hand_key == anc["col"]:
        return "COL"
    return "OTHER"


def is_separating(anc):
    return anc["comp"] is not None and anc["comp"] != anc["ord"] and anc["comp"] != anc["col"]


# ========================= RENDER-GATE (no policy server) =========================
def _project(env, cam, h, w, pts):
    try:
        import robosuite.utils.camera_utils as CU
        M = CU.get_camera_transform_matrix(env.sim, cam, h, w)
        out = []
        for p in pts:
            ph = M @ np.array([p[0], p[1], p[2], 1.0])
            out.append((ph[0] / ph[3], ph[1] / ph[3]))
        return np.array(out)
    except Exception as e:
        print("PROJECT_UNAVAILABLE", e, flush=True)
        return None


def _annotate(img, uv, labels, cols):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(np.ascontiguousarray(img)); d = ImageDraw.Draw(im)
        for (u, v), lab, col in zip(uv, labels, cols):
            d.ellipse([u - 6, v - 6, u + 6, v + 6], outline=col, width=3)
            d.text((u + 7, v - 9), lab, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception as e:
        print("ANNOTATE_SKIP", e, flush=True)
        return img


_RGB = {"red": (255, 40, 40), "green": (40, 220, 60), "blue": (60, 90, 240), "yellow": (240, 220, 40)}


def run_gate(a):
    pool = [a.cat] * a.K
    env = make_env(pool, a.layout, a.style, a.camw, a.seed)
    rng = np.random.RandomState(a.seed)
    env.reset()
    ai = 0 if CO.ROW_AXIS == "x" else 1
    print("GATE OrdinalPnP objects=%s axis=%s sign=%s ref=%s reach_x=%s reach_y=%s space=[%s,%s]" % (
        list(env.objects.keys()), CO.ROW_AXIS, CO.ROW_SIGN, CO.REF, CO.REACH_X, CO.REACH_Y,
        CO.SPACE_MIN, CO.SPACE_MAX), flush=True)
    for si in range(a.n_gate):
        env.reset()
        lr, N, obs, meta = place_row(env, rng, pin_N=a.K)
        P, cf, cc = ordcol_scene(rng, a.K)
        slot_colour = assign_ordcol(lr, P, cf, cc)
        recolour(env, slot_colour)
        env.sim.forward(); obs, _, _, _ = env.step(np.zeros(env.action_dim))
        # eef home (to verify nearest-cf == leftmost-cf, not COMP)
        eef = eef_pos_obs(obs)
        anc = anchors_for(lr, slot_colour, cf, 2)   # primary "2nd cf"
        coords = [round(float(objpos(obs, k)[ai]), 3) for k in lr]
        pts = np.array([objpos(obs, k) for k in lr])
        uv = _project(env, CAM_AGENT, a.camw, a.camw, pts)
        # nearest cf to eef home
        cf_keys = [k for k in lr if slot_colour[k] == cf]
        nearest_cf = min(cf_keys, key=lambda k: float(np.linalg.norm(eef - objpos(obs, k))))
        print("\n--- scene %d: K=%d cf=%s contrast=%s P=%s ---" % (si, a.K, cf, cc, P), flush=True)
        for i, k in enumerate(lr):
            print("   pos%d %-8s colour=%-6s row%s=%+.3f%s" % (
                i + 1, k, slot_colour[k], CO.ROW_AXIS, coords[i],
                "  u=%.1f" % uv[i, 0] if uv is not None else ""), flush=True)
        print("   INSTR 'pick the second %s object from the left ...'" % cf, flush=True)
        print("   COMP(2nd %s)=pos%d[%s]  ORD(2nd)=pos%d[%s]  COL(1st %s)=pos%d[%s]  separating=%s" % (
            cf, lr.index(anc["comp"]) + 1, anc["comp"], lr.index(anc["ord"]) + 1, anc["ord"],
            cf, lr.index(anc["col"]) + 1, anc["col"], is_separating(anc)), flush=True)
        print("   eef_home xy=(%.3f,%.3f)  nearest-cf=pos%d[%s]  (want nearest-cf==COL=pos%d so colour-heuristic!=COMP)" % (
            eef[0], eef[1], lr.index(nearest_cf) + 1, nearest_cf, lr.index(anc["col"]) + 1), flush=True)
        # annotate + dump
        frame = np.asarray(obs[CAM_AGENT + "_image"])[::-1].copy()
        if uv is not None:
            labs = ["p%d:%s" % (i + 1, slot_colour[lr[i]]) for i in range(len(lr))]
            cols = [_RGB.get(slot_colour[lr[i]], (255, 0, 0)) for i in range(len(lr))]
            frame = _annotate(frame, uv, labs, cols)
        imageio.imwrite("%s/gate_s%d_init.png" % (EVIZ, si), frame)
        # oracle filmstrip driven to COMP (confirms the colour-filtered ordinal maps to the correct PHYSICAL object)
        CO.RENDER = True
        succ, _, frames = rollout(env, obs, anc["comp"], record=False)
        if frames:
            strip = np.concatenate(frames[:16], axis=1)
            imageio.imwrite("%s/gate_s%d_oracle_COMP_%s.png" % (EVIZ, si, "S" if succ else "F"), strip)
        print("   wrote gate_s%d_init.png + oracle_COMP filmstrip (oracle->COMP succ=%s)" % (si, succ), flush=True)
    env.close()
    print("RC_EVAL_ORDCOL_GATE_DONE", flush=True)


# ========================= POLICY EVAL: compositional swap / occ =========================
def _rollout_policy(conn, env, obs, lr, instr, target_key, mode, max_steps, render, tag_prefix):
    """Drive the policy; return (hand_key, mind, succ_of_target). argmin over the K row objects."""
    keys = list(lr)
    mind = {k: 1e9 for k in keys}
    sxy = sink_xy(env)
    z0 = float(objpos(obs, target_key)[2]); zmax = z0
    frames = []
    send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
    for step in range(max_steps):
        img = np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1])
        wr = np.ascontiguousarray(np.asarray(obs[CAM_WRIST + "_image"])[::-1])
        if render and step % 10 == 0:
            frames.append(img.copy())
        if mode == "occ":
            img = np.zeros_like(img); wr = np.zeros_like(wr)
        send_msg(conn, {"cmd": "act", "blank": mode == "occ",
                        "obs": {"image_b": img.tobytes(), "image_shape": list(img.shape), "image_dtype": str(img.dtype),
                                "wrist_b": wr.tobytes(), "wrist_shape": list(wr.shape), "wrist_dtype": str(wr.dtype),
                                "instruction": instr, "state": mkstate(obs).tolist()}})
        act = np.asarray(recv_msg(conn)["action"], np.float32).ravel()
        ad = {"right": np.concatenate([act[:6], np.zeros(max(0, 6 - len(act)))])[:6],
              "right_gripper": np.array([act[6] if len(act) > 6 else -1.0], np.float32)}
        obs, r, done, info = env.step(env.robots[0].create_action_vector(ad))
        ep = eef_pos_obs(obs)
        for k in keys:
            mind[k] = min(mind[k], float(np.linalg.norm(ep - objpos(obs, k))))
        zmax = max(zmax, float(objpos(obs, target_key)[2]))
    hand_key = min(mind, key=mind.get)
    succ = bool(target_in_sink(env, obs, target_key, sxy)) and (zmax - z0 > MIN_LIFT)
    if render and frames:
        strip = np.concatenate(frames[:16], axis=1)
        imageio.imwrite("%s/%s.png" % (EVIZ, tag_prefix), strip)
    return hand_key, mind, succ


# instruction phrasings
def instr_comp(m, colour):
    return "pick the %s %s object from the %s and place it in the sink" % (ORDW[m], colour, REF)


def run_swap(a):
    """compositional SWAP-OBEY + 4-way tally + counterfactual triple. mode in {swap, occ}."""
    pool = [a.cat] * a.K
    conn = rconnect(a.port)   # robust connect (retries while a supervisor-restarted server comes back)
    env = make_env(pool, a.layout, a.style, a.camw, a.seed)
    rng = np.random.RandomState(a.seed)
    # per-instruction-kind tallies: kind in {comp2, cf1, cc2}. Each stores list of (class, separating, chance, absK)
    rows = []                     # one dict per scored episode
    # RESILIENCE (box3 external GPU-reset storms): append each episode to jsonl immediately (flush), and RESUME on
    # relaunch — reload prior rows, skip already-scored scenes (rng fast-forwards deterministically via the same
    # resampling replay), append new ones. A keep-alive wrapper relaunches on worker death => converges to n_scene.
    jpath = "%s/rows_%s.jsonl" % (EVIZ, a.mode)
    done_si = set()
    if os.path.exists(jpath):
        for line in open(jpath):
            try:
                r = json.loads(line); rows.append(r); done_si.add(r["si"])
            except Exception:
                pass
    jf = open(jpath, "a")
    if done_si:
        print("RESUME: %d scenes already scored -> fast-forwarding rng past them" % len(done_si), flush=True)
    n_skip = 0
    for si in range(a.n_scene):
        # §2 CONFOUND FILTER: re-place until nearest-cf-to-eef == leftmost-cf (COL). Guarantees COMP is distinct
        # from BOTH colour-heuristic readings (leftmost-cf AND nearest-cf) — a 'grab nearest {cf}' policy -> COL, not COMP.
        obs0 = None; ok = False
        for attempt in range(16):
            env.reset()
            lr, N, obs0, meta = place_row(env, rng, pin_N=a.K)
            P, cf, cc = ordcol_scene(rng, a.K)
            slot_colour = assign_ordcol(lr, P, cf, cc)
            recolour(env, slot_colour)
            env.sim.forward(); obs0, _, _, _ = env.step(np.zeros(env.action_dim))
            eef0 = eef_pos_obs(obs0)
            col_key = nth_of_colour(lr, slot_colour, cf, 1)
            near_cf = nearest_of_colour(obs0, lr, slot_colour, cf, eef0)
            if near_cf == col_key:          # nearest-cf == leftmost-cf => colour-heuristic unambiguously != COMP
                ok = True; break
        if not ok:
            n_skip += 1; continue
        if si in done_si:          # RESUME: already scored in a prior run; rng is now aligned -> skip re-scoring
            continue
        # counterfactual triple: (name, colour, ordinal)
        triple = [("comp2", cf, 2), ("cf1", cf, 1), ("cc2", cc, 2)]
        for (kind, colour, m) in triple:
            anc = anchors_for(lr, slot_colour, colour, m)
            if anc["comp"] is None:
                continue
            instr = instr_comp(m, colour)
            # SAME scene per triple. Re-step to settle + recompute eef for this instruction's nearest-colour check.
            env.sim.forward(); obs, _, _, _ = env.step(np.zeros(env.action_dim))
            eefh = eef_pos_obs(obs)
            near_key = nearest_of_colour(obs, lr, slot_colour, colour, eefh)
            nearest_ok = (near_key == anc["col"])   # this instruction's nearest-colour == its leftmost-colour (COL)
            tag = "%s_s%d_%s_%s%d" % (a.mode, si, kind, colour, m)
            try:
                hand_key, mind, succ = _rollout_policy(conn, env, obs, lr, instr, anc["comp"], a.mode,
                                                       a.max_steps, a.render, tag)
            except CONN_ERRS as e:
                # server died mid-episode (box3 external GPU-reset cycle). Reconnect + skip rest of this scene's triple.
                # rng was NOT advanced by the triple loop (only the resampling loop advances it) => univ/base pairing
                # stays aligned; this scene simply contributes fewer episodes. Then continue to the next scene.
                print("  [CONN lost @ s%d %s: %s -> reconnecting to :%d]" % (si, kind, e, a.port), flush=True)
                conn = rconnect(a.port)
                break
            cls = classify(hand_key, anc)
            sep = is_separating(anc)
            rows.append(dict(si=si, kind=kind, colour=colour, m=m, K=a.K, P=list(P), cf=cf, cc=cc,
                             comp_pos=lr.index(anc["comp"]) + 1,
                             ord_pos=(lr.index(anc["ord"]) + 1) if anc["ord"] else -1,
                             col_pos=(lr.index(anc["col"]) + 1) if anc["col"] else -1,
                             near_pos=(lr.index(near_key) + 1) if near_key else -1, nearest_ok=nearest_ok,
                             hand_pos=lr.index(hand_key) + 1, hand_colour=slot_colour[hand_key],
                             cls=cls, sep=sep, chance=1.0 / a.K, succ=int(succ)))
            jf.write(json.dumps(rows[-1]) + "\n"); jf.flush()
            print("  s%d %-5s '%s' -> hand=pos%d(%s) cls=%s sep=%s COMP=pos%d ORD=pos%d COL=pos%d" % (
                si, kind, instr.replace(" and place it in the sink", ""), lr.index(hand_key) + 1,
                slot_colour[hand_key], cls, sep, rows[-1]["comp_pos"], rows[-1]["ord_pos"], rows[-1]["col_pos"]),
                flush=True)
    jf.close()
    env.close()
    print("SCENES_SKIPPED_by_nearest_filter=%d" % n_skip, flush=True)
    _report(rows, a)


def _report(rows, a):
    # per-episode jsonl already written incrementally (resilience). Just analyze in-memory rows.
    sep = [r for r in rows if r["sep"] and r.get("nearest_ok", True)]
    n = len(sep)
    print("\n==================== ORDCOL %s REPORT ====================" % a.mode.upper(), flush=True)
    print("total episodes=%d  clean-separating (COMP!=ORD, COMP!=COL, nearest-colour==COL) episodes=%d" % (
        len(rows), n), flush=True)
    if n == 0:
        print("NO separating episodes — check scene generation", flush=True); return
    def rate(cls):
        k = sum(1 for r in sep if r["cls"] == cls)
        return k, wilson(k, n)
    for cls in ("COMP", "ORD", "COL", "OTHER"):
        k, (p, lo, hi) = rate(cls)
        print("  %-5s %3d/%d = %5.1f%% CI[%.1f,%.1f]" % (cls, k, n, p, lo, hi), flush=True)
    mchance = 100 * float(np.mean([r["chance"] for r in sep]))
    print("  chance (mean 1/K)          = %5.1f%%" % mchance, flush=True)
    # hand absolute-position histogram (expose any 'always grab pos-N' shortcut)
    from collections import Counter
    hp = Counter(r["hand_pos"] for r in sep)
    print("  hand abs-pos hist (sep): %s" % dict(sorted(hp.items())), flush=True)
    cp = Counter(r["comp_pos"] for r in sep)
    print("  COMP abs-pos hist (sep): %s" % dict(sorted(cp.items())), flush=True)
    # counterfactual tracking: per scene, fraction of triple obeyed (hand==that instr's correct target)
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["si"], []).append(r)
    obeyed_all = 0; scenes_full = 0
    for si, rs in by_scene.items():
        # obeyed = hand==COMP (the correct object for THAT instruction)
        ob = [1 if rr["cls"] == "COMP" else 0 for rr in rs]
        if len(rs) == 3:
            scenes_full += 1
            if sum(ob) == 3:
                obeyed_all += 1
    print("  counterfactual-triple: scenes where hand tracked ALL 3 targets = %d/%d" % (obeyed_all, scenes_full),
          flush=True)
    print("RC_EVAL_ORDCOL_DONE", flush=True)


# ========================= PARAPHRASE control: pure ordinal SWAP, reworded =========================
PARA_PHRASINGS = {
    "train": "pick the %s object from the left and place it in the sink",
    # reworded, same meaning, single-type ordinal (NO colour). Tests phrasing-robustness of a TRAINED primitive.
    "para":  "grab the item in the %s position counting from the left side and put it into the sink",
}


def run_para(a):
    """PURE ordinal SWAP with a reworded instruction (paraphrase control). Multi-cat pool, NO recolour.
    Mirrors rc_eval_ordinal --mode swap: name a DIFFERENT ordinal, check hand->named ordinal (OBEYED)."""
    pool = a.para_pool.split(",")
    conn = rconnect(a.port)
    env = make_env(pool, a.layout, a.style, a.camw, a.seed)
    rng = np.random.RandomState(a.seed)
    phr = PARA_PHRASINGS[a.phrasing]
    swap_hit = swap_tot = 0
    Ns = [int(x) for x in a.Ns.split(",")]
    cells = [(N, k) for N in Ns for k in range(1, N + 1) for _ in range(a.n)]
    ep = 0
    for (N, k) in cells:
        env.reset()
        lr, N, obs, meta = place_row(env, rng, pin_N=N)
        alts = [x for x in range(1, N + 1) if x != k]
        named_ord = alts[ep % len(alts)]
        named_key = lr[named_ord - 1]
        instr = phr % ORDW[named_ord]
        keys = list(lr); mind = {kk: 1e9 for kk in keys}
        try:
            send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
            for step in range(a.max_steps):
                img = np.ascontiguousarray(np.asarray(obs[CAM_AGENT + "_image"])[::-1])
                wr = np.ascontiguousarray(np.asarray(obs[CAM_WRIST + "_image"])[::-1])
                send_msg(conn, {"cmd": "act", "blank": False,
                                "obs": {"image_b": img.tobytes(), "image_shape": list(img.shape), "image_dtype": str(img.dtype),
                                        "wrist_b": wr.tobytes(), "wrist_shape": list(wr.shape), "wrist_dtype": str(wr.dtype),
                                        "instruction": instr, "state": mkstate(obs).tolist()}})
                act = np.asarray(recv_msg(conn)["action"], np.float32).ravel()
                ad = {"right": np.concatenate([act[:6], np.zeros(max(0, 6 - len(act)))])[:6],
                      "right_gripper": np.array([act[6] if len(act) > 6 else -1.0], np.float32)}
                obs, r, done, info = env.step(env.robots[0].create_action_vector(ad))
                ep_pos = eef_pos_obs(obs)
                for kk in keys:
                    mind[kk] = min(mind[kk], float(np.linalg.norm(ep_pos - objpos(obs, kk))))
        except CONN_ERRS as e:
            print("  [PARA CONN lost @ ep%d: %s -> reconnecting]" % (ep, e), flush=True)
            conn = rconnect(a.port); continue
        hand_key = min(mind, key=mind.get)
        hit = int(hand_key == named_key)
        swap_hit += hit; swap_tot += 1
        print("  PARA s%d N=%d named-ord=%d '%s' -> hand-ord=%d OBEY=%d" % (
            ep, N, named_ord, instr, lr.index(hand_key) + 1, hit), flush=True)
        ep += 1
    env.close()
    lo = wilson(swap_hit, swap_tot)
    print("[PARA-%s] hand->named-ordinal %d/%d = %.1f%% CI[%.1f,%.1f]" % (a.phrasing, swap_hit, swap_tot, *lo),
          flush=True)
    print("RC_EVAL_ORDCOL_DONE", flush=True)


def main(a):
    if a.mode == "gate":
        run_gate(a)
    elif a.mode == "para":
        run_para(a)
    else:
        run_swap(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="swap", choices=["gate", "swap", "occ", "para"])
    ap.add_argument("--cat", default="can")               # single category => shape constant
    ap.add_argument("--K", type=int, default=4)            # objects per row (chance = 1/K)
    ap.add_argument("--n_scene", type=int, default=30)     # scenes (each => counterfactual triple = 3 episodes)
    ap.add_argument("--n_gate", type=int, default=10)      # render-gate scenes
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--port", type=int, default=5710)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--camw", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--render", action="store_true")
    # paraphrase-control args
    ap.add_argument("--phrasing", default="para", choices=["train", "para"])
    ap.add_argument("--para_pool", default="apple,banana,lemon,carrot,can")
    ap.add_argument("--Ns", default="4")
    ap.add_argument("--n", type=int, default=12)
    main(ap.parse_args())
