"""do_nothing_check.py — B57 init-success red-team. Runs a ZERO-ACTION policy through each new type's SUCCESS metric.
A do-nothing MUST score ~0% (else the success is init-True and the un-fakeable battery is silently defeated).
Reports do-nothing SR per type + per condition. Also prints init-success at reset (must be False every episode).

Run: TYPE=goal N=16 ... python do_nothing_check.py     |     TYPE=verb N=16 ... python do_nothing_check.py
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
sys.path.insert(0, "/root")
TYPE = os.environ.get("TYPE", "goal")
N = int(os.environ.get("N", "16"))

def log(*a): print(*a, flush=True)

if TYPE == "goal":
    import collect_rc_goal as G
    G.RENDER = False; G.DEBUG = False
    rng = np.random.RandomState(555)
    OBJ = os.environ.get("OBJ_POOL", "apple,banana,carrot").split(",")
    REC = G.REC_POOL
    succ = 0; init_true = 0
    for i in range(N):
        obj = OBJ[i % len(OBJ)]
        env = G.make_env(obj, 40000 + i * 3)
        env.reset()
        rec_keys = [k for k in env.objects.keys() if k != "obj"]
        obs = G.randomize(env, rng)
        named = rec_keys[i % len(rec_keys)]
        # init check (lifted=0)
        if G.delivered(env, obs, "obj", rec_keys, named, 0.0): init_true += 1
        # zero-action rollout
        obj_zmax = float(G.objpos(obs, "obj")[2]); z0 = obj_zmax
        for _ in range(G.MAXS):
            obs, r, d, info = env.step(np.zeros(env.action_dim))
            obj_zmax = max(obj_zmax, float(G.objpos(obs, "obj")[2]))
        if G.delivered(env, obs, "obj", rec_keys, named, obj_zmax - z0): succ += 1
        env.close()
    log("GOAL do-nothing: SR %d/%d  init-True %d/%d  (both MUST be ~0)" % (succ, N, init_true, N))

elif TYPE == "verb":
    import collect_rc_verb as V
    import robocasa.utils.object_utils as OU
    V.RENDER = False; V.DEBUG = False
    rng = np.random.RandomState(777)
    POOL = V.POOL
    for verb in ("pick", "push"):
        succ = 0; init_true = 0
        for i in range(N):
            tcat = POOL[i % len(POOL)]
            env = V.make_counter_env(tcat, 41000 + i * 3)
            env.reset()
            obs = V.randomize_objs(env, rng)
            z0 = float(V.objpos(obs, "obj")[2]); obj_zmax = z0; tp0 = V.objpos(obs, "obj")[:2].copy()
            # init check
            if verb == "pick":
                if bool(OU.obj_inside_of(env, "obj", env.sink, partial_check=True)): init_true += 1
            # zero-action rollout
            for _ in range(V.MAXS):
                obs, r, d, info = env.step(np.zeros(env.action_dim))
                obj_zmax = max(obj_zmax, float(V.objpos(obs, "obj")[2]))
            tp = V.objpos(obs, "obj")
            if verb == "pick":
                s = bool(OU.obj_inside_of(env, "obj", env.sink, partial_check=True)) and (obj_zmax - z0) > V.MIN_LIFT
            else:
                moved = float(np.linalg.norm(tp[:2] - tp0)); s = moved >= V.PUSH_MIN and (obj_zmax - z0) < V.LIFT_MAX
            succ += int(s)
            env.close()
        log("VERB %-4s do-nothing: SR %d/%d  init-True %d/%d  (MUST be ~0)" % (verb, succ, N, init_true, N))

log("DO_NOTHING_DONE")
