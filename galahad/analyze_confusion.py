"""analyze_confusion.py — pool per-episode JSONL from rc_eval_liftslide_cf.py -> the DEFINITIVE verb 2x2 confusion.
GROUNDED  => diagonal >> off-diagonal (told-lift->lifts, told-slide->slides; told-slide rarely lifts).
NULL/fixed-motor => motion independent of the verb word (P(lift|told-lift) ~= P(lift|told-slide)).
Usage: python3 analyze_confusion.py /dev/shm/ls_cf
"""
import os, sys, json, math, glob

def wilson(k, n):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; z = 1.96; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*p, 100*max(0, c-h), 100*min(1, c+h))

H_HI = 0.12; H_LO = 0.06; PUSH_MIN = 0.10; H_PEAK = 0.10

def load(path):
    rows = []
    if os.path.exists(path):
        for ln in open(path):
            ln = ln.strip()
            if ln:
                try: rows.append(json.loads(ln))
                except: pass
    return rows

def motion_peak(r):
    """PEAK/ATTEMPT gate (grounding arbiter): did the verb control the MOTION TYPE attempted, independent of
    grasp-stability (a lift-then-drop = a lift ATTEMPT). lift = raised high at ANY point; slide = moved along the
    surface, NEVER raised. Disjoint (lift peak>0.12 vs slide peak<0.10)."""
    if r["peak"] > H_HI: return "lift"
    if r["moved"] >= PUSH_MIN and r["peak"] < H_PEAK: return "slide"
    return "noop"

def confusion(rows, gate="end"):
    cf = {"lift": {"lift":0,"slide":0,"noop":0}, "slide": {"lift":0,"slide":0,"noop":0}}
    for r in rows:
        m = r["motion"] if gate == "end" else motion_peak(r)
        cf[r["told_verb"]][m] += 1
    return cf

def print_2x2(name, rows, gate="end"):
    cf = confusion(rows, gate)
    n_l = sum(cf["lift"].values()); n_s = sum(cf["slide"].values())
    glabel = "END-STATE gate (held to final step; = pre-registered task-completion)" if gate=="end" else "PEAK/ATTEMPT gate (raised-at-any-point; = grounding arbiter, grasp-stability-agnostic)"
    print(f"\n===== {name} [{glabel}]  (told-lift n={n_l}, told-slide n={n_s}) =====")
    print(f"{'':12}| {'did_lift':>22} | {'did_slide':>22} | {'noop':>6}")
    smoke = []
    for told in ["lift","slide"]:
        nt = sum(cf[told].values())
        pl = wilson(cf[told]["lift"], nt); ps = wilson(cf[told]["slide"], nt)
        print(f"told-{told:7}| {cf[told]['lift']:3}/{nt:<3} {pl[0]:5.1f}%[{pl[1]:4.1f},{pl[2]:4.1f}] | "
              f"{cf[told]['slide']:3}/{nt:<3} {ps[0]:5.1f}%[{ps[1]:4.1f},{ps[2]:4.1f}] | {cf[told]['noop']:>6}")
        for cell,k in [("lift",cf[told]["lift"]),("slide",cf[told]["slide"])]:
            if nt>=20 and (k==0 or k==nt): smoke.append(f"told-{told}->{cell} = {k}/{nt} (0/N or N/N — cold-verify §2)")
    # OBEY (did the NAMED verb): diagonal pooled
    obey_k = cf["lift"]["lift"] + cf["slide"]["slide"]; obey_n = n_l + n_s
    ob = wilson(obey_k, obey_n)
    print(f"  POOLED OBEY (did-named-verb) = {obey_k}/{obey_n} = {ob[0]:.1f}% [{ob[1]:.1f},{ob[2]:.1f}]  (fixed-motor baseline=50%)")
    # grounded-vs-null diagnostics
    if n_l and n_s:
        p_lift_given_lift = cf["lift"]["lift"]/n_l
        p_lift_given_slide = cf["slide"]["lift"]/n_s
        p_slide_given_slide = cf["slide"]["slide"]/n_s
        p_slide_given_lift = cf["lift"]["slide"]/n_l
        print(f"  DIAG-vs-OFFDIAG: P(lift|told-lift)={100*p_lift_given_lift:.1f}%  P(lift|told-slide)={100*p_lift_given_slide:.1f}%  "
              f"(lift-selectivity delta={100*(p_lift_given_lift-p_lift_given_slide):+.1f}pp)")
        print(f"                   P(slide|told-slide)={100*p_slide_given_slide:.1f}%  P(slide|told-lift)={100*p_slide_given_lift:.1f}%  "
              f"(slide-selectivity delta={100*(p_slide_given_slide-p_slide_given_lift):+.1f}pp)")
    for s in smoke: print(f"  🔴 SMOKE-BOMB: {s}")
    return cf, (obey_k, obey_n)

def main(d):
    print(f"# VERB lift-vs-slide CONFUSION BATTERY — pooled from {d}")
    def load_glob(pat):
        rows = []
        for f in sorted(glob.glob(f"{d}/{pat}")):
            rows += load(f)
        return rows
    cured_task = load_glob("cured_task*.jsonl")   # pools main + any cured_task_sN shards
    base_task = load_glob("base_task*.jsonl")
    cured_occ = load_glob("cured_occ*.jsonl")
    cured_non = load_glob("cured_nonsense*.jsonl")
    cf_c = cf_b = None; obey_c = obey_b = None
    if cured_task:
        print_2x2("CURED  TASK (9000 ckpt)", cured_task, "end")
        cf_c, obey_c = print_2x2("CURED  TASK (9000 ckpt)", cured_task, "peak")
    if base_task:
        print_2x2("BASE   TASK (uncured anchor)", base_task, "end")
        cf_b, obey_b = print_2x2("BASE   TASK (uncured anchor)", base_task, "peak")
    # peak-gate is the grounding arbiter -> cf_c/cf_b hold the peak confusion for the verdict
    # OCC hygiene (both gates: blank cams should collapse lift/slide toward the motor default under EITHER gate)
    if cured_occ:
        n = len(cured_occ)
        print(f"\n===== CURED OCC (blank cams+zero state; MUST collapse) n={n} =====")
        for gate in ("end","peak"):
            cf = confusion(cured_occ, gate)
            allmot = {"lift":cf["lift"]["lift"]+cf["slide"]["lift"], "slide":cf["lift"]["slide"]+cf["slide"]["slide"],
                      "noop":cf["lift"]["noop"]+cf["slide"]["noop"]}
            occ_obey = cf["lift"]["lift"]+cf["slide"]["slide"]; ob = wilson(occ_obey, n)
            print(f"  [{gate:4}] OBEY(occ)={occ_obey}/{n}={ob[0]:.1f}%[{ob[1]:.1f},{ob[2]:.1f}]  mix: lift={allmot['lift']} slide={allmot['slide']} noop={allmot['noop']}  "
                  f"| told-lift(l/s/n)={cf['lift']['lift']}/{cf['lift']['slide']}/{cf['lift']['noop']} told-slide={cf['slide']['lift']}/{cf['slide']['slide']}/{cf['slide']['noop']}")
    # NONSENSE (both gates)
    if cured_non:
        n = len(cured_non)
        print(f"\n===== CURED NONSENSE ('fribble the X'; grounded=>LOW) n={n} =====")
        anymot_e = sum(1 for r in cured_non if r["did_lift"] or r["did_slide"])
        anymot_p = sum(1 for r in cured_non if motion_peak(r) != "noop")
        lift_p = sum(1 for r in cured_non if motion_peak(r)=="lift"); slide_p = sum(1 for r in cured_non if motion_peak(r)=="slide")
        be = wilson(anymot_e, n); bp = wilson(anymot_p, n)
        print(f"  [end ] any-clean-verb-motion = {anymot_e}/{n} = {be[0]:.1f}% [{be[1]:.1f},{be[2]:.1f}]")
        print(f"  [peak] any-motion-attempt   = {anymot_p}/{n} = {bp[0]:.1f}% [{bp[1]:.1f},{bp[2]:.1f}]  (lift={lift_p} slide={slide_p})")
    # VERDICT (arbiter = PEAK/attempt confusion: does the verb word control the motion TYPE, grasp-stability aside)
    print("\n===== VERDICT (arbiter = PEAK/attempt gate) =====")
    if cf_c:
        n_l = sum(cf_c["lift"].values()); n_s = sum(cf_c["slide"].values())
        sel_lift = cf_c["lift"]["lift"]/n_l - cf_c["slide"]["lift"]/n_s if n_l and n_s else 0
        sel_slide = cf_c["slide"]["slide"]/n_s - cf_c["lift"]["slide"]/n_l if n_l and n_s else 0
        obey = obey_c[0]/obey_c[1] if obey_c[1] else 0
        print(f"  CURED (peak) lift-selectivity = {100*sel_lift:+.1f}pp ; slide-selectivity = {100*sel_slide:+.1f}pp ; pooled OBEY(peak) = {100*obey:.1f}%")
        if cf_b:
            nbl = sum(cf_b['lift'].values()); nbs = sum(cf_b['slide'].values())
            bsel_l = cf_b['lift']['lift']/nbl - cf_b['slide']['lift']/nbs if nbl and nbs else 0
            print(f"  BASE  (peak) lift-selectivity = {100*bsel_l:+.1f}pp (anchor: cure-attributable excess = cured - base)")
        print("  GROUNDED  <= both selectivities strongly +, off-diagonal small (told-slide rarely lifts).")
        print("  NULL/fixed-motor <= selectivities ~0 (motion independent of the verb word).")
        print("  NOTE: END-STATE OBEY is lower than PEAK OBEY by the lift-then-drop rate = grasp-stability, NOT grounding.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/ls_cf")
