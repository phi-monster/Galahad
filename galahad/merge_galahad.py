"""merge_galahad.py — F1: fold the C3-jfs LoRA into the base MolmoAct2 + keep the `_gf` foresight head in the SAME
safetensors => the ONE merged single-weight Galahad release (LoRA merged, foresight head same file, dual output from
one weight-set). NOT a base+adapter+heads parts-bundle (the thing we critique in V-JEPA2-AC / Genie cascades).

WHAT THE FLAGSHIP IS (LAB B31/B40): MolmoAct2-Think-LIBERO base + LoRA rank32/alpha16 on VLM+action_expert (PEFT
get_peft_model) + a `_gf` ForesightHeads module (8 learnable tokens + region/depth decoders sharing the trunk),
trained joint-from-start 3000 steps. Inference appends the 8 foresight tokens (train/eval contract seq 470->478,
galahad_foresight.py INFER patch). The `_gf` head shares the trunk hidden states => one forward, dual output.

MERGE = peft merge_and_unload folds W += (alpha/r) B A into the base Linear weights, removes the LoRA modules =>
plain (non-PEFT) module tree. The forward math is IDENTICAL, so actions are preserved (verified below). The `_gf.*`
tensors are carried through unchanged and saved in the same model.safetensors. Config is rewritten with LoRA OFF so a
fresh load builds a plain policy that the merged (no-lora, no base_model.model prefix) state_dict populates cleanly.

SELF-CHECKS (the "单文件加载自检"):
  A key-level  : merged state_dict has 0 lora keys, has _gf.* keys, no base_model.model prefix.
  B load-and-run: fresh-load the SAVED dir (enable_lora=false) + galahad_foresight patch => select_action returns a
                  7-d action with no error  (proves the single file loads and runs).
  C equivalence : pre-merge vs post-merge vs fresh-reload actions on the SAME fixed obs batch match within tol
                  (proves the fold is faithful AND the saved file reproduces the flagship policy).

Run (GPU, ~10-20 min):
  cd /root/research/phi-arena/scripts/grounding_ladder
  GALAHAD_FS=1 GALAHAD_FS_INFER=1 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
    SRC=/root/runs/c3_jfs/checkpoints/003000/pretrained_model OUT=/dev/shm/galahad-merged \
    REPO=lerobot/libero_c1v2 ROOT=/dev/shm/libero_c1v2 /venv/main/bin/python merge_galahad.py
"""
import os, sys, json, shutil, glob

os.environ.setdefault("GALAHAD_FS", "1")          # so patch_all() builds `_gf` and the ckpt's _gf.* load cleanly
os.environ.setdefault("GALAHAD_FS_INFER", "1")    # append foresight tokens at inference (the TRUE flagship path)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galahad_foresight  # noqa: F401  — import applies the monkey-patch (ForesightHeads + inference token-append)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from safetensors.torch import save_file

SRC = os.environ.get("SRC", "/root/runs/c3_jfs/checkpoints/003000/pretrained_model")
OUT = os.environ.get("OUT", "/dev/shm/galahad-merged")
REPO = os.environ.get("REPO", "lerobot/libero_c1v2")
ROOT = os.environ.get("ROOT", "/dev/shm/libero_c1v2")
DEV = "cuda"
N_PROBE = int(os.environ.get("N_PROBE", "8"))


def zero_state_encoder(policy, tag):
    """Replicate the arm_eval_server CURED-INVARIANT: the untrained state_encoder out-layer is random-nonzero and
    corrupts the AR generation (hang). Zero it so pose-injection is a NO-OP == exactly how B40 was evaluated."""
    try:
        ae = policy._action_expert()
        se = getattr(ae, "state_encoder", None)
        if se is not None:
            se[-1].weight.data.zero_(); se[-1].bias.data.zero_()
            print(f"[{tag}] state_encoder out-layer zeroed |w|max={float(se[-1].weight.abs().max()):.2e}", flush=True)
        else:
            print(f"[{tag}] no state_encoder module — OK", flush=True)
    except Exception as e:
        print(f"[{tag}] state_encoder zero SKIPPED ({e})", flush=True)


def build_policy(pretrained_path, lora_on):
    cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    cfg.pretrained_path = pretrained_path
    cfg.device = DEV
    cfg.inference_action_mode = "continuous"
    if not lora_on:
        cfg.enable_lora_vlm = False
        cfg.enable_lora_action_expert = False
    dsm = LeRobotDatasetMetadata(REPO, root=ROOT)
    policy = make_policy(cfg=cfg, ds_meta=dsm)
    policy.eval()
    zero_state_encoder(policy, "load")
    pre, post = make_pre_post_processors(cfg, pretrained_path=pretrained_path)
    return policy, pre, post


def fixed_obs_batch(n):
    """Deterministic synthetic obs — equivalence is input-agnostic (same math on same inputs); we only need identical
    inputs to both models, plus a real instruction so the language path is exercised."""
    rng = np.random.RandomState(0)
    instrs = ["pick up the alphabet soup and place it in the basket",
              "pick up the cream cheese and place it in the basket",
              "pick up the ketchup and place it in the basket",
              "pick up the milk and place it in the basket"]
    obs = []
    for i in range(n):
        img = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        wrist = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        obs.append({"image": img, "wrist": wrist, "instr": instrs[i % len(instrs)]})
    return obs


@torch.no_grad()
def act(policy, pre, post, ob):
    img = np.ascontiguousarray(ob["image"]).astype(np.uint8)
    wrist = np.ascontiguousarray(ob["wrist"]).astype(np.uint8)
    state = np.zeros(8, np.float32)                      # ZERO_STATE (arm_eval_server convention)
    batch = {
        "observation.images.image": torch.from_numpy(img).permute(2, 0, 1)[None].to(DEV),
        "observation.images.wrist_image": torch.from_numpy(wrist).permute(2, 0, 1)[None].to(DEV),
        "observation.state": torch.from_numpy(state)[None].to(DEV),
        "task": [ob["instr"]],
    }
    policy.reset()
    torch.manual_seed(1234); np.random.seed(1234)
    b = pre(batch); a = policy.select_action(b); a = post(a)
    if isinstance(a, dict):
        a = a["action"]
    a = a.detach().float().cpu().numpy() if hasattr(a, "detach") else np.asarray(a, np.float32)
    return np.asarray(a, np.float32).ravel()[:7]


def run_all(policy, pre, post, obs):
    return np.stack([act(policy, pre, post, ob) for ob in obs])


def main():
    os.makedirs(OUT, exist_ok=True)
    obs = fixed_obs_batch(N_PROBE)

    print("=== [1] load C3-jfs (base + LoRA + _gf) ===", flush=True)
    policy, pre, post = build_policy(SRC, lora_on=True)
    n_lora_before = sum(1 for k in policy.state_dict() if "lora" in k.lower())
    print(f"lora keys before merge = {n_lora_before}", flush=True)
    print("=== [2] pre-merge actions on fixed obs (flagship, tokens ON) ===", flush=True)
    a_orig = run_all(policy, pre, post, obs)
    print("a_orig[0] =", np.round(a_orig[0], 4).tolist(), flush=True)

    print("=== [3] peft merge_and_unload (fold LoRA into base) ===", flush=True)
    from peft import PeftModel
    merged = []

    def walk(m, prefix=""):
        for n, c in list(m.named_children()):
            full = f"{prefix}.{n}" if prefix else n
            if isinstance(c, PeftModel):
                setattr(m, n, c.merge_and_unload())
                merged.append(full)
            else:
                walk(c, full)

    walk(policy)
    print("MERGED_PEFT_MODULES =", merged, flush=True)
    assert merged, "no PeftModel found — nothing to merge (check enable_lora / module tree)"

    print("=== [4] post-merge actions (must match pre-merge) ===", flush=True)
    a_merged = run_all(policy, pre, post, obs)
    d_mm = float(np.abs(a_orig - a_merged).max())
    print(f"max|a_orig - a_merged| = {d_mm:.3e}", flush=True)

    print("=== [5] save merged single-weight + lora-off config + processors ===", flush=True)
    sd = policy.state_dict()
    lora_after = [k for k in sd if "lora" in k.lower()]
    gf_keys = [k for k in sd if k.startswith("_gf")]
    bmm = [k for k in sd if "base_model.model" in k]
    assert not lora_after, f"lora keys remain after merge: {lora_after[:3]}"
    assert gf_keys, "_gf foresight head missing from merged state_dict!"
    assert not bmm, f"base_model.model prefix remains (PEFT not unwrapped): {bmm[:3]}"
    # move to cpu + contiguous for safetensors
    sd = {k: v.contiguous().cpu() for k, v in sd.items()}
    save_file(sd, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    cfg["enable_lora_vlm"] = False
    cfg["enable_lora_action_expert"] = False
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    for f in glob.glob(os.path.join(SRC, "policy_pre*")) + glob.glob(os.path.join(SRC, "policy_post*")) + \
             glob.glob(os.path.join(SRC, "train_config.json")):
        shutil.copy(f, OUT)
    print(f"saved -> {OUT}  (keys={len(sd)} lora=0 gf={len(gf_keys)})", flush=True)
    del policy
    torch.cuda.empty_cache()

    print("=== [6] fresh-load the SAVED single file (enable_lora=false) + run ===", flush=True)
    policy2, pre2, post2 = build_policy(OUT, lora_on=False)
    a_reload = run_all(policy2, pre2, post2, obs)
    d_rl = float(np.abs(a_orig - a_reload).max())
    print(f"max|a_orig - a_reload| = {d_rl:.3e}", flush=True)

    tol = float(os.environ.get("TOL", "5e-2"))
    ok_mm = d_mm <= tol
    ok_rl = d_rl <= tol
    verdict = "PASS" if (ok_mm and ok_rl) else "FAIL"
    report = {
        "src": SRC, "out": OUT, "n_probe": N_PROBE,
        "lora_keys_before": n_lora_before, "lora_keys_after": 0,
        "gf_keys": len(gf_keys), "merged_peft_modules": merged,
        "max_abs_diff_merge": d_mm, "max_abs_diff_reload": d_rl, "tol": tol,
        "self_check_A_keys": True, "self_check_B_loads_and_runs": True,
        "self_check_C_equivalence": bool(ok_mm and ok_rl), "verdict": verdict,
    }
    json.dump(report, open(os.path.join(OUT, "merge_selfcheck.json"), "w"), indent=2)
    print("=== MERGE_SELFCHECK ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"MERGE_GALAHAD_DONE verdict={verdict} d_merge={d_mm:.2e} d_reload={d_rl:.2e}", flush=True)
    if verdict != "PASS":
        print("🔴 equivalence exceeded tol — bf16 fold drift; inspect before trusting the release", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
