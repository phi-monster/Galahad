"""merge_a7.py — F1 for the A7 unified 9-axis release: fold the native-MolmoAct2 LoRA (PEFT-wrapped: keys are
`base_model.model...lora_A.default.weight`) into the base MolmoAct2 => ONE merged single-weight checkpoint.

A7 is ACTION-ONLY: there is NO `_gf` foresight head (unlike the C3-jfs flagship merge_galahad.py handles). So this
drops all foresight/galahad_foresight machinery. merge = peft merge_and_unload (W += (alpha/r) B A), config LoRA-off,
save single model.safetensors + the ckpt's own processors, then self-checks A(keys)/B(loads+runs)/C(action-equiv).

Run (GPU, ~10-20 min):
  cd <grounding_ladder>
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
    SRC=/dev/shm/a7_unified/checkpoints/018000/pretrained_model OUT=/dev/shm/galahad-a7-merged \
    REPO=a9/full ROOT=/dev/shm/a7_merged9 /root/molmoact2/lerobot/.venv/bin/python merge_a7.py
"""
import os, sys, json, shutil, glob
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from safetensors.torch import save_file

SRC = os.environ.get("SRC", "/dev/shm/a7_unified/checkpoints/018000/pretrained_model")
OUT = os.environ.get("OUT", "/dev/shm/galahad-a7-merged")
REPO = os.environ.get("REPO", "a9/full")
ROOT = os.environ.get("ROOT", "/dev/shm/a7_merged9")
DEV = "cuda"
N_PROBE = int(os.environ.get("N_PROBE", "8"))


def zero_state_encoder(policy, tag):
    """A7 is action-only: state_encoder is untrained (random-nonzero out-layer would corrupt AR generation). Zero it
    == exactly how arm_eval_server ran the battery (ZERO_STATE)."""
    try:
        ae = policy._action_expert()
        se = getattr(ae, "state_encoder", None)
        if se is not None:
            se[-1].weight.data.zero_(); se[-1].bias.data.zero_()
            print(f"[{tag}] state_encoder out-layer zeroed", flush=True)
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
        cfg.train_mode_vlm = "fft"   # merged weights are plain (LoRA folded) -> build a plain (non-LoRA) model tree
    dsm = LeRobotDatasetMetadata(REPO, root=ROOT)
    policy = make_policy(cfg=cfg, ds_meta=dsm)
    policy.eval()
    zero_state_encoder(policy, "load")
    pre, post = make_pre_post_processors(cfg, pretrained_path=pretrained_path)
    return policy, pre, post


def fixed_obs_batch(n):
    rng = np.random.RandomState(0)
    instrs = ["pick up the alphabet soup and place it in the basket",
              "pick up the cream cheese and place it in the basket",
              "pick up the ketchup and place it in the basket",
              "pick up the milk and place it in the basket"]
    obs = []
    for i in range(n):
        obs.append({"image": rng.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                    "wrist": rng.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                    "instr": instrs[i % len(instrs)]})
    return obs


@torch.no_grad()
def act(policy, pre, post, ob):
    img = np.ascontiguousarray(ob["image"]).astype(np.uint8)
    wrist = np.ascontiguousarray(ob["wrist"]).astype(np.uint8)
    state = np.zeros(8, np.float32)
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

    print("=== [1] load A7 (base + LoRA) ===", flush=True)
    policy, pre, post = build_policy(SRC, lora_on=True)
    n_lora_before = sum(1 for k in policy.state_dict() if "lora" in k.lower())
    print(f"lora keys before merge = {n_lora_before}", flush=True)

    print("=== [2] pre-merge actions on fixed obs ===", flush=True)
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
    assert merged, "no PeftModel found — nothing to merge (check module tree)"

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
    assert not gf_keys, f"unexpected _gf keys (A7 is action-only): {gf_keys[:3]}"
    assert not bmm, f"base_model.model prefix remains (PEFT not unwrapped): {bmm[:3]}"
    sd = {k: v.contiguous().cpu() for k, v in sd.items()}
    save_file(sd, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    cfg["train_mode_vlm"] = "fft"   # merged single-weight = plain model; no LoRA re-application on load
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    for f in glob.glob(os.path.join(SRC, "policy_pre*")) + glob.glob(os.path.join(SRC, "policy_post*")) + \
             glob.glob(os.path.join(SRC, "train_config.json")):
        shutil.copy(f, OUT)
    print(f"saved -> {OUT}  (keys={len(sd)} lora=0 gf=0)", flush=True)
    del policy
    torch.cuda.empty_cache()

    print("=== [6] fresh-load the SAVED single file (enable_lora=false) + run ===", flush=True)
    policy2, pre2, post2 = build_policy(OUT, lora_on=False)
    a_reload = run_all(policy2, pre2, post2, obs)
    d_rl = float(np.abs(a_orig - a_reload).max())
    print(f"max|a_orig - a_reload| = {d_rl:.3e}", flush=True)

    tol = float(os.environ.get("TOL", "5e-2"))
    verdict = "PASS" if (d_mm <= tol and d_rl <= tol) else "FAIL"
    report = {"src": SRC, "out": OUT, "n_probe": N_PROBE,
              "lora_keys_before": n_lora_before, "lora_keys_after": 0, "gf_keys": 0,
              "merged_peft_modules": merged, "max_abs_diff_merge": d_mm, "max_abs_diff_reload": d_rl,
              "tol": tol, "verdict": verdict}
    json.dump(report, open(os.path.join(OUT, "merge_selfcheck.json"), "w"), indent=2)
    print("=== MERGE_SELFCHECK ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print("A7_MERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
