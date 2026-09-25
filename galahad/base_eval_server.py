"""base_eval_server.py — serve the UNCURED BASE MolmoAct2-Think-LIBERO through the libero_pro_eval socket,
using the SAME lerobot MolmoAct2 policy + act() contract as arm_eval_server.py (ARM=A: observation.state=proprio(8),
task=instruction). Purpose: the BASE ANCHOR / positive control — score the uncured base on the EXACT SAME env-side
contract (libero_pro_eval) the cured b1_spatial model scored on, so "cured 51%" can be attributed to the cure/data
rather than to eval hardness.

Why a separate server: the cured ckpt is lerobot-format (config.json is a lerobot PreTrainedConfig + saved
policy_pre/postprocessor_*.safetensors), so arm_eval_server.py loads it with make_pre_post_processors(pretrained_path).
The BASE instead ships HF-transformers weights (sharded model-0000N-of-00005.safetensors) + norm_stats.json and has
NO lerobot processor files -> that path can't load it. Loading path used here:

  1. POLICY: build the lerobot MolmoAct2 config with checkpoint_path=<base dir>, pretrained_path=None. make_policy
     with a falsy pretrained_path builds the policy FRESH -> MolmoAct2Policy._load_hf_model() loads the base's trained
     VLM + trained CONTINUOUS action expert (base config: add_action_expert=true, action_mode="both", 660 action_expert
     tensors present). LoRA is turned OFF so the forward pass is the exact base. action_mode/inference_action_mode are
     forced to "continuous" (same head the cured eval uses).
  2. NORMALIZER: build pre/post via make_molmoact2_pre_post_processors(cfg) with norm_tag="libero". That reads the
     base's OWN norm_stats.json ["libero"] q01/q99 = the base's NATIVE normalization. (The cured model instead used
     spatial-dataset-fitted stats saved in its processors. Each model uses ITS OWN training normalizer -- that is
     correct, not a confound: the env-side contract that must match across the two runs -- 180deg agentview+wrist flip,
     state=[eef_pos,quat2axisangle,gripper_qpos] 8-dim, continuous 7-DoF action, chunk_size=10 -- lives entirely in
     libero_pro_eval.py, which is byte-identical for both. So base-vs-cured isolates the model weights.)

Run (mirror arm_eval_server.py; use a FREE gpu + a port NOT in 5000-5003 which the cured fleet holds):
  BASE=/root/MolmoAct2-Think-LIBERO NORM_TAG=libero CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
    /venv/main/bin/python base_eval_server.py --port 5950
"""
import os, socket, struct, pickle, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5950)
args = ap.parse_args()

BASE = os.environ.get("BASE", "/root/MolmoAct2-Think-LIBERO")
NORM_TAG = os.environ.get("NORM_TAG", "libero")
# ds_meta ONLY supplies the feature schema (image/state/action shapes) to make_policy; the normalization stats come
# from norm_stats.json via make_molmoact2_pre_post_processors, NOT from this dataset. Reuse the cured training set's
# metadata so the feature schema is identical to the cured run.
REPO = os.environ.get("REPO", "phimonster/b1-spatial-s1")
ROOT = os.environ.get("ROOT", "/root/libdata/b1_spatial_s1")
# Start from the cured lerobot config so the policy CLASS + all architecture/inference hyperparameters (chunk_size=10,
# num_state_tokens=256, expected_max_action_dim=32, flow-matching schedule) are byte-identical to the cured run; then
# flip the four fields that select "uncured base" instead of "cured fine-tune".
CFG_DIR = os.environ.get("CURED_CFG_DIR", "/root/runs/b1_spatial_s1/checkpoints/003000/pretrained_model")

cfg = PreTrainedConfig.from_pretrained(CFG_DIR)
cfg.pretrained_path = None                # <- do NOT load the cured fine-tuned weights: pure base
cfg.checkpoint_path = BASE                # <- load base VLM + trained continuous action expert from here
cfg.enable_lora_vlm = False               # <- no adapters -> exact base forward pass (older-fork flag)
cfg.enable_lora_action_expert = False
cfg.use_peft = False
try: cfg.train_mode_vlm = "freeze"        # <- a4f15bf-era flag: freeze = no LoRA modules = exact base
except Exception: pass
cfg.action_mode = "continuous"
cfg.inference_action_mode = "continuous"
cfg.norm_tag = NORM_TAG                   # <- make_molmoact2_pre_post_processors reads BASE/norm_stats.json[NORM_TAG]
cfg.device = "cuda"

dsm = LeRobotDatasetMetadata(REPO, root=ROOT)
policy = make_policy(cfg=cfg, ds_meta=dsm)   # pretrained_path falsy -> fresh policy -> _load_hf_model loads base weights
policy.eval()
pre, post = make_molmoact2_pre_post_processors(cfg)   # base's native normalizer from norm_stats.json[NORM_TAG]
DEV = "cuda"

# INVARIANT (adaLN-Zero): the base lacks the Galahad `action_expert.state_encoder` (loads as MISSING -> newly init),
# but its OUTPUT layer is zero-initialized in __init__, so state_encoder(x)=0 -> _pose_cond=0 -> the pose injection
# (conditioning += _pose_cond) is a NO-OP. That is what makes this a FAITHFUL uncured base (native MolmoAct2, 256
# discrete state tokens; Galahad continuous-pose-injection inert). If the output layer were NONzero, a random pose
# signal would corrupt the base's actions = a confounded anchor. Assert it here at every launch.
try:
    _ae = policy._action_expert()
    _se = getattr(_ae, "state_encoder", None)
    if _se is not None:
        _se[-1].weight.data.zero_(); _se[-1].bias.data.zero_()  # base has NO trained state_encoder (loads random) -> force adaLN-Zero inert = faithful native base (this IS the agent's stated intent; the random-load just didn't land on zero)
        _wmax = float(_se[-1].weight.abs().max()); _bmax = float(_se[-1].bias.abs().max())
        _ok = (_wmax == 0.0 and _bmax == 0.0)
        print(f"[BASE-INVARIANT] state_encoder out-layer |w|max={_wmax:.3e} |b|max={_bmax:.3e} "
              f"-> pose-injection {'INERT (adaLN-Zero, base faithful)' if _ok else 'NONZERO !! CONFOUND — base anchor INVALID'}", flush=True)
        assert _ok, "state_encoder output layer is NONzero on the base -> random pose corrupts actions; anchor invalid"
    else:
        print("[BASE-INVARIANT] no state_encoder module present (pure native base) — OK", flush=True)
except AssertionError:
    raise
except Exception as _e:
    print(f"[BASE-INVARIANT] check skipped ({_e})", flush=True)

print(f"BASE_EVAL_SERVER loaded (uncured {BASE}, norm_tag={NORM_TAG})", flush=True)


@torch.no_grad()
def act(ob):
    img = np.ascontiguousarray(ob["image"]).astype(np.uint8)
    wrist = np.ascontiguousarray(ob.get("wrist", img)).astype(np.uint8)
    state = np.asarray(ob.get("state", np.zeros(8)), np.float32)[:8]   # ARM=A proprio(8), same as arm_eval_server
    task = str(ob.get("instruction", ""))
    batch = {
        "observation.images.image": torch.from_numpy(img).permute(2, 0, 1)[None].to(DEV),
        "observation.images.wrist_image": torch.from_numpy(wrist).permute(2, 0, 1)[None].to(DEV),
        "observation.state": torch.from_numpy(state)[None].to(DEV),
        "task": [task],
    }
    b = pre(batch)
    a = policy.select_action(b)
    a = post(a)
    if isinstance(a, dict):
        a = a["action"]
    a = a.detach().float().cpu().numpy() if hasattr(a, "detach") else np.asarray(a, np.float32)
    return np.asarray(a, np.float32).ravel()[:7]


def _recv(c, n):
    b = b""
    while len(b) < n:
        d = c.recv(n - len(b))
        if not d:
            return None
        b += d
    return b


def recv_msg(c):
    h = _recv(c, 4)
    return None if h is None else pickle.loads(_recv(c, struct.unpack(">I", h)[0]))


def send_msg(c, o):
    d = pickle.dumps(o, protocol=4); c.sendall(struct.pack(">I", len(d)) + d)


EP_SEED = 0  # per-episode seed counter (reproducible flow sampling), identical to arm_eval_server
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args.port)); srv.listen(8)
print(f"BASE_EVAL_SERVER_READY port={args.port}", flush=True)
while True:
    conn, _ = srv.accept()
    try:
        while True:
            msg = recv_msg(conn)
            if msg is None:
                break
            cmd = msg.get("cmd")
            if cmd == "reset":
                EP_SEED += 1; torch.manual_seed(1234 + EP_SEED); np.random.seed(1234 + EP_SEED)
                policy.reset(); send_msg(conn, {})
            elif cmd == "act":
                ob = msg["obs"]
                ob["image"] = np.frombuffer(ob.pop("image_b"), dtype=ob.pop("image_dtype")).reshape(ob.pop("image_shape"))
                if "wrist_b" in ob:
                    ob["wrist"] = np.frombuffer(ob.pop("wrist_b"), dtype=ob.pop("wrist_dtype")).reshape(ob.pop("wrist_shape"))
                a = act(ob)
                send_msg(conn, {"action": [float(x) for x in a]})
            elif cmd == "ping":
                send_msg(conn, {"ok": True})
            else:
                send_msg(conn, {"err": f"unknown {cmd}"})
    except Exception as e:
        print(f"[base_srv] err {e}", flush=True)
    finally:
        conn.close()
