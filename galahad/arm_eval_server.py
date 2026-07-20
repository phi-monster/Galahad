"""arm_eval_server.py — serve a CURED (LoRA fine-tuned) MolmoAct2 lerobot ckpt through the rc_eval/libero_pro_eval
socket (ARM=A: observation.state=proprio(8), task=instruction, continuous 7-DoF action, chunk_size=10).

The cured ckpt is lerobot-format (`<ckpt>/pretrained_model/`: config.json = lerobot PreTrainedConfig + model.safetensors
+ saved policy_pre/postprocessor_*.safetensors). Load path (verified API, lerobot 0.6.0):
  cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.pretrained_path = CKPT  (loads the CURED weights, incl. LoRA)
  policy = make_policy(cfg, ds_meta)                                        (ds_meta = feature schema only)
  pre, post = make_pre_post_processors(cfg, pretrained_path=CKPT)           (the CKPT's OWN saved normalizer)
Keep the ckpt's LoRA flags (do NOT disable — the fine-tuned forward pass IS the cured model).

Run (free GPU; port not colliding with a live server):
  CKPT=/root/runs/g1_universal/checkpoints/018000/pretrained_model \
  REPO=g1/universal-merged ROOT=/root/g1_datasets/merged \
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 /venv/main/bin/python arm_eval_server.py --port 5600
Ready line: ARM_EVAL_SERVER_READY
"""
import os, socket, struct, pickle, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5600)
args = ap.parse_args()

CKPT = os.environ.get("CKPT", "/root/runs/g1_universal/checkpoints/018000/pretrained_model")
# ds_meta ONLY supplies the feature schema (image/state/action shapes) to make_policy; the normalization stats come
# from the cured ckpt's OWN saved processors via make_pre_post_processors(pretrained_path=CKPT), NOT this dataset.
REPO = os.environ.get("REPO", "g1/universal-merged")
ROOT = os.environ.get("ROOT", "/root/g1_datasets/merged")

cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.pretrained_path = CKPT                 # <- load the CURED fine-tuned weights (incl. LoRA)
cfg.device = "cuda"
cfg.inference_action_mode = "continuous"

dsm = LeRobotDatasetMetadata(REPO, root=ROOT)
policy = make_policy(cfg=cfg, ds_meta=dsm)
policy.eval()
# 🔴 Galahad pose-injection inert (adaLN-Zero): the box1 MolmoAct2 policy sets `action_expert._pose_cond` from
# observation.state via `state_encoder` (predict_action_chunk). Our LoRA ckpt has NO trained state_encoder (loads
# MISSING -> random NONZERO out-layer) -> a random pose_cond CORRUPTS the autoregressive generation = HUNG/runaway
# inference. Force the out-layer to zero (== base_eval_server's base-anchor treatment) so pose-injection is a NO-OP and
# the model is evaluated on its VISION+LANGUAGE grounding. Cured(zeroed) vs base(zeroed) => apples-to-apples.
try:
    _ae = policy._action_expert()
    _se = getattr(_ae, "state_encoder", None)
    if _se is not None:
        _se[-1].weight.data.zero_(); _se[-1].bias.data.zero_()
        _wmax = float(_se[-1].weight.abs().max()); _bmax = float(_se[-1].bias.abs().max())
        print(f"[CURED-INVARIANT] state_encoder out-layer zeroed |w|max={_wmax:.3e} |b|max={_bmax:.3e} -> pose-injection INERT", flush=True)
    else:
        print("[CURED-INVARIANT] no state_encoder module (pose-injection absent) — OK", flush=True)
except Exception as _e:
    print(f"[CURED-INVARIANT] state_encoder zero SKIPPED ({_e}) — if inference hangs this is why", flush=True)
pre, post = make_pre_post_processors(cfg, pretrained_path=CKPT)   # cured's native normalizer (saved during training)
DEV = "cuda"
print(f"ARM_EVAL_SERVER loaded (cured {CKPT})", flush=True)


@torch.no_grad()
def act(ob):
    img = np.ascontiguousarray(ob["image"]).astype(np.uint8)
    wrist = np.ascontiguousarray(ob.get("wrist", img)).astype(np.uint8)
    # ZERO_STATE (default on): the cured model's MolmoAct2-Think generation RUNS AWAY on real proprio state (real state
    # tokens -> non-terminating depth-reasoning AR = minutes/inference, GPU-idle+CPU-bound). Zero state = vision+language
    # grounding only (the claim; base convention is state-inert too) => fast + bounded generation. Verified: real-state
    # inference = runaway (431% CPU, GPU 0%); zero-state inference = 2.8s first / 0.03s buffered.
    if os.environ.get("ZERO_STATE", "1") == "1":
        state = np.zeros(8, np.float32)
    else:
        state = np.asarray(ob.get("state", np.zeros(8)), np.float32)[:8]
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


EP_SEED = 0
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args.port)); srv.listen(8)
print(f"ARM_EVAL_SERVER_READY port={args.port}", flush=True)
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
        print(f"[arm_srv] err {e}", flush=True)
    finally:
        conn.close()
