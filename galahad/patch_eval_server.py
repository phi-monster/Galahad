"""patch_eval_server.py — make arm_eval_server load the release-v2 DUAL-FACE (foresight) model for C6.

Inserts, BEFORE make_policy(), two imports:
  1. c6_processor_shim         -> registers the no-op molmoact2_action_frame_transform (absent in a4f15bf)
  2. galahad_foresight; patch_all()  -> monkey-patches the MolmoAct2 trunk with the +8 foresight tokens +
                                         region/depth heads (env-gated by GALAHAD_FS=1); GALAHAD_FS_CAPTURE=1
                                         stashes _GF['region'] which add_region_forward.py then sends over the socket.

galahad_foresight.py ships INSIDE the release-v2 HF repo (phi-monster/galahad-release-v2). On the a4f15bf fork it
needs one edit: its `import lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2` -> `...hf_model...`
(the backbone module was renamed molmoact2_hf_model -> hf_model). Idempotent; mirrors add_region_forward.py.

Then run add_region_forward.py (region over the socket). Server env for the dual-face:
  GALAHAD_FS=1 GALAHAD_FS_INFER=1 GALAHAD_FS_CAPTURE=1 GALAHAD_FS_M=8 GALAHAD_FS_G=16 ARM=A ZERO_STATE=1
release-v2 config must be: train_mode_vlm=freeze, use_peft=false, checkpoint_path=<base MolmoAct2-Think-LIBERO>
(NOT release-v2 itself — base VLM loads first, then release-v2's merged model.safetensors overlays); its saved
policy_pre/postprocessor.json RESTORED (baked norm). C7 tax = flip GALAHAD_FS_INFER 1<->0 (tokens on/off).
"""
import os
import sys

P = os.environ.get("ARM_EVAL_SERVER", "/root/c6/arm_eval_server.py")
s = open(P).read()
if "galahad_foresight.patch_all" in s:
    print("already patched")
    sys.exit(0)
a = "policy = make_policy(cfg=cfg, ds_meta=dsm)"
assert a in s, "make_policy anchor missing in " + P
ins = ("import c6_processor_shim  # no-op molmoact2_action_frame_transform\n"
       "import galahad_foresight; galahad_foresight.patch_all()  # C6 foresight branch\n")
s = s.replace(a, ins + a, 1)
open(P, "w").write(s)
print("arm_eval_server patched (c6_processor_shim + galahad_foresight)")
