import json, dataclasses, shutil, os
import lerobot.policies.molmoact2.configuration_molmoact2 as MC
valid = {f.name for f in dataclasses.fields(MC.MolmoAct2Config)}
p = "/root/galahad-release-v2/config.json"
if not os.path.exists(p + ".relv2orig"): shutil.copy(p, p + ".relv2orig")
c = json.load(open(p))
dropped = sorted(set(c) - valid)
newc = {k: v for k, v in c.items() if k in valid}
newc["type"] = "molmoact2"
newc["train_mode_vlm"] = "freeze"     # merged single-weight, no LoRA
newc["use_peft"] = False
newc["checkpoint_path"] = "/root/galahad-release-v2"
newc["action_mode"] = "continuous"; newc["inference_action_mode"] = "continuous"
json.dump(newc, open(p, "w"), indent=2)
moved = []
for f in ["policy_preprocessor.json", "policy_postprocessor.json"]:
    fp = "/root/galahad-release-v2/" + f
    if os.path.exists(fp): os.rename(fp, fp + ".bak"); moved.append(f)
print("RELV2_CONVERTED dropped=%s moved=%s baked_norm=%s" % (dropped, moved, any("stat" in str(v).lower() for v in c.get("input_features", {}).values()) if isinstance(c.get("input_features"), dict) else "?"))
