"""galahad_foresight.py — C2: the ISOLATED foresight branch for Galahad-0 (Route A), by monkey-patch.
Import BEFORE lerobot_train.main(). Env-gated by GALAHAD_FS=1 (else fully inert).

ARCHITECTURE (Seer/DreamVLA pattern, adapted to MolmoAct2's per-layer-interleaved trunk):
  MolmoAct2 runs `hidden_states` (VLM stream: image+text) and `action_hidden` (action stream) TOGETHER per layer;
  the action expert cross-attends action_hidden -> hidden_states each layer.
  We APPEND M learnable foresight tokens to the END of the VLM stream. The block-mask spec then holds BY CONSTRUCTION:
    * action CAN see foresight   — the expert cross-attends the whole hidden_states, which now contains them.  ✓
    * foresight CANNOT see action — the VLM stream never attends to action_hidden.                              ✓
    * image/text are NOT perturbed by foresight — causal mask + foresight appended LAST ⇒ nothing before sees them. ✓
  After the layers, hidden_states[:, -M:] = the foresight representation -> two small heads:
    region head: (G,G) logits  = WHICH object's region is occupied at t+K   <-- the grounding-carrying target
    depth  head: (G,G)         = agentview depth at t+K

GT (LAB B25 — do NOT revert to a frame-diff): region GT = the NAMED object's instance-seg mask @ t+K.
  A frame-diff GT is 52.5% ARM / 44.4% target (render-confirmed) ⇒ it teaches "predict where the arm sweeps",
  which is instruction-INDEPENDENT and would gut the "prediction = a 2nd grounding trainer" core + fake a null C5.
  The region is sparse (~4.8% of the grid) ⇒ BCE needs pos_weight or the head collapses to all-zeros.

ENV:
  GALAHAD_FS=1            enable
  GALAHAD_FS_DEBUG=1      print shapes on step 0 and DO NOT add the loss (contract check first)
  GALAHAD_FS_STAGE=1|2    1 = obs-first (action loss OFF, foresight only)  2 = joint (both on)   [B24 law 2]
  GALAHAD_FS_W=0.05       foresight loss weight alpha (B24 law 3: start low, then rebalance)
  GALAHAD_FS_M=8          number of foresight tokens
  GALAHAD_FS_G=16         head output grid (matches the converter)
  GALAHAD_FS_POSW=15      BCE pos_weight for the sparse region target
  GALAHAD_FS_GRADLOG=100  every N steps, measure per-branch grad norms on a shared trunk param (B24 law 3)
"""
import os, math, torch, torch.nn as nn, torch.nn.functional as F

ON = os.environ.get("GALAHAD_FS", "") == "1"
DEBUG = os.environ.get("GALAHAD_FS_DEBUG", "") == "1"
STAGE = int(os.environ.get("GALAHAD_FS_STAGE", "2"))
W = float(os.environ.get("GALAHAD_FS_W", "0.05"))
M = int(os.environ.get("GALAHAD_FS_M", "8"))
G = int(os.environ.get("GALAHAD_FS_G", "16"))
POSW = float(os.environ.get("GALAHAD_FS_POSW", "15"))
GRADLOG = int(os.environ.get("GALAHAD_FS_GRADLOG", "100"))
OBS_FIRST = int(os.environ.get("GALAHAD_FS_OBS_FIRST", "0"))
INFER = os.environ.get("GALAHAD_FS_INFER", "") == "1"
CAPTURE = os.environ.get("GALAHAD_FS_CAPTURE", "") == "1"
# C5-fix (law 1): action shares the trunk but does NOT read the foresight tokens.
# 🔴 THIS FILE IS A RECOVERED PRE-RENAME SNAPSHOT (2026-07-21). The surviving `c5_spatial_eval.sh` / `c5_spatial_fsdebug.sh`
# drive the masked arm with `GALAHAD_ERVLA=1`, i.e. the var was renamed AFTER this snapshot was taken. Accepting BOTH
# names is not cosmetic: with only the old name, those scripts run the masked arm SILENTLY UNMASKED — a mislabelled
# ablation, which is the same class of error as shipping an action-only LoRA as the dual-branch flagship.
MASK_ACT = (os.environ.get("GALAHAD_FS_MASK_ACT", "") == "1") or (os.environ.get("GALAHAD_ERVLA", "") == "1")
_GF = {"mod": None}   # >0: action loss OFF for the first N steps (obs-first staging, B24 law 2), then ON. 0 = joint-from-scratch (the compute-matched C5 ablation).
_S = {"step": 0}


class ForesightHeads(nn.Module):
    """M learnable tokens + 2 small decoders. Kept tiny (Seer: 65M-trainable head is enough at 24GB)."""

    def __init__(self, d_model: int, m: int, g: int):
        super().__init__()
        self.m, self.g = m, g
        self.tokens = nn.Parameter(torch.randn(m, d_model) * 0.02)
        h = 512
        self.region = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, g * g))
        self.depth = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, g * g))

    def decode(self, fs):                      # fs: (B, M, D) bf16 -> pooled (B, D)
        # the trunk runs bf16 but the heads stay fp32 (BCE on a sparse target is numerically happier in fp32);
        # the cast is differentiable, so grads flow back into the bf16 trunk correctly.
        w = next(self.region.parameters())
        p = fs.mean(1).to(w.dtype)
        return self.region(p).view(-1, self.g, self.g), self.depth(p).view(-1, self.g, self.g)


def patch_all():
    if not ON:
        return
    from lerobot.policies.molmoact2 import modeling_molmoact2 as LW
    P = LW.MolmoAct2Policy
    _orig_init = P.__init__
    _orig_prep = P._prepare_joint_training_backbone_inputs
    _orig_flow = P._compute_flow_matching_loss_joint_per_layer
    _orig_fwd = P.forward

    # ---- 1) heads must exist BEFORE lerobot builds the optimizer from policy.parameters() ----
    def __init__(self, *a, **kw):
        _orig_init(self, *a, **kw)
        d = None
        for attr in ("hidden_size", "d_model"):
            cfg = getattr(getattr(self, "config", None), attr, None)
            if isinstance(cfg, int):
                d = cfg; break
        if d is None:                                        # fall back: read it off the embedding matrix
            for p in self.parameters():
                if p.dim() == 2 and p.shape[0] > 1000:
                    d = int(p.shape[1]); break
        self._gf = ForesightHeads(d, M, G)
        _GF["mod"] = self._gf                       # global handle: the BACKBONE (not the policy) owns the inference path
        for p in self._gf.parameters():
            p.requires_grad_(True)
        print(f"[GALAHAD_FS] heads attached d_model={d} M={M} G={G} "
              f"params={sum(p.numel() for p in self._gf.parameters())/1e6:.1f}M", flush=True)
    P.__init__ = __init__

    # ---- 2) append the foresight tokens to the VLM stream; rebuild the mask with the MODEL'S OWN builder ----
    def _prep(self, model_inputs):
        embeds, mask_map, pos, cache = _orig_prep(self, model_inputs)
        gf = getattr(self, "_gf", None)
        if gf is None:
            return embeds, mask_map, pos, cache
        B, S, D = embeds.shape
        ftok = gf.tokens.to(device=embeds.device, dtype=embeds.dtype).unsqueeze(0).expand(B, gf.m, D)
        embeds2 = torch.cat([embeds, ftok], dim=1)                       # (B, S+M, D) — appended LAST
        am = model_inputs.get("attention_mask")
        tti = model_inputs.get("token_type_ids")
        am2 = torch.cat([am, am.new_ones(B, gf.m)], 1) if (am is not None and not isinstance(am, dict)) else None
        tti2 = torch.cat([tti, tti.new_zeros(B, gf.m)], 1) if tti is not None else None   # 0 = text-like ⇒ causal
        backbone = self._backbone()
        mask2 = backbone._build_native_attention_bias(
            inputs_embeds=embeds2, attention_mask=am2, token_type_ids=tti2, past_key_values=None)
        cache2 = torch.arange(0, S + gf.m, device=embeds.device)
        pos2 = cache2.unsqueeze(0)
        if _S["step"] == 0:
            print(f"[GALAHAD_FS] seq {S} -> {S + gf.m} (+{gf.m} foresight tokens); mask keys={list(mask2)[:4]}", flush=True)
        return embeds2, mask2, pos2, cache2
    P._prepare_joint_training_backbone_inputs = _prep

    # ---- 2b) the ACTION EXPERT has its OWN cross-attn key mask, built from the ORIGINAL input_ids/attention_mask
    #          (-> (B,1,1,S)). Extending only the VLM mask left it at S while k_ctx/v_ctx grew to S+M:
    #          "expanded size (478) must match existing size (470) at dim 3". Append M VALID keys — the foresight
    #          tokens are exactly the keys the action stream MUST attend to (this IS "action can see foresight"). ----
    _orig_encmask = P._encoder_attention_mask_for_action_expert

    def _encmask(self, *, input_ids=None, attention_mask=None):
        m = _orig_encmask(self, input_ids=input_ids, attention_mask=attention_mask)
        gf = getattr(self, "_gf", None)
        if gf is None or m is None:
            return m
        B = m.shape[0]
        fill = 0.0 if MASK_ACT else 1.0                                   # C5-fix: 0 = action CANNOT read foresight
        extra = torch.full((B, gf.m), fill, dtype=m.dtype, device=m.device)
        m2 = torch.cat([m, extra], dim=1)
        if _S["step"] == 0:
            print(f"[GALAHAD_FS] action-expert cross-attn key mask {tuple(m.shape)} -> {tuple(m2.shape)}  "
                  f"foresight-keys={'MASKED(law-1 isolation)' if MASK_ACT else 'readable'}", flush=True)
        return m2
    P._encoder_attention_mask_for_action_expert = _encmask

    # ---- 3) stash the trunk output so forward() can read the foresight slots ----
    def _flow(self, **kw):
        loss, hidden = _orig_flow(self, **kw)
        self._gf_hidden = hidden
        return loss, hidden
    P._compute_flow_matching_loss_joint_per_layer = _flow

    # ---- 4) decode + loss ----
    def forward(self, batch, *a, **kw):
        out = _orig_fwd(self, batch, *a, **kw)
        act_loss = out[0] if isinstance(out, tuple) else out
        gf, hid = getattr(self, "_gf", None), getattr(self, "_gf_hidden", None)
        if gf is None or hid is None:
            return out
        fs = hid[:, -gf.m:, :]                                            # (B, M, D) = the foresight slots
        rlog, dpred = gf.decode(fs)                                       # (B,G,G), (B,G,G)
        # The GT columns MUST reach here through lerobot_train's `batch = preprocessor(batch)` (PolicyProcessorPipeline),
        # which keeps only the structured slots -> they carry the `observation.` prefix (a bare `foresight.*` column
        # exists in the dataset but is DROPPED before the policy sees it; `observation.role_frame` survived this same
        # path in the coupling run). Search by substring so either naming works.
        def _find(sub):
            k = next((k for k in batch if "foresight" in k and sub in k), None)
            return batch[k] if k else None
        rgt, dgt, vld = _find("region"), _find("depth"), _find("valid")
        if _S["step"] == 0:
            print(f"[GALAHAD_FS] hidden={tuple(hid.shape)} foresight_slots={tuple(fs.shape)} "
                  f"region_logits={tuple(rlog.shape)} region_gt={None if rgt is None else tuple(rgt.shape)} "
                  f"valid={None if vld is None else tuple(vld.shape)} act_loss={float(act_loss):.4f}", flush=True)
            if rgt is None:
                print(f"[GALAHAD_FS] 🔴 foresight GT NOT in the policy batch. batch keys = {sorted(batch.keys())}", flush=True)
            else:
                print(f"[GALAHAD_FS] region_gt raw stats: min={float(rgt.min()):.4f} max={float(rgt.max()):.4f} "
                      f"mean={float(rgt.mean()):.4f} n_unique={int(torch.unique(rgt).numel())}", flush=True)
                print(f"[GALAHAD_FS] depth_gt  raw stats: min={float(dgt.min()):.4f} max={float(dgt.max()):.4f} "
                      f"mean={float(dgt.mean()):.4f}", flush=True)
                # is a normalizer touching it? (the `observation.` prefix that gets it THROUGH the preprocessor
                # can also get it NORMALIZED -> the binary target would be destroyed)
                for nm in ("normalize_inputs", "normalize_targets", "_normalize_inputs"):
                    n = getattr(self, nm, None)
                    if n is not None:
                        ks = [k for k, _ in n.named_buffers()] if hasattr(n, "named_buffers") else []
                        fk = [k for k in ks if "foresight" in k]
                        print(f"[GALAHAD_FS] {nm}: has_foresight_buffers={bool(fk)} {fk[:4]}", flush=True)
            if rgt is not None:
                print(f"[GALAHAD_FS] region_gt positive-fraction={float(rgt.mean()):.4f} (expect ~0.05 = a compact "
                      f"object blob; ~0.5 would mean the GT is the arm/scene ⇒ STOP, see LAB B25)", flush=True)
        _S["step"] += 1
        if DEBUG or rgt is None or dgt is None:
            return out                                                    # contract check only, no loss added
        rgt = rgt.to(rlog.dtype).view(-1, gf.g, gf.g)
        dgt = dgt.to(dpred.dtype).view(-1, gf.g, gf.g)
        m = (vld.view(-1, 1, 1).to(rlog.dtype) if vld is not None else torch.ones_like(rgt[:, :1, :1]))
        pw = torch.tensor(POSW, device=rlog.device, dtype=rlog.dtype)
        l_reg = (F.binary_cross_entropy_with_logits(rlog, rgt, pos_weight=pw, reduction="none") * m).sum() / m.sum().clamp_min(1) / (gf.g * gf.g)
        l_dep = ((dpred - dgt).abs() * m).sum() / m.sum().clamp_min(1) / (gf.g * gf.g)
        fs_loss = l_reg + 0.5 * l_dep
        # B24 law 2 (obs-prediction BEFORE action SFT, pred-loss KEPT ON after): run it as ONE continuous job with a
        # staged loss schedule instead of stage1 -> checkpoint -> stage2-resume. The foresight heads are params added by
        # this patch; a checkpoint round-trip risks silently re-initialising them (losing all of stage 1). A schedule
        # cannot lose them. OBS_FIRST=0 => joint-from-scratch = the COMPUTE-MATCHED C5 ablation arm.
        if OBS_FIRST > 0:
            aw = 0.0 if _S["step"] <= OBS_FIRST else 1.0
        else:
            aw = 0.0 if STAGE == 1 else 1.0
        total = aw * act_loss + W * fs_loss
        if OBS_FIRST > 0 and _S["step"] == OBS_FIRST + 1:
            print(f"[GALAHAD_FS] === obs-first phase over at step {OBS_FIRST}: action loss ON, foresight loss STAYS on "
                  f"(B24 law 2) ===", flush=True)
        if _S["step"] % 25 == 1:
            with torch.no_grad():
                p = torch.sigmoid(rlog)
                iou = ((p > 0.5) & (rgt > 0.5)).sum().float() / (((p > 0.5) | (rgt > 0.5)).sum().float().clamp_min(1))
            print(f"[GALAHAD_FS] step={_S['step']} stage={STAGE} act={float(act_loss):.4f} "
                  f"reg={float(l_reg):.4f} dep={float(l_dep):.4f} regionIoU={float(iou):.3f} total={float(total):.4f}", flush=True)
        # B24 law 3: token/gradient mass imbalance crushes actions by default -> MEASURE per-branch grad norms
        if GRADLOG and _S["step"] % GRADLOG == 1 and aw > 0:
            try:
                shared = [p for p in self._backbone().transformer.parameters() if p.requires_grad][:6]
                if shared:
                    ga = torch.autograd.grad(act_loss, shared, retain_graph=True, allow_unused=True)
                    gf_ = torch.autograd.grad(W * fs_loss, shared, retain_graph=True, allow_unused=True)
                    na = math.sqrt(sum(float((g ** 2).sum()) for g in ga if g is not None))
                    nf = math.sqrt(sum(float((g ** 2).sum()) for g in gf_ if g is not None))
                    print(f"[GALAHAD_FS] GRADNORM action={na:.3e} foresight={nf:.3e} ratio={nf / max(na, 1e-12):.3f} "
                          f"(B24 law 3: rebalance GALAHAD_FS_W so ratio ~= 1)", flush=True)
            except Exception as e:
                print(f"[GALAHAD_FS] gradlog skipped: {e}", flush=True)
        return (total,) + out[1:] if isinstance(out, tuple) else total
    P.forward = forward

    # ---- 5) 🔴 TRAIN<->EVAL CONTRACT. The training path appends the foresight tokens via
    #      _prepare_joint_training_backbone_inputs, but INFERENCE never touches that function — it goes
    #      MolmoAct2Policy.predict_action_chunk -> backbone.generate_actions_from_inputs -> self(input_ids=...).
    #      So an unpatched eval feeds the action expert 470 cross-attn keys when it was TRAINED on 478.
    #      That is a train/eval mismatch of exactly the class that has burned this repo (the 44%->63% chunk lesson),
    #      and it would silently under-report Galahad => a FALSE NEGATIVE on C5.
    #      Injection point: generate_actions_from_inputs SKIPS its own VLM forward if we hand it encoder_kv_states.
    #      GALAHAD_FS_INFER=1  -> the TRUE model (tokens present, as trained)
    #      GALAHAD_FS_INFER=0  -> tokens absent = exactly the pre-registered C4-iv foresight-off ablation.
    #      The DIFFERENCE between the two IS the answer to "is the prediction branch load-bearing at inference?"
    if INFER:
        B_ = LW.MolmoAct2Policy._backbone(LW.MolmoAct2Policy) if False else None  # noqa (kept for clarity)
        import lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 as HFM
        BK = None
        for nm in dir(HFM):
            o = getattr(HFM, nm)
            if isinstance(o, type) and hasattr(o, "generate_actions_from_inputs"):
                BK = o; break
        if BK is None:
            print("[GALAHAD_FS] 🔴 could not locate the backbone class — inference tokens NOT appended", flush=True)
        else:
            _orig_gen = BK.generate_actions_from_inputs

            def _gen(self, *, input_ids, pixel_values=None, image_token_pooling=None, image_grids=None,
                     image_num_crops=None, pixel_values_videos=None, video_token_pooling=None, video_grids=None,
                     attention_mask=None, token_type_ids=None, encoder_kv_states=None,
                     encoder_attention_mask=None, **kw):
                gf = _GF["mod"]
                if gf is None or encoder_kv_states is not None:
                    return _orig_gen(self, input_ids=input_ids, pixel_values=pixel_values,
                                     image_token_pooling=image_token_pooling, image_grids=image_grids,
                                     image_num_crops=image_num_crops, pixel_values_videos=pixel_values_videos,
                                     video_token_pooling=video_token_pooling, video_grids=video_grids,
                                     attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     encoder_kv_states=encoder_kv_states,
                                     encoder_attention_mask=encoder_attention_mask, **kw)
                try:
                    images, tp = self.merge_visual_inputs(
                        input_ids=input_ids, pixel_values=pixel_values, image_token_pooling=image_token_pooling,
                        image_grids=image_grids, image_num_crops=image_num_crops,
                        pixel_values_videos=pixel_values_videos, video_token_pooling=video_token_pooling,
                        video_grids=video_grids)
                    embeds, _ = self.build_input_embeddings(input_ids, images, tp)
                    B, S, D = embeds.shape
                    ftok = gf.tokens.to(device=embeds.device, dtype=embeds.dtype).unsqueeze(0).expand(B, gf.m, D)
                    embeds2 = torch.cat([embeds, ftok], dim=1)
                    am2 = torch.cat([attention_mask, attention_mask.new_ones(B, gf.m)], 1) if attention_mask is not None else None
                    tti2 = torch.cat([token_type_ids, token_type_ids.new_zeros(B, gf.m)], 1) if token_type_ids is not None else None
                    out = self(inputs_embeds=embeds2, attention_mask=am2, token_type_ids=tti2, use_cache=True)
                    if CAPTURE:
                        _h = getattr(out, "last_hidden_state", None)
                        if _h is None and isinstance(out, (tuple, list)):
                            _h = out[0]
                        if _h is None:
                            if not _S.get("cap_fail"):
                                _S["cap_fail"] = True
                                print("[GALAHAD_FS] 🔴🔴 C4 CAPTURE FAILED: no last_hidden_state on the trunk output. "
                                      "The prediction face CANNOT be scored — do NOT report a C4 number.", flush=True)
                        else:
                            with torch.no_grad():
                                _r, _d = gf.decode(_h[:, -gf.m:, :].to(embeds.dtype))
                            _GF["region"] = _r.detach().float().cpu().numpy()
                            _GF["depth"] = _d.detach().float().cpu().numpy()
                            if not _S.get("cap_ok"):
                                _S["cap_ok"] = True
                                _pf = float(torch.sigmoid(_r).mean())
                                _flag = "" if 0.005 < _pf < 0.35 else "  🔴 OUT OF RANGE (trained head emits ~0.05)"
                                print(f"[GALAHAD_FS] C4 CAPTURE ok: hidden {tuple(_h.shape)} -> region "
                                      f"{tuple(_r.shape)}; sigmoid-mean={_pf:.4f}{_flag}", flush=True)
                    kv = self._extract_kv_states(out.past_key_values)
                    eam = self._get_encoder_attention_mask(input_ids, attention_mask)
                    _fk = eam.new_zeros(B, gf.m) if MASK_ACT else eam.new_ones(B, gf.m)  # C5-fix: mask at infer too
                    eam2 = torch.cat([eam, _fk], 1) if eam is not None else None
                    # 🔴 input_ids MUST grow too. Downstream, _depth_gate_from_condition builds
                    #    depth_mask = _get_depth_token_mask(input_ids, ...)  -> length S (470), which is then applied
                    #    to the layer KV states (S+M = 478) => "size of tensor a (470) must match tensor b (478)".
                    #    Pad with the FIRST token's id (BOS/system): it is guaranteed NOT the depth token, so the 8
                    #    foresight positions are correctly excluded from the depth gate.
                    ids2 = torch.cat([input_ids, input_ids[:, :1].expand(B, gf.m)], dim=1)
                    if not _S.get("infer_ok"):
                        _S["infer_ok"] = True
                        print(f"[GALAHAD_FS] INFERENCE action-expert foresight-keys="
                              f"{'MASKED(law-1)' if MASK_ACT else 'readable'}", flush=True)
                        print(f"[GALAHAD_FS] INFERENCE: seq {S} -> {S + gf.m} (+{gf.m} foresight tokens); "
                              f"action-expert keys {tuple(eam.shape)} -> {tuple(eam2.shape)}; "
                              f"input_ids {tuple(input_ids.shape)} -> {tuple(ids2.shape)}  [train/eval contract MATCHED]", flush=True)
                    return _orig_gen(self, input_ids=ids2, attention_mask=am2,
                                     token_type_ids=tti2, encoder_kv_states=kv,
                                     encoder_attention_mask=eam2, **kw)
                except Exception as e:
                    if not _S.get("infer_fail"):
                        _S["infer_fail"] = True
                        print(f"[GALAHAD_FS] 🔴🔴 INFERENCE token-append FAILED ({type(e).__name__}: {str(e)[:90]}) "
                              f"-> FALLING BACK to the no-token path. THIS EVAL IS THE FORESIGHT-OFF ABLATION, "
                              f"NOT the true model. Do NOT report it as Galahad.", flush=True)
                    return _orig_gen(self, input_ids=input_ids, pixel_values=pixel_values,
                                     image_token_pooling=image_token_pooling, image_grids=image_grids,
                                     image_num_crops=image_num_crops, pixel_values_videos=pixel_values_videos,
                                     video_token_pooling=video_token_pooling, video_grids=video_grids,
                                     attention_mask=attention_mask, token_type_ids=token_type_ids, **kw)
            BK.generate_actions_from_inputs = _gen
            print(f"[GALAHAD_FS] inference token-append PATCHED on {BK.__name__}", flush=True)

    print(f"[GALAHAD_FS] patched (DEBUG={DEBUG} STAGE={STAGE} OBS_FIRST={OBS_FIRST} INFER={INFER} W={W} M={M} G={G} POSW={POSW})", flush=True)


patch_all()
