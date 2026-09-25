"""c6_processor_shim.py — register a NO-OP molmoact2_action_frame_transform so release-v2's saved
post-processor (which lists this custom step, ABSENT from the a4f15bf fork) loads on this box.

Why a no-op is correct: a7 (a4f15bf-trained LIBERO, action-only, NO such step) grounds correctly, i.e. the
a4f15bf masked_unnormalizer already yields correct LIBERO OSC_POSE actions — so this custom step is inert here.
Validated empirically by action-grounding: with the no-op, C6 EEF reaches the named object to 0.3-1.7cm
(scene0 cream_cheese 0.9 / alphabet_soup 1.7; scene4 alphabet_soup 0.5 / butter 0.3) — precise, not garbage.

Import this module BEFORE lerobot make_pre_post_processors() (e.g. from patch_eval_server.py, alongside the
galahad_foresight import), so the registry has the step when it rebuilds the saved processor pipeline.
"""
from dataclasses import dataclass
from lerobot.processor import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register(name="molmoact2_action_frame_transform")
@dataclass
class NoOpActionFrameTransform(ProcessorStep):
    def __call__(self, transition):
        return transition

    def transform_features(self, features):
        return features


print("[c6_shim] registered no-op molmoact2_action_frame_transform", flush=True)
