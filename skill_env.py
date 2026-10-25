"""G1Skill: your G1 loco-manipulation env + reward terms written by Nemotron.

The generated reward module must define:
    SCALES = {"term": weight, ...}
    def extra_rewards(env, data, action, info) -> {"term": scalar, ...}
"""
import importlib.util
import os
import sys

import jax.numpy as jp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_locomanip import G1LocoManip  # your thesis env (copied into this repo)
from mujoco_playground._src.locomotion.g1 import joystick as g1joy


def load_reward_module(path):
    spec = importlib.util.spec_from_file_location("generated_reward", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not isinstance(getattr(mod, "SCALES", None), dict):
        raise ValueError("reward module must define SCALES as a dict")
    if not callable(getattr(mod, "extra_rewards", None)):
        raise ValueError("reward module must define extra_rewards(env, data, action, info)")
    return mod


class G1Skill(G1LocoManip):
    def __init__(self, reward_path, task="flat_terrain", config_overrides=None):
        self._rm = load_reward_module(reward_path)
        config = g1joy.default_config()
        config.reward_config.scales.hand_reach = 1.5
        config.njmax = 128
        for k, v in self._rm.SCALES.items():
            config.reward_config.scales[k] = float(v)
        self._extra_keys = set(self._rm.SCALES.keys())
        super().__init__(task=task, config=config, config_overrides=config_overrides)

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        extra = self._rm.extra_rewards(self, data, action, info)
        unknown = set(extra) - self._extra_keys
        if unknown:
            raise KeyError(f"extra_rewards returned terms missing from SCALES: {sorted(unknown)}")
        for k, v in extra.items():
            rewards[k] = jp.asarray(v, dtype=jp.float32).reshape(())
        return rewards
