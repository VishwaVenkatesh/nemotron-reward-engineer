"""Train a G1 whole-body LOCO-MANIPULATION policy.

Extends the open-source G1 joystick walking environment with a manipulation
objective: while tracking a walking command, the right hand must reach and hold a
target sampled in the pelvis frame. One policy coordinates legs, waist and arm --
the centrepiece of the thesis.

    conda activate g1
    PYTHONUNBUFFERED=1 python train_locomanip.py 2>&1 | tee train_locomanip.txt

Saves checkpoints under ~/logs/G1LocoManip-<timestamp>/checkpoints. Evaluate with
eval_locomanip.py / eval_stats.py.
"""
import argparse, csv, functools
from datetime import datetime

import jax
import jax.numpy as jp
from etils import epath

from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.locomotion.g1 import joystick as g1joy
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo


class G1LocoManip(g1joy.Joystick):
    """G1 joystick walking + a right-hand reach-and-hold objective."""

    def __init__(self, task="flat_terrain", config=None, config_overrides=None):
        if config is None:
            config = g1joy.default_config()
            config.reward_config.scales.hand_reach = 1.5
            config.njmax = 128
        super().__init__(task=task, config=config, config_overrides=config_overrides)
        self._reach_sigma = 0.05
        self._target_lo = jp.array([0.15, -0.35, -0.05])
        self._target_hi = jp.array([0.45, 0.05, 0.35])

    def _sample_target(self, rng):
        return jax.random.uniform(rng, (3,), minval=self._target_lo, maxval=self._target_hi)

    def _hand_and_target(self, data, info):
        offset = info.get("hand_target", jp.zeros(3))
        R = data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3)
        p_pelvis = data.site_xpos[self._pelvis_imu_site_id]
        target_world = p_pelvis + R @ offset
        hand_world = data.site_xpos[self._hands_site_id[1]]
        return R, hand_world, target_world, offset

    def reset(self, rng):
        rng, tkey = jax.random.split(rng)
        state = super().reset(rng)
        state.info["hand_target"] = self._sample_target(tkey)
        return state

    def _get_obs(self, data, info, contact):
        obs = super()._get_obs(data, info, contact)
        R, hand_world, target_world, offset = self._hand_and_target(data, info)
        err_local = R.T @ (hand_world - target_world)
        extra = jp.concatenate([offset, err_local])
        obs["state"] = jp.concatenate([obs["state"], extra])
        obs["privileged_state"] = jp.concatenate([obs["privileged_state"], extra])
        return obs

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        _, hand_world, target_world, _ = self._hand_and_target(data, info)
        dist2 = jp.sum(jp.square(hand_world - target_world))
        rewards["hand_reach"] = jp.exp(-dist2 / self._reach_sigma)
        return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000_000)
    args = ap.parse_args()

    ppo_params = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
    ppo_params.num_timesteps = args.timesteps
    training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in training_params:
        del training_params["network_factory"]
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory)
    num_eval_envs = ppo_params.get("num_eval_envs", 128)
    training_params.pop("num_eval_envs", None)

    env = G1LocoManip(task="flat_terrain")
    eval_env = G1LocoManip(task="flat_terrain")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = epath.Path(f"~/logs/G1LocoManip-{ts}").expanduser()
    ckpt = logdir / "checkpoints"; ckpt.mkdir(parents=True, exist_ok=True)
    print("Checkpoint path:", ckpt)

    rows = []
    def progress(num_steps, metrics):
        r = float(metrics.get("eval/episode_reward", float("nan")))
        rows.append((int(num_steps), r)); print(f"{num_steps}: reward={r:.3f}", flush=True)

    ppo.train(**training_params, network_factory=network_factory, seed=1,
              save_checkpoint_path=ckpt, wrap_env_fn=wrapper.wrap_for_brax_training,
              num_eval_envs=num_eval_envs, environment=env, eval_env=eval_env, progress_fn=progress)

    with open(logdir / "progress.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "reward"]); w.writerows(rows)
    print("Done. Checkpoints in:", ckpt)


if __name__ == "__main__":
    main()
