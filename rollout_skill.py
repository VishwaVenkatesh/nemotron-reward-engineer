"""Roll out a trained skill policy: video, key frames for the vision critic, and stats.

    python rollout_skill.py --reward reward.py --checkpoint <ckpt> --outdir runs/x/rollout
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import functools
import json
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skill_env import G1Skill


def build(reward_path, checkpoint):
    from mujoco_playground import wrapper
    from mujoco_playground.config import locomotion_params
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    env = G1Skill(reward_path, config_overrides={"naconmax": 2048, "njmax": 128})
    ppo_params = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
    tp = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in tp:
        del tp["network_factory"]
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory)
    tp.update(num_timesteps=0, num_evals=1, num_envs=1)
    tp.pop("num_eval_envs", None)
    make_inference_fn, params, _ = ppo.train(
        environment=env, network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=checkpoint, **tp)
    return env, jax.jit(make_inference_fn(params, deterministic=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    env, policy = build(args.reward, args.checkpoint)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    states, sums, fell = [state], {}, False
    hands = {"left": [], "right": []}  # hand positions in pelvis frame (x fwd, y left, z up)
    for _ in range(args.horizon):
        rng, k = jax.random.split(rng)
        act, _ = policy(state.obs, k)
        state = step(state, act)
        states.append(state)
        for name, v in state.metrics.items():
            if name.startswith("reward/"):
                sums[name[7:]] = sums.get(name[7:], 0.0) + float(v)
        d = state.data
        pel = np.asarray(d.site_xpos[env._pelvis_imu_site_id])
        R = np.asarray(d.site_xmat[env._pelvis_imu_site_id]).reshape(3, 3)
        for side, i in (("left", 0), ("right", 1)):
            hands[side].append(R.T @ (np.asarray(d.site_xpos[env._hands_site_id[i]]) - pel))
        if bool(state.done):
            fell = True
            break

    n = len(states) - 1
    hand_stats = {}
    for side, arr in hands.items():
        a = np.array(arr)
        hand_stats[side] = {
            "height_above_pelvis_mean": round(float(a[:, 2].mean()), 3),
            "height_above_pelvis_max": round(float(a[:, 2].max()), 3),
            "sideways_y_min": round(float(a[:, 1].min()), 3),
            "sideways_y_max": round(float(a[:, 1].max()), 3),
            "forward_x_mean": round(float(a[:, 0].mean()), 3),
        }
    summary = {
        "stayed_upright": not fell,
        "steps_survived": n,
        "horizon": args.horizon,
        "hands_in_pelvis_frame_m": hand_stats,
        "mean_reward_terms": {k: round(v / max(n, 1), 4) for k, v in sorted(sums.items())},
    }

    import mediapy
    try:
        frames = env.render(states[::2], height=360, width=480, camera="track")
    except Exception:
        frames = env.render(states[::2], height=360, width=480)
    mediapy.write_video(os.path.join(args.outdir, "rollout.mp4"), frames, fps=25)
    idx = np.linspace(0, len(frames) - 1, args.frames).astype(int)
    paths = []
    for i, j in enumerate(idx):
        p = os.path.join(args.outdir, f"frame_{i}.png")
        mediapy.write_image(p, frames[j])
        paths.append(p)
    summary["frames"] = paths

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
