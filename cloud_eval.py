"""Evaluate the final policy of every skill on many episodes, on a Nebius Serverless Job (or locally).

Reads checkpoints/manifest.json, runs N parallel episodes per skill with different random walking commands,
and prints one line per skill starting with CLOUD_EVAL_RESULT (read back with `nebius ai job logs`).

    python cloud_eval.py --episodes 64 --horizon 500
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import json
import sys

import jax
import jax.numpy as jp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from skill_env import G1Skill
from rollout_skill import build


def evaluate(reward, checkpoint, episodes, horizon, seed=0):
    _, policy = build(reward, checkpoint)          # loads the trained policy
    env = G1Skill(reward)                          # batched env, same settings as training
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    pel_id, (lh_id, rh_id) = env._pelvis_imu_site_id, env._hands_site_id

    @jax.jit
    def features(d):
        pel = d.site_xpos[:, pel_id]
        R = d.site_xmat[:, pel_id].reshape(-1, 3, 3)
        local = lambda p: jp.einsum("nji,nj->ni", R, p - pel)
        return pel[:, 2], local(d.site_xpos[:, rh_id]), local(d.site_xpos[:, lh_id])

    rng = jax.random.PRNGKey(seed)
    state = reset(jax.random.split(rng, episodes))
    alive = np.ones(episodes, bool)
    steps_alive = np.zeros(episodes)
    sums = {"pelvis_height": 0.0, "right_hand_z": 0.0, "left_hand_z": 0.0, "hands_gap": 0.0}
    rh_max = np.full(episodes, -9.0)
    count = 0.0
    for _ in range(horizon):
        rng, k = jax.random.split(rng)
        act, _ = policy(state.obs, k)
        state = step(state, act)
        alive &= ~np.asarray(state.done).astype(bool)
        pz, rh, lh = (np.asarray(x) for x in features(state.data))
        w = alive.astype(float)
        steps_alive += w
        count += w.sum()
        sums["pelvis_height"] += (pz * w).sum()
        sums["right_hand_z"] += (rh[:, 2] * w).sum()
        sums["left_hand_z"] += (lh[:, 2] * w).sum()
        sums["hands_gap"] += (np.linalg.norm(rh - lh, axis=1) * w).sum()
        rh_max = np.where(alive, np.maximum(rh_max, rh[:, 2]), rh_max)
    mean = {k: round(v / max(count, 1), 3) for k, v in sums.items()}
    return {
        "episodes": episodes,
        "horizon_steps": horizon,
        "upright_rate": round(float(alive.mean()), 3),
        "mean_steps_survived": round(float(steps_alive.mean()), 1),
        "mean_pelvis_height_m": mean["pelvis_height"],
        "mean_right_hand_height_above_pelvis_m": mean["right_hand_z"],
        "mean_left_hand_height_above_pelvis_m": mean["left_hand_z"],
        "median_peak_right_hand_height_m": round(float(np.median(rh_max)), 3),
        "mean_distance_between_hands_m": mean["hands_gap"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--manifest", default=os.path.join(HERE, "checkpoints", "manifest.json"))
    args = ap.parse_args()
    print("devices:", jax.devices(), flush=True)
    manifest = json.load(open(args.manifest))
    for item in manifest:
        res = evaluate(os.path.join(HERE, item["reward"]), os.path.join(HERE, item["checkpoint"]),
                       args.episodes, args.horizon)
        res["skill"] = item["skill"]
        print("CLOUD_EVAL_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
