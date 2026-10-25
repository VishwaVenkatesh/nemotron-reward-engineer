"""Fine-tune the G1 policy on a generated reward, or just check the reward compiles.

    python train_skill.py --reward reward.py --check
    python train_skill.py --reward reward.py --restore <base_ckpt> --timesteps 20000000 --logdir runs/x
"""
import os
import argparse
import functools
import json
import sys

import jax
import jax.numpy as jp
from etils import epath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skill_env import G1Skill


def check(reward_path):
    """Build the env, reset and take one step under jit. Fails fast on bad generated code."""
    env = G1Skill(reward_path)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    terms = {k: float(v) for k, v in state.metrics.items() if k.startswith("reward/")}
    bad = [k for k, v in terms.items() if not jp.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite reward terms: {bad}")
    print(json.dumps({"ok": True, "terms": terms}))


def probe(reward_path):
    """Measure the robot's body geometry at rest, in the pelvis frame, so the LLM writes realistic targets."""
    import mujoco
    import numpy as np
    env = G1Skill(reward_path)
    d = jax.jit(env.reset)(jax.random.PRNGKey(0)).data
    m = env.mj_model
    pel = np.asarray(d.site_xpos[env._pelvis_imu_site_id])
    R = np.asarray(d.site_xmat[env._pelvis_imu_site_id]).reshape(3, 3)
    loc = lambda p: [round(float(x), 3) for x in R.T @ (np.asarray(p) - pel)]
    out = {"pelvis_height_above_ground": round(float(pel[2]), 3),
           "left_hand_at_rest": loc(d.site_xpos[env._hands_site_id[0]]),
           "right_hand_at_rest": loc(d.site_xpos[env._hands_site_id[1]])}
    for i in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if any(k in name for k in ("shoulder_pitch", "elbow", "torso")):
            out[name] = loc(d.xpos[i])
    sh = [i for i in range(m.nbody) if "right_shoulder_pitch" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")]
    if sh:
        arm = float(np.linalg.norm(np.asarray(d.site_xpos[env._hands_site_id[1]]) - np.asarray(d.xpos[sh[0]])))
        out["right_arm_length_shoulder_to_hand"] = round(arm, 3)
        out["max_right_hand_height_above_pelvis"] = round(loc(d.xpos[sh[0]])[2] + arm, 3)
    out["approx_head_top_above_pelvis"] = round(1.32 - float(pel[2]), 3)  # G1 standing height ~1.32 m
    print(json.dumps(out))


def latest_checkpoint(ckpt_dir):
    steps = sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit())
    return steps[-1] if steps else None


def train(reward_path, restore, timesteps, logdir):
    from mujoco_playground import wrapper
    from mujoco_playground.config import locomotion_params
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    ppo_params = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
    ppo_params.num_timesteps = timesteps
    tp = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in tp:
        del tp["network_factory"]
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory)
    num_eval_envs = tp.pop("num_eval_envs", 128)

    logdir = epath.Path(os.path.abspath(os.path.expanduser(logdir)))
    ckpt = logdir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    def progress(num_steps, metrics):
        r = float(metrics.get("eval/episode_reward", float("nan")))
        print(f"{num_steps}: reward={r:.3f}", flush=True)

    try:
        ppo.train(**tp, network_factory=network_factory, seed=1,
                  save_checkpoint_path=ckpt, restore_checkpoint_path=restore,
                  wrap_env_fn=wrapper.wrap_for_brax_training, num_eval_envs=num_eval_envs,
                  environment=G1Skill(reward_path), eval_env=G1Skill(reward_path),
                  progress_fn=progress)
    except Exception as e:  # e.g. GPU out of memory in the final eval
        print(f"WARNING: training stopped early: {type(e).__name__}: {str(e)[:300]}", flush=True)

    last = latest_checkpoint(ckpt)
    if last is None:
        raise RuntimeError("no checkpoint was saved")
    print(f"CKPT={last}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--probe", action="store_true", help="print body geometry at rest as JSON")
    ap.add_argument("--restore", default=None, help="base checkpoint to fine-tune from")
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--logdir", default="runs/manual")
    args = ap.parse_args()
    if args.probe:
        probe(args.reward)
    elif args.check:
        check(args.reward)
    else:
        train(args.reward, args.restore, args.timesteps, args.logdir)


if __name__ == "__main__":
    main()
