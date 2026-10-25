# Nemotron Reward Engineer

**Type a skill in plain English. An AI agent writes the reinforcement-learning reward, trains a Unitree G1 humanoid, watches the result, and rewrites the reward until the robot gets it right.**

Built for the **Nebius × NVIDIA Global AI Hackathon 2026 — Physical AI track**.

![Round 1 vs final round](docs/before_after.gif)

▶ **Demo page:** https://vishwavenkatesh.github.io/nemotron-reward-engineer/
▶ **Video:** _YouTube link_

---

## Why

Reward design is the slowest part of robot learning. A researcher writes a reward, trains for an hour, watches the video, sees the robot doing something silly, and edits the reward again. This project hands that whole loop to NVIDIA Nemotron models: one writes the code, one watches the robot, and the loop runs on its own.

## How it works

```
"raise the right hand above the head and wave while walking"
        │
        ▼
Nemotron 3 Ultra ── writes a JAX reward module (SCALES + extra_rewards)
        │             └─ compile check in the real env; errors go back to Ultra for auto-repair
        ▼
PPO fine-tune in MuJoCo MJX (Brax), starting from a G1 walk + reach policy
        │
        ▼
Rollout ── video, 6 key frames, measured hand positions, per-term reward means
        │
        ▼
Nemotron 3.5 Super-VL ── watches the frames + numbers, scores 0–10, explains what is wrong
        │
        └──► back to Ultra, which rewrites the reward. The next round continues from the best checkpoint so far.
```

Design details that made it work:

- **Grounded geometry.** Before writing any reward, the agent measures the robot's body in simulation (shoulder height, arm length, highest reachable hand height, head height) and gives these numbers to Ultra. Without them, Ultra guessed targets that were only shoulder-high.
- **Numbers + vision for the critic.** The robot often faces the camera, so left and right are mirrored in the image. The critic receives exact hand positions from the simulator alongside the frames.
- **Self-repair.** Generated code is compiled in the real environment before any GPU time is spent. Compile errors go back to Ultra (e.g. it once used `jp.sigmoid`, which does not exist, and fixed it on the next attempt).
- **Curriculum by checkpoint chaining.** Each round fine-tunes from the best round so far, so progress accumulates instead of restarting.

## NVIDIA models used (all through Nebius Token Factory)

| Model | Role |
|---|---|
| `nvidia/Nemotron-3-Ultra-550b-a55b` | Writes and repairs the reward code; reasons about the critic's feedback |
| `nvidia/Nemotron-3.5-Super-VL` | Vision critic: watches rollout frames and scores the behaviour |

Token Factory's OpenAI-compatible API meant the whole agent is ~250 lines with the standard `openai` client: one base URL, two model names, and a vision model that accepts base64 frames directly.

## Starting policy

The agent fine-tunes my own G1 whole-body loco-manipulation policy (PPO, MuJoCo Playground / MJX): one policy that walks on command while the right hand reaches and holds a target, trained on a single 6 GB laptop GPU. The env class is in `train_locomanip.py`; full training and evaluation code: [g1-whole-body-loco-manipulation](https://github.com/VishwaVenkatesh/g1-whole-body-loco-manipulation).

## Run it

Requirements: Linux, NVIDIA GPU, a Nebius Token Factory API key, and a trained G1 checkpoint from `train_locomanip.py`.

```bash
conda create -n g1 python=3.11 -y && conda activate g1
pip install -r requirements.txt
echo "NEBIUS_API_KEY=..." > .env

# optional: train the starting walk + reach policy (~200M steps)
python train_locomanip.py

# run the agent
python agent.py --task "raise the right hand above the head and wave it side to side while walking" \
    --base <path/to/checkpoint> --iters 3 --timesteps 20000000

# build the demo page from one or more runs
python build_site.py runs/<run_dir>
```

| File | What it does |
|---|---|
| `agent.py` | The loop: Ultra → check → train → rollout → Super-VL → Ultra |
| `skill_env.py` | Loads a generated reward module into the G1 env |
| `train_skill.py` | `--probe` measures geometry, `--check` compiles a reward, default fine-tunes |
| `rollout_skill.py` | Video, key frames, hand positions, reward-term means |
| `build_site.py` | Static demo page (GitHub Pages) from run folders |
| `example_reward.py` | Hand-written example of the reward format |

## Results

_Filled in from the final run: per-round critic score and right-hand height._

## Credits

MuJoCo Playground and Brax (Google DeepMind), MuJoCo MJX, Unitree G1 model. NVIDIA Nemotron models via Nebius Token Factory.

## License

MIT — see `LICENSE`.
