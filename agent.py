"""Nemotron Reward Engineer: language -> reward code -> train -> watch -> improve.

    python agent.py --task "wave the right hand while walking" --base <base_ckpt> --iters 3

Models (Nebius Token Factory):
    Ultra      writes and repairs reward code
    Super-VL   watches rollout frames and critiques the behaviour
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/",
                api_key=os.environ["NEBIUS_API_KEY"], timeout=600)

CODER = "nvidia/Nemotron-3-Ultra-550b-a55b"   # designs the reward (hard reasoning)
FIXER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"  # first try at repairing compile errors (fast, cheap)
CRITIC = "nvidia/Nemotron-3.5-Super-VL"
PY = sys.executable

HEADER = "import jax\nimport jax.numpy as jp\n\n"

API_DOC = """You design reward terms for a Unitree G1 humanoid trained with PPO in MuJoCo MJX (JAX).
The base environment already rewards: following a walking velocity command, staying upright,
smooth actions, and a right-hand reach term "hand_reach" (scale 1.5) towards a target in front of the pelvis.
The policy starts from a checkpoint that already walks and reaches. You ADD terms to teach a new skill.

Write ONE Python module (no imports needed: `jax` and `jp` = jax.numpy are already imported) containing exactly:

SCALES = {"term_name": weight, ...}
def extra_rewards(env, data, action, info):
    ...
    return {"term_name": scalar, ...}

Rules:
- Every key returned by extra_rewards must be in SCALES. SCALES may also override "hand_reach" (e.g. 0.0 to switch it off if it conflicts with the new skill).
- Must be jit-traceable JAX: use jp/jax only (sigmoid is jax.nn.sigmoid, not jp.sigmoid), no numpy, no Python if/for/while on array values, no printing, no globals that change.
- Each returned value is a scalar. Prefer bounded tracking terms like jp.exp(-err**2 / sigma) with positive weight.
- SIGN CONVENTION for penalties: return a POSITIVE magnitude (e.g. a squared error) and give it a NEGATIVE weight in SCALES. Never negate both the value and the weight.
- The robot starts with its arms hanging down. Tracking terms must give useful signal from that rest pose: use a wide sigma
  (error of 0.3-0.5 m should still give clearly non-zero reward), or add a dense shaping term that grows steadily as the hand rises.
- Use the MEASURED BODY GEOMETRY below to choose realistic targets. Never ask for a hand height above max_right_hand_height_above_pelvis.
- The summed per-step reward is clipped at 0, so keep positive terms dominant. Keep weights modest (0.5-3.0) so walking is not destroyed.

Available quantities (world frame, z up, metres, seconds):
- data.time                                         simulation time, scalar (use for rhythm: jp.sin(2*jp.pi*f*data.time))
- pelvis = data.site_xpos[env._pelvis_imu_site_id]  pelvis position (3,)
- R = data.site_xmat[env._pelvis_imu_site_id].reshape(3, 3)  pelvis orientation; forward=R[:,0], left=R[:,1], up=R[:,2]
- data.site_xpos[env._hands_site_id[0]]             LEFT hand position (3,)
- data.site_xpos[env._hands_site_id[1]]             RIGHT hand position (3,)
- data.qvel[0:3]                                    base linear velocity
- action                                            policy action vector
- A point p in the pelvis frame: R.T @ (p - pelvis)

Reply with ONLY one ```python code block."""


def chat(model, messages, max_tokens=8000, retries=2):
    """Call a model; if it returns no answer (e.g. it spent its token budget thinking), retry with a bigger budget."""
    for attempt in range(retries + 1):
        r = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.4)
        msg = r.choices[0].message
        text = msg.content or ""
        if text.strip():
            return text
        print(f"  {model} returned an empty answer (finish_reason={r.choices[0].finish_reason}); retrying", flush=True)
        max_tokens *= 2
    return getattr(msg, "reasoning_content", None) or ""


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return (blocks[-1] if blocks else text).strip()


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def write_reward(messages, path, max_repairs=3):
    """Ultra designs the reward; compile errors go to Nano first (fast, cheap), then back to Ultra if Nano fails.
    Returns (code, log) where log lists which model produced each attempt and whether it compiled."""
    log = []
    for attempt in range(max_repairs + 1):
        model = CODER if attempt != 1 else FIXER
        t0 = time.time()
        reply = chat(model, messages)
        code = extract_code(reply)
        with open(path, "w") as f:
            f.write(HEADER + code + "\n")
        rc, out, err = run([PY, "train_skill.py", "--reward", path, "--check"])
        entry = {"attempt": attempt + 1, "model": model.split("/")[-1], "seconds": round(time.time() - t0, 1),
                 "compiled": rc == 0}
        if rc == 0:
            log.append(entry)
            print(f"  attempt {attempt + 1} by {entry['model']}: compiled", flush=True)
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            return code, log
        tail = (err or out)[-2500:]
        entry["error"] = tail.strip().splitlines()[-1][:300] if tail.strip() else "unknown error"
        log.append(entry)
        print(f"  attempt {attempt + 1} by {entry['model']} failed: {entry['error']}", flush=True)
        messages += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                     {"role": "user", "content": f"That code failed when compiled:\n{tail}\nFix it. Reply with ONLY one ```python code block."}]
    raise RuntimeError("could not produce a working reward")


def research(task, max_queries=2, per_query=4):
    """Nano turns the skill into search queries; Tavily finds how researchers reward similar skills.
    Returns (notes_for_ultra, sources). Skipped quietly if TAVILY_API_KEY is not set."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        print("  (no TAVILY_API_KEY: skipping web research)", flush=True)
        return "", []
    import httpx
    q_text = chat(FIXER, [{"role": "user", "content": (
        f"A humanoid robot (Unitree G1) must learn this skill with reinforcement learning: {task}\n"
        f"Write {max_queries} short web search queries (one per line, no numbering) to find how researchers "
        "design reward functions for this kind of humanoid skill.")}], max_tokens=2000)
    queries = [q.strip(" -*\"'0123456789.") for q in q_text.splitlines() if len(q.strip()) > 8][:max_queries]
    queries = queries or [f"reinforcement learning reward function humanoid {task}"]
    sources, notes = [], []
    for q in queries:
        try:
            r = httpx.post("https://api.tavily.com/search", timeout=60,
                           headers={"Authorization": f"Bearer {key}"},
                           json={"query": q, "max_results": per_query, "search_depth": "advanced",
                                 "include_answer": True})
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            print(f"  Tavily search failed for {q!r}: {e}", flush=True)
            continue
        if d.get("answer"):
            notes.append(f"Summary for '{q}': {d['answer']}")
        for res in d.get("results", []):
            sources.append({"query": q, "title": res.get("title", ""), "url": res.get("url", "")})
            notes.append(f"- {res.get('title', '')} ({res.get('url', '')}): {res.get('content', '')[:600]}")
    print(f"  Tavily: {len(sources)} sources from {len(queries)} queries", flush=True)
    return "\n".join(notes), sources


def probe_geometry():
    rc, out, err = run([PY, "train_skill.py", "--reward", "example_reward.py", "--probe"])
    lines = [l for l in out.splitlines() if l.startswith("{")]
    if rc != 0 or not lines:
        raise RuntimeError(f"geometry probe failed:\n{(err or out)[-2000:]}")
    return json.loads(lines[-1])


def critique(task, summary, geometry):
    content = [{"type": "text", "text": (
        f"Task for the humanoid robot: {task}\n"
        f"These {len(summary['frames'])} frames are in time order from a simulated Unitree G1 rollout.\n"
        "IMPORTANT: the robot often faces the camera, so its RIGHT arm can appear on the LEFT side of the image. "
        "Use the measured hand positions below (exact, from the simulator) to decide which arm is which and how high it is.\n"
        f"Measured hand positions in the pelvis frame (metres; z = height above pelvis, y = sideways, left positive): "
        f"{json.dumps(summary['hands_in_pelvis_frame_m'])}\n"
        f"Reference body geometry at rest: {json.dumps(geometry)}\n"
        f"Stats: stayed upright={summary['stayed_upright']}, steps survived={summary['steps_survived']}/{summary['horizon']}.\n"
        "Judge how well the robot performs the task from what you SEE. Reply with JSON only: "
        '{"score": 0-10, "what_happens": "...", "problems": "...", "suggestion": "..."}')}]
    for p in summary["frames"]:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    text = chat(CRITIC, [{"role": "user", "content": content}], max_tokens=8000)
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else {"score": None, "raw": text}
    except json.JSONDecodeError:
        sc = re.search(r'"score"\s*:\s*(\d+)', text)  # answer cut off: keep at least the score
        return {"score": int(sc.group(1)) if sc else None, "raw": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--base", required=True, help="base walk+reach checkpoint to fine-tune from")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--init-reward", default=None,
                    help="reward module the --base policy was trained with; Ultra improves it instead of starting blank")
    ap.add_argument("--no-chain", action="store_true",
                    help="always restart from --base instead of continuing from the best round so far")
    args = ap.parse_args()

    run_dir = os.path.join(HERE, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print("=== Measuring body geometry ===", flush=True)
    geometry = probe_geometry()
    print(json.dumps(geometry, indent=2), flush=True)
    json.dump(geometry, open(os.path.join(run_dir, "geometry.json"), "w"), indent=2)
    print("=== Researching reward design (Nemotron Nano + Tavily) ===", flush=True)
    notes, sources = research(args.task)
    json.dump({"notes": notes, "sources": sources}, open(os.path.join(run_dir, "research.json"), "w"), indent=2)
    system = API_DOC + "\n\nMEASURED BODY GEOMETRY at rest (pelvis frame, metres: x forward, y left, z up):\n" + json.dumps(geometry, indent=2)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Skill to teach: {args.task}\nWrite the reward module."}]
    if args.init_reward:
        prev = open(args.init_reward).read()
        messages[-1]["content"] = (f"Skill to teach: {args.task}\n"
            "The starting policy was already trained with the reward below. Keep what works and improve it.\n"
            f"```python\n{prev}\n```\nWrite the improved reward module.")
    if notes:
        messages[-1]["content"] += ("\n\nWEB RESEARCH on how researchers reward similar skills (from Tavily). Use ideas that fit; "
                                    "add a short comment in the code naming any idea you took from it:\n" + notes[:12000])
    history = []
    start_ckpt, best_score = args.base, -1

    for i in range(1, args.iters + 1):
        it_dir = os.path.join(run_dir, f"iter_{i}")
        os.makedirs(it_dir, exist_ok=True)
        reward_path = os.path.join(it_dir, "reward.py")
        print(f"\n=== Iteration {i}: Ultra writes the reward ===", flush=True)
        _, write_log = write_reward(messages, reward_path)

        print(f"=== Iteration {i}: training on GPU ===", flush=True)
        rc, out, err = run([PY, "train_skill.py", "--reward", reward_path, "--restore", start_ckpt,
                            "--timesteps", str(args.timesteps), "--logdir", os.path.join(it_dir, "train")])
        open(os.path.join(it_dir, "train_log.txt"), "w").write(out + err)
        ckpt = re.findall(r"CKPT=(\S+)", out)
        if rc != 0 or not ckpt or ckpt[-1] == "None":
            raise RuntimeError(f"training failed, see {it_dir}/train_log.txt")

        print(f"=== Iteration {i}: rollout + Super-VL critique ===", flush=True)
        rc, out, err = run([PY, "rollout_skill.py", "--reward", reward_path, "--checkpoint", ckpt[-1],
                            "--outdir", os.path.join(it_dir, "rollout")])
        if rc != 0:
            raise RuntimeError(f"rollout failed:\n{err[-2000:]}")
        summary = json.load(open(os.path.join(it_dir, "rollout", "summary.json")))
        review = critique(args.task, summary, geometry)
        json.dump(review, open(os.path.join(it_dir, "critique.json"), "w"), indent=2)
        print(json.dumps(review, indent=2), flush=True)
        history.append({"iter": i, "checkpoint": ckpt[-1], "started_from": start_ckpt, "write_log": write_log,
                        "summary": summary, "critique": review})
        score = review.get("score")
        if not args.no_chain and isinstance(score, (int, float)) and score >= best_score:
            best_score, start_ckpt = score, ckpt[-1]
            print(f"  next round continues from iteration {i} (score {score})", flush=True)

        messages.append({"role": "user", "content": (
            f"Result of training with your reward:\nStats: {json.dumps({k: v for k, v in summary.items() if k != 'frames'})}\n"
            f"Vision critic watching the video: {json.dumps(review)}\n"
            "Improve the reward so the robot performs the skill better while staying upright. "
            "Briefly think about what caused the problems, then reply with ONLY one ```python code block.")})

    scored = [h for h in history if isinstance(h["critique"].get("score"), (int, float))]
    best = max(scored, key=lambda h: h["critique"]["score"]) if scored else history[-1]
    json.dump({"task": args.task, "best_iter": best["iter"], "history": history},
              open(os.path.join(run_dir, "report.json"), "w"), indent=2)
    print(f"\nDone. Best iteration: {best['iter']} (score {best['critique'].get('score')}). Results in {run_dir}")


if __name__ == "__main__":
    main()
