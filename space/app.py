"""Try-it-yourself demo: Nemotron writes a reward for a Unitree G1 skill, live.

Type a skill -> Nemotron 3 Ultra writes a JAX reward module -> the module is compiled and
evaluated on the G1's measured body geometry (no simulator needed) -> a plot shows how each
reward term changes as the right hand rises from the hip to full reach. Compile errors are
sent back to Ultra for automatic repair, as in the full agent.

Training is not run here (it needs a GPU and ~10-20 min per round); see the results page.
"""
import json
import os
import re
import time
from types import SimpleNamespace

import gradio as gr
import jax
import jax.numpy as jp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI

MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"
RESULTS_URL = "https://vishwavenkatesh.github.io/nemotron-reward-engineer/"
REPO_URL = "https://github.com/VishwaVenkatesh/nemotron-reward-engineer"

client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/",
                api_key=os.environ.get("NEBIUS_API_KEY", "missing"), timeout=300)

GEOMETRY = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry.json")))

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

SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ("abs", "min", "max", "range", "float", "int", "len", "dict", "list", "tuple", "zip",
                           "enumerate", "sum", "isinstance", "round", "bool", "str", "ValueError", "KeyError")}

# --- simple abuse guard: limits Token Factory spend on a public page ---
_calls = []
MAX_CALLS_PER_HOUR = 30


def _rate_ok():
    now = time.time()
    _calls[:] = [t for t in _calls if now - t < 3600]
    if len(_calls) >= MAX_CALLS_PER_HOUR:
        return False
    _calls.append(now)
    return True


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return (blocks[-1] if blocks else text).strip()


def load_module(code):
    ns = {"jax": jax, "jp": jp, "__builtins__": SAFE_BUILTINS}
    exec(compile(code, "<generated_reward>", "exec"), ns)
    if not isinstance(ns.get("SCALES"), dict):
        raise ValueError("SCALES must be a dict")
    if not callable(ns.get("extra_rewards")):
        raise ValueError("extra_rewards(env, data, action, info) is missing")
    return ns["SCALES"], ns["extra_rewards"]


# Mock robot state built from the G1's measured geometry: pelvis at its standing height,
# identity orientation, left hand at rest, right hand moved along a path from rest to full reach.
PELVIS = jp.array([0.0, 0.0, GEOMETRY["pelvis_height_above_ground"]])
ENV = SimpleNamespace(_pelvis_imu_site_id=0, _hands_site_id=(1, 2))
REST_R = jp.array(GEOMETRY["right_hand_at_rest"])
REST_L = jp.array(GEOMETRY["left_hand_at_rest"])
TOP_R = jp.array([0.0, -0.15, GEOMETRY["max_right_hand_height_above_pelvis"] - 0.01])


def make_data(right_local, t):
    site_xpos = jp.stack([PELVIS, PELVIS + REST_L, PELVIS + right_local])
    site_xmat = jp.tile(jp.eye(3).reshape(1, 9), (3, 1))
    return SimpleNamespace(site_xpos=site_xpos, site_xmat=site_xmat, time=t,
                           qvel=jp.zeros(35), qpos=jp.zeros(36))


def evaluate(scales, fn, n=40):
    """Weighted value of every new term along the hand path from rest to full reach."""
    terms = [k for k in scales if k != "hand_reach"]

    @jax.jit
    def at(alpha, t):
        out = fn(ENV, make_data(REST_R + alpha * (TOP_R - REST_R), t), jp.zeros(29), {})
        return {k: jp.asarray(out[k], jp.float32).reshape(()) * scales[k] for k in terms}

    heights, rows = [], []
    for i in range(n):
        a = i / (n - 1)
        vals = at(a, 0.0)
        heights.append(float((REST_R + a * (TOP_R - REST_R))[2]))
        rows.append({k: float(v) for k, v in vals.items()})
    bad = [k for k in terms if any(not jp.isfinite(r[k]) for r in rows)]
    if bad:
        raise ValueError(f"non-finite values for terms: {bad}")
    return heights, rows, terms


def plot(heights, rows, terms):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=110)
    for k in terms:
        ax.plot(heights, [r[k] for r in rows], label=k, linewidth=2)
    ax.plot(heights, [sum(r[k] for k in terms) for r in rows], label="total (new terms)", color="black",
            linewidth=2.5, linestyle="--")
    sh = GEOMETRY.get("right_shoulder_pitch_link", [0, 0, 0.375])[2]
    hd = GEOMETRY.get("approx_head_top_above_pelvis", 0.65)
    for x, name in ((sh, "shoulder"), (hd, "top of head")):
        ax.axvline(x, color="grey", linewidth=1, linestyle=":")
        ax.text(x, ax.get_ylim()[1], f" {name}", color="grey", fontsize=9, va="top")
    ax.set_xlabel("right-hand height above pelvis (m)  ·  rest pose = 0.02 m")
    ax.set_ylabel("weighted reward per step")
    ax.set_title("How the reward responds as the right hand rises")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def run(skill, progress=gr.Progress()):
    skill = (skill or "").strip()
    if not skill:
        raise gr.Error("Type a skill first.")
    if len(skill) > 300:
        raise gr.Error("Please keep the skill under 300 characters.")
    if not _rate_ok():
        raise gr.Error("This public demo is rate-limited. Please try again later.")

    system = API_DOC + "\n\nMEASURED BODY GEOMETRY at rest (pelvis frame, metres: x forward, y left, z up):\n" \
        + json.dumps(GEOMETRY, indent=2)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Skill to teach: {skill}\nWrite the reward module."}]
    log = []
    for attempt in range(1, 4):
        progress(0.1 + 0.25 * attempt, desc=f"Nemotron 3 Ultra is writing the reward (attempt {attempt})")
        r = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=8000, temperature=0.4)
        code = extract_code(r.choices[0].message.content or "")
        try:
            scales, fn = load_module(code)
            heights, rows, terms = evaluate(scales, fn)
            log.append(f"Attempt {attempt}: compiled and evaluated OK.")
            summary = (f"**Terms:** " + ", ".join(f"`{k}` × {v}" for k, v in scales.items())
                       + f"\n\n**At rest (hand at hip):** total new-term reward = "
                       f"{sum(rows[0][k] for k in terms):.3f}  ·  "
                       f"**at full reach:** {sum(rows[-1][k] for k in terms):.3f}")
            return code, plot(heights, rows, terms), summary, "\n".join(log)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            log.append(f"Attempt {attempt}: failed → {err[:300]}  (sent back to Ultra)")
            messages += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                         {"role": "user", "content": f"That code failed:\n{err[:1500]}\nFix it. Reply with ONLY one ```python code block."}]
    return code, None, "Ultra could not produce working code in 3 attempts.", "\n".join(log)


EXAMPLES = [
    "raise the right hand above the head and wave it side to side while walking",
    "hold the right hand out to the side at shoulder height while walking",
    "keep both hands close to the chest while walking, like holding a box",
    "touch the top of the head with the right hand",
]

with gr.Blocks(title="Nemotron Reward Engineer · try it") as demo:
    gr.Markdown(f"""
# Nemotron Reward Engineer · try it yourself
Type a skill for a **Unitree G1 humanoid**. **Nemotron 3 Ultra** (via Nebius Token Factory) writes the
reinforcement‑learning reward. The code is compiled and evaluated on the robot's **measured body geometry**,
and the plot shows whether the reward gives the robot a learning signal from its arms‑down starting pose.
If the code fails, the error goes back to Ultra and it repairs itself.

Training runs offline on a GPU (~10–20 min per round), with **Nemotron 3.5 Super‑VL** watching each rollout.
See the full runs, videos and critiques on the **[results page]({RESULTS_URL})** · [source code]({REPO_URL})
""")
    with gr.Row():
        skill = gr.Textbox(label="Skill", placeholder=EXAMPLES[0], scale=4)
        go = gr.Button("Write reward", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=skill)
    summary = gr.Markdown()
    with gr.Row():
        code = gr.Code(label="Reward written by Nemotron 3 Ultra", language="python")
        fig = gr.Plot(label="Reward vs. right-hand height")
    log = gr.Textbox(label="Compile check / self-repair log", lines=4)
    go.click(run, inputs=skill, outputs=[code, fig, summary, log])
    skill.submit(run, inputs=skill, outputs=[code, fig, summary, log])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
