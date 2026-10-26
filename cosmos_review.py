"""Physics review of rollout videos by NVIDIA Cosmos Reason 2, served from a Nebius Serverless Endpoint.

Super-VL (via Token Factory) judges whether the robot did the task from 6 frames; Cosmos Reason 2 watches the
whole video and judges the physics: balance, foot contact and slipping, and how natural the motion is.

    export COSMOS_URL=https://<endpoint>   COSMOS_TOKEN=<endpoint token>
    python cosmos_review.py runs/<run_dir> [more run dirs]     -> runs/<run>/iter_k/cosmos.json
"""
import base64
import json
import os
import re
import sys

from openai import OpenAI

MODEL = os.environ.get("COSMOS_MODEL", "nvidia/Cosmos-Reason2-8B")


def client():
    url = os.environ["COSMOS_URL"].rstrip("/")
    return OpenAI(base_url=url + "/v1", api_key=os.environ["COSMOS_TOKEN"], timeout=600)


PROMPT = """You are watching a video of a simulated Unitree G1 humanoid robot (MuJoCo physics).
The robot was trained with reinforcement learning to do this task: {task}

Judge the motion as a physical-AI expert:
1. Balance and stability (does it stay upright, sway, stumble?)
2. Foot contact (do feet slide, float, or penetrate the floor?)
3. Naturalness and smoothness of the joint motion
4. Whether the motion achieves the task

Reply with JSON only:
{{"physics_score": 0-10, "task_score": 0-10, "observations": "...", "physical_issues": "..."}}"""


def review(cli, task, video_path):
    b64 = base64.b64encode(open(video_path, "rb").read()).decode()
    r = cli.chat.completions.create(
        model=MODEL, max_tokens=4096, temperature=0.2,
        messages=[{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
            {"type": "text", "text": PROMPT.format(task=task)}]}])
    text = r.choices[0].message.content or ""
    blocks = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    return {"physics_score": None, "raw": text[-2000:]}


def main():
    runs = sys.argv[1:]
    if not runs:
        sys.exit("usage: python cosmos_review.py runs/<run_dir> [...]")
    cli = client()
    for run_dir in runs:
        report = json.load(open(os.path.join(run_dir, "report.json")))
        for h in report["history"]:
            it = os.path.join(run_dir, f"iter_{h['iter']}")
            video = os.path.join(it, "rollout", "rollout.mp4")
            if not os.path.exists(video):
                continue
            res = review(cli, report["task"], video)
            json.dump(res, open(os.path.join(it, "cosmos.json"), "w"), indent=2)
            print(f"{run_dir} round {h['iter']}: physics {res.get('physics_score')}/10, task {res.get('task_score')}/10",
                  flush=True)


if __name__ == "__main__":
    main()
