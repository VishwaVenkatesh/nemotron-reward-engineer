"""Build the static demo site (GitHub Pages) from one or more agent runs.

    python build_site.py "runs/A::Stage 1 label" "runs/B::Stage 2 label"   -> docs/index.html

Each run folder is what agent.py writes: report.json, geometry.json, iter_k/{reward.py,
critique.json, train_log.txt, rollout/{summary.json, rollout.mp4}}.
"""
import html
import json
import os
import re
import shutil
import sys

OUT = "docs"


def curve(train_log):
    pts = [(int(a), float(b)) for a, b in re.findall(r"^(\d+): reward=([-\d.]+)", train_log, flags=re.M)
           if b not in ("nan",)]
    return pts


def sparkline(pts, w=260, h=60):
    if len(pts) < 2:
        return ""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = lambda x: 4 + (w - 8) * (x - x0) / max(x1 - x0, 1)
    sy = lambda y: h - 4 - (h - 8) * (y - y0) / max(y1 - y0, 1e-9)
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" class="spark" role="img" aria-label="training reward curve">'
            f'<path d="{path}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>'
            f'<div class="muted small">training reward {ys[0]:.1f} → {ys[-1]:.1f} over {x1/1e6:.0f}M steps</div>')


def height_bar(hand, geo):
    """Right-hand max height vs. shoulder / head / max reach, as a horizontal gauge."""
    top = geo.get("max_right_hand_height_above_pelvis", 0.8) + 0.05
    pct = lambda v: max(0.0, min(100.0, 100 * v / top))
    mean, mx = hand["height_above_pelvis_mean"], hand["height_above_pelvis_max"]
    marks = [("shoulder", geo.get("right_shoulder_pitch_link", [0, 0, 0.375])[2]),
             ("head top", geo.get("approx_head_top_above_pelvis", 0.65))]
    marks_html = "".join(f'<span class="mark" style="left:{pct(v):.1f}%"><i>{n}</i></span>' for n, v in marks)
    return (f'<div class="gauge"><span class="fill-mean" style="width:{pct(mean):.1f}%"></span>'
            f'<span class="fill-max" style="left:{pct(mx):.1f}%"></span>{marks_html}</div>'
            f'<div class="muted small">right hand above pelvis: mean {mean:.2f} m, max {mx:.2f} m</div>')


def fix_critique(c):
    """Recover fields from a critique whose JSON answer was cut off."""
    if c.get("score") is not None or not c.get("raw"):
        return c
    raw, out = c["raw"], dict(c)
    m = re.search(r'"score"\s*:\s*(\d+)', raw)
    if m:
        out["score"] = int(m.group(1))
    for k in ("what_happens", "problems", "suggestion"):
        m = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
        if m:
            out[k] = m.group(1).replace('\\"', '"')
    return out


def build_run(run_dir, idx, label=""):
    report = json.load(open(os.path.join(run_dir, "report.json")))
    geo_path = os.path.join(run_dir, "geometry.json")
    geo = json.load(open(geo_path)) if os.path.exists(geo_path) else {}
    rid = f"run{idx}"
    os.makedirs(os.path.join(OUT, rid), exist_ok=True)
    cards, progress = [], []
    for h in report["history"]:
        i = h["iter"]
        it = os.path.join(run_dir, f"iter_{i}")
        vid_src = os.path.join(it, "rollout", "rollout.mp4")
        vid = f"{rid}/iter_{i}.mp4"
        if os.path.exists(vid_src):
            shutil.copy(vid_src, os.path.join(OUT, vid))
        code = open(os.path.join(it, "reward.py")).read()
        log_path = os.path.join(it, "train_log.txt")
        pts = curve(open(log_path).read()) if os.path.exists(log_path) else []
        c = fix_critique(h["critique"])
        score = c.get("score")
        score_txt = "–" if score is None else f"{score}/10"
        wl = h.get("write_log") or []
        repair = ""
        if len(wl) > 1:
            steps = " → ".join(
                f"{html.escape(e['model'].replace('NVIDIA-', '').split('-A')[0].replace('-550b', ''))} "
                + ("compiled ✓" if e["compiled"] else f"failed (<code>{html.escape(e.get('error', ''))}</code>)")
                for e in wl)
            repair = f'<p class="small"><b>Self‑repair:</b> {steps}</p>'
        hand = h["summary"].get("hands_in_pelvis_frame_m", {}).get("right")
        if hand:
            progress.append((f"{label or rid} · round {i}", hand["height_above_pelvis_max"], score))
        best = " best" if i == report.get("best_iter") else ""
        cards.append(f"""
<article class="iter{best}">
  <header><h3>Round {i}</h3><span class="score">critic {score_txt}</span>{'<span class="badge">best</span>' if best else ''}</header>
  <div class="cols">
    <div>
      <video src="{vid}" controls muted loop playsinline preload="metadata"></video>
      {height_bar(hand, geo) if hand else ''}
      {sparkline(pts)}
    </div>
    <div>
      <h4>Nemotron Super‑VL watched the rollout</h4>
      <p>{html.escape(str(c.get('what_happens', c.get('raw', ''))))}</p>
      <p><b>Problems:</b> {html.escape(str(c.get('problems', '')))}</p>
      <p><b>Suggestion:</b> {html.escape(str(c.get('suggestion', '')))}</p>
      {repair}
      <details><summary>Reward code Nemotron Ultra wrote</summary><pre><code>{html.escape(code)}</code></pre></details>
    </div>
  </div>
</article>""")
    head = f"<p class='kicker'>{html.escape(label)}</p>" if label else ""
    return f"""
<section class="run">
  {head}<h2>“{html.escape(report['task'])}”</h2>
  <p class="muted">{len(report['history'])} rounds of write → train → watch → improve, fully automatic.</p>
  {''.join(cards)}
</section>""", progress, geo


def overview(progress, geo):
    """One bar per round: highest right-hand height, with shoulder and head lines."""
    if not progress:
        return ""
    top = geo.get("max_right_hand_height_above_pelvis", 0.8)
    pct = lambda v: max(0.0, min(100.0, 100 * v / top))
    sh = geo.get("right_shoulder_pitch_link", [0, 0, 0.375])[2]
    hd = geo.get("approx_head_top_above_pelvis", 0.65)
    rows = "".join(
        f'<div class="ov-row"><span class="ov-label">{html.escape(n)}</span>'
        f'<span class="ov-track"><span class="ov-bar" style="width:{pct(v):.1f}%"></span>'
        f'<span class="mark" style="left:{pct(sh):.1f}%"></span><span class="mark" style="left:{pct(hd):.1f}%"></span></span>'
        f'<span class="ov-val">{v:.2f} m{"" if s is None else f" · {s}/10"}</span></div>'
        for n, v, s in progress)
    return f"""
<section class="overview">
  <h2>Progress across every round</h2>
  <p class="muted small">Highest right‑hand height above the pelvis in each rollout (hand at rest: 0.02 m).
  Grey lines: shoulder ({sh:.2f} m) and top of head ({hd:.2f} m). Right column: vision‑critic score.</p>
  {rows}
</section>"""


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nemotron Reward Engineer</title>
<style>
:root{--bg:#0f1210;--card:#171c18;--text:#e8ede9;--muted:#93a097;--accent:#76b900;--line:#2a322c}
@media (prefers-color-scheme: light){:root{--bg:#f6f8f5;--card:#fff;--text:#18201a;--muted:#5d6a61;--accent:#4f8000;--line:#dfe5dd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:clamp(28px,5vw,44px);line-height:1.1;margin:0 0 8px}h2{margin:48px 0 4px}h3{margin:0}h4{margin:0 0 6px;font-size:15px;color:var(--accent)}
.muted{color:var(--muted)}.small{font-size:13px}
.lede{font-size:18px;max-width:760px}
.pipeline{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0}
.pipeline span{border:1px solid var(--line);background:var(--card);border-radius:8px;padding:8px 12px;font-size:14px}
.pipeline b{color:var(--accent)}
.kicker{margin:48px 0 0;color:var(--accent);font-size:13px;letter-spacing:.06em;text-transform:uppercase}
.kicker+h2{margin-top:4px}
.overview{margin-top:32px}
.ov-row{display:grid;grid-template-columns:200px minmax(0,1fr) 110px;gap:12px;align-items:center;margin:8px 0;font-size:14px}
.ov-track{position:relative;height:16px;background:var(--card);border:1px solid var(--line);border-radius:4px}
.ov-bar{position:absolute;left:0;top:0;bottom:0;background:var(--accent);border-radius:3px}
.ov-val{color:var(--muted);font-variant-numeric:tabular-nums}
@media (max-width:640px){.ov-row{grid-template-columns:1fr 92px}.ov-label{grid-column:1/-1}}
.hero{margin:28px 0 8px}.hero video{aspect-ratio:auto}.hero figcaption{color:var(--muted);font-size:14px;margin-top:6px}
.iter{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0}
.iter.best{border-color:var(--accent)}
.iter header{display:flex;gap:12px;align-items:baseline;margin-bottom:12px}
.score{color:var(--muted);font-size:14px}.badge{background:var(--accent);color:#0b0d0b;border-radius:99px;padding:1px 10px;font-size:12px;font-weight:600}
.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}
@media (max-width:760px){.cols{grid-template-columns:1fr}}
video{width:100%;border-radius:8px;background:#000;aspect-ratio:4/3}
pre{overflow:auto;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px;font-size:12.5px;line-height:1.45}
details summary{cursor:pointer;color:var(--accent);margin-top:8px}
.spark{width:100%;height:60px;margin-top:10px}
.gauge{position:relative;height:14px;background:var(--bg);border:1px solid var(--line);border-radius:99px;margin-top:12px;overflow:visible}
.fill-mean{position:absolute;left:0;top:0;bottom:0;background:var(--accent);opacity:.55;border-radius:99px}
.fill-max{position:absolute;top:-3px;bottom:-3px;width:3px;background:var(--accent)}
.mark{position:absolute;top:-6px;bottom:-6px;width:1px;background:var(--muted)}
.mark i{position:absolute;top:-18px;left:-20px;font-size:11px;font-style:normal;color:var(--muted);white-space:nowrap}
footer{margin-top:64px;color:var(--muted);font-size:14px}
a{color:var(--accent)}
</style></head><body><main>
<h1>Nemotron Reward Engineer</h1>
<p class="lede">Type a skill in plain English. An AI agent writes the reinforcement‑learning reward, trains a Unitree G1 humanoid,
<b>watches the result</b>, and rewrites the reward until the robot gets it right. No human tuning in the loop.</p>
<div class="pipeline">
<span><b>Nemotron 3 Ultra</b> writes reward code</span><span>→ compile check; <b>Nemotron Nano</b> repairs errors</span>
<span>→ PPO fine‑tune in MuJoCo MJX</span><span>→ rollout video + measured hand positions</span>
<span><b>Nemotron 3.5 Super‑VL</b> critiques</span><span>→ back to Ultra</span>
</div>
<p class="muted small">All models served through Nebius Token Factory. Starting policy: my whole‑body walk + reach controller for the G1.</p>
{hero}{runs}
<footer>Built for the Nebius × NVIDIA Global AI Hackathon 2026 · Physical AI track · by Vishwa Venkatesh ·
<a href="https://github.com/VishwaVenkatesh/nemotron-reward-engineer">source code</a></footer>
</main></body></html>"""


def main():
    runs = sys.argv[1:]
    if not runs:
        sys.exit("usage: python build_site.py runs/<run_dir> [more run dirs]")
    os.makedirs(OUT, exist_ok=True)
    parts, progress, geo = [], [], {}
    for i, arg in enumerate(runs):
        path, _, label = arg.partition("::")
        html_part, prog, g = build_run(path, i + 1, label)
        parts.append(html_part); progress += prog; geo = geo or g
    body = overview(progress, geo) + "".join(parts)
    hero = ""
    if os.path.exists(os.path.join(OUT, "before_after.mp4")):
        hero = ('<figure class="hero"><video src="before_after.mp4" autoplay muted loop playsinline controls></video>'
                '<figcaption>Left: the first round. Right: the final round. Same robot, same starting policy; '
                'every reward in between was written by Nemotron 3 Ultra and judged by Nemotron 3.5 Super‑VL.</figcaption></figure>')
    open(os.path.join(OUT, "index.html"), "w").write(PAGE.replace("{hero}", hero).replace("{runs}", body))
    open(os.path.join(OUT, ".nojekyll"), "w").write("")
    print(f"wrote {OUT}/index.html")


if __name__ == "__main__":
    main()
