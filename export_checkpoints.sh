#!/usr/bin/env bash
# Copy each skill's best policy + reward into checkpoints/ so cloud jobs (and judges) can evaluate them.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p checkpoints

export_skill () {   # name  run_dir  iter
  local name=$1 run=$2 it=$3
  local ck
  ck=$(ls -d "$run/iter_$it/train/checkpoints/"0* | tail -1)
  rm -rf "checkpoints/$name"; mkdir -p "checkpoints/$name"
  cp -r "$ck" "checkpoints/$name/policy"
  cp "$run/iter_$it/reward.py" "checkpoints/$name/reward.py"
  echo "  $name <- $ck"
}

export_skill wave       runs/20261025-124041 2
export_skill half_squat runs/20261025-134237 2
export_skill carry_box  runs/20261025-143512 1

cat > checkpoints/manifest.json <<'EOF'
[
  {"skill": "wave",       "reward": "checkpoints/wave/reward.py",       "checkpoint": "checkpoints/wave/policy"},
  {"skill": "half_squat", "reward": "checkpoints/half_squat/reward.py", "checkpoint": "checkpoints/half_squat/policy"},
  {"skill": "carry_box",  "reward": "checkpoints/carry_box/reward.py",  "checkpoint": "checkpoints/carry_box/policy"}
]
EOF
du -sh checkpoints
