#!/usr/bin/env bash
# Launch cloud_eval.py as a Nebius Serverless Job on one L40S GPU, wait for it, and save the results.
#   bash cloud/run_eval_job.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# The job downloads cloud/job_entry.py from GitHub and runs it (clone, install, evaluate).
ENTRY=https://raw.githubusercontent.com/VishwaVenkatesh/nemotron-reward-engineer/main/cloud/job_entry.py
NAME=g1-skill-eval-$(date +%H%M%S)
SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')

nebius ai job create \
  --name "$NAME" \
  --image python:3.11 \
  --container-command python \
  --args "-c exec(__import__('urllib.request').request.urlopen('$ENTRY').read())" \
  --platform gpu-l40s-a \
  --preset 1gpu-8vcpu-32gb \
  --timeout 1h \
  --subnet-id "$SUBNET_ID"

JOB_ID=$(nebius ai job get-by-name --name "$NAME" --format jsonpath='{.metadata.id}')
echo "job $JOB_ID started; waiting for results (installs take a few minutes)..."
for _ in $(seq 1 120); do
  sleep 30
  STATUS=$(nebius ai job get "$JOB_ID" --format json | python -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get('status',{})))")
  echo "  status: $STATUS" | cut -c1-160
  if echo "$STATUS" | grep -qiE '"(COMPLETED|SUCCEEDED|FAILED|CANCELLED|ERROR)'; then break; fi
done
nebius ai job logs "$JOB_ID" > "cloud/eval_job_$JOB_ID.log" || true
grep "CLOUD_EVAL_RESULT" "cloud/eval_job_$JOB_ID.log" | sed 's/^.*CLOUD_EVAL_RESULT //' \
  | python -c "import json,sys; r=[json.loads(l) for l in sys.stdin if l.strip()]; json.dump({'job_id':'$JOB_ID','platform':'Nebius Serverless Jobs, 1x L40S','results':r}, open('docs/cloud_eval.json','w'), indent=2); print(json.dumps(r, indent=2))"
echo "saved docs/cloud_eval.json and cloud/eval_job_$JOB_ID.log"
