#!/usr/bin/env bash
# Start / stop NVIDIA Cosmos Reason 2 (8B) as a Nebius Serverless Endpoint with vLLM on one L40S GPU.
#   bash cloud/cosmos_endpoint.sh up      # ~5-10 min until Running; prints COSMOS_URL / COSMOS_TOKEN
#   bash cloud/cosmos_endpoint.sh down    # delete it (billing stops)
set -euo pipefail
NAME=cosmos-reason2
MODEL=nvidia/Cosmos-Reason2-8B

if [ "${1:-}" = "up" ]; then
  TOKEN=$(openssl rand -hex 32)
  SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')
  nebius ai endpoint create \
    --name "$NAME" \
    --image vllm/vllm-openai:v0.18.0-cu130 \
    --container-command "python3 -m vllm.entrypoints.openai.api_server" \
    --args "--model $MODEL --host 0.0.0.0 --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.9 --reasoning-parser qwen3" \
    --platform gpu-l40s-a \
    --preset 1gpu-8vcpu-32gb \
    --public \
    --container-port 8000 \
    --auth token \
    --token "$TOKEN" \
    --shm-size 16Gi \
    --subnet-id "$SUBNET_ID"
  echo "export COSMOS_TOKEN=$TOKEN" > cloud/.cosmos_env
  echo "Created. Waiting for it to be Running (model download + load takes several minutes)..."
  for _ in $(seq 1 60); do
    sleep 20
    URL=$(nebius ai endpoint get-by-name --name "$NAME" --format json \
      | python -c "import json,sys; d=json.load(sys.stdin); e=[x for x in d.get('status',{}).get('public_endpoints',[]) if x]; print(e[0] if e else '')")
    if [ -n "$URL" ]; then
      case "$URL" in http*) ;; *) URL="http://$URL" ;; esac
      if curl -sf -m 10 "$URL/v1/models" -H "Authorization: Bearer $TOKEN" >/dev/null; then
        echo "export COSMOS_URL=$URL" >> cloud/.cosmos_env
        echo "Ready: $URL"; echo "Run: source cloud/.cosmos_env"
        exit 0
      fi
    fi
    echo "  not ready yet..."
  done
  echo "Still not ready after 20 min. Check: nebius ai endpoint get-by-name --name $NAME"
elif [ "${1:-}" = "down" ]; then
  ID=$(nebius ai endpoint get-by-name --name "$NAME" --format jsonpath='{.metadata.id}')
  nebius ai endpoint delete "$ID" && echo "Deleted $NAME ($ID). Billing stopped."
else
  echo "usage: bash cloud/cosmos_endpoint.sh up|down"
fi
