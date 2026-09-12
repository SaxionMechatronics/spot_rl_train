#!/bin/bash

# ntfy topic to publish progress to (public server: anyone with this string can read the messages)
NTFY_TOPIC="${NTFY_TOPIC:-rl-train-qbOMRxi8gtdwsvAZmLrdLPNZ}"

for i in {1..5}; do
  echo "Running seed $i"
  python3 scripts/rsl_rl/train.py \
    --task Spot-Rl-Handstand-v0 \
    --headless \
    --seed $i
  status=$?

  if [ $status -eq 0 ]; then
    curl -s -o /dev/null -d "Seed $i complete" "https://ntfy.sh/$NTFY_TOPIC"
  else
    curl -s -o /dev/null -d "Seed $i FAILED (exit $status)" "https://ntfy.sh/$NTFY_TOPIC"
  fi
done
