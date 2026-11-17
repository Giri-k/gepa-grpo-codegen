#!/bin/bash
# GEPA for the system prompt (condition C2), on two GH200 nodes:
#   node 1: GEPA + Qwen3-4B-Thinking-2507 (vLLM in process) + the sandbox (codegen.gepa_run)
#   node 2: the reflection model, Qwen3-30B-A3B-Thinking-2507, behind vLLM's OpenAI server
#
#   cd $WORK/post-training/gepa-grpo-codegen
#   OUT=$SCRATCH/gepa-grpo/gepa-main BUDGET_NODE_HOURS=16 sbatch tacc/gepa.sh [codegen.gepa_run args]
#   OUT=... sbatch -t 01:00:00 tacc/gepa.sh --train-limit 6 --val-limit 6 --max-metric-calls 40   # smoke test
#
# GEPA keeps its state in $OUT. A job stopped by its time limit or the node-hour
# budget exits 3; submit the same command again to continue (TACC does not accept
# sbatch from compute nodes, so jobs can't resubmit themselves).
#
#SBATCH -J gepa
#SBATCH -p gh-dev
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node 1
#SBATCH -c 72  # Vista binds a task to one core otherwise
#SBATCH -t 02:00:00
#SBATCH -A CDA26008
#SBATCH -o logs/gepa-%j.out

cd "$SLURM_SUBMIT_DIR"
module load tacc-apptainer/1.4.1
unset PYTHONPATH PYTHONHOME
export CC=gcc CXX=g++  # Triton/inductor compile helpers with $CC; TACC sets it to nvc, which fails
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1
export CODEGEN_SANDBOX_SIF=$WORK/sif/python311.sif
export TOKENIZERS_PARALLELISM=false

OUT=${OUT:-$SCRATCH/gepa-grpo/gepa-$SLURM_JOB_ID}
NAME=$(basename "$OUT")
REFLECT_MODEL=${REFLECT_MODEL:-Qwen/Qwen3-30B-A3B-Thinking-2507}
PORT=8000
mkdir -p "$OUT"

# Stop 8 min before the time limit (gepa_run also leaves room for one more iteration).
LIMIT_STR=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
LEFT_MIN=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF == 4) print $1*1440 + $2*60 + $3; else if (NF == 3) print $1*60 + $2; else print $1 }')
DEADLINE=$(( $(date +%s) + 60 * (LEFT_MIN - 8) ))
# Node-hour budget across all jobs of this run, counted for both nodes.
BUDGET_NODE_HOURS=${BUDGET_NODE_HOURS:-16}
START=$(date +%s)
USED=$(awk '{ s += $2 } END { print s + 0 }' "$OUT/node_seconds.log" 2>/dev/null)
LEFT_BUDGET=$(( BUDGET_NODE_HOURS * 3600 - ${USED:-0} ))
echo "== node-hours used so far: $(( ${USED:-0} / 3600 )) of $BUDGET_NODE_HOURS"
if [ "$LEFT_BUDGET" -lt 1800 ]; then
  echo "== $(date) $NAME has used its $BUDGET_NODE_HOURS node-hour budget; not starting"
  exit 0
fi
BUDGET_DEADLINE=$(( START + LEFT_BUDGET / SLURM_NNODES - 8 * 60 ))
[ "$BUDGET_DEADLINE" -lt "$DEADLINE" ] && DEADLINE=$BUDGET_DEADLINE
record_use() { echo "$SLURM_JOB_ID $(( ($(date +%s) - START) * SLURM_NNODES ))" >> "$OUT/node_seconds.log"; }
trap record_use EXIT
trap 'exit 143' TERM

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MAIN=${NODES[0]} REFLECT=${NODES[1]}
echo "== $(date) $NAME: GEPA on $MAIN, reflection model on $REFLECT (limit $LIMIT_STR, stop by $(date -d @$DEADLINE +%H:%M)), args: $*"

# Reflection server on the second node. No reasoning parser: gepa_run splits the
# thinking from the answer itself.
srun -N 1 -n 1 -w "$REFLECT" --overlap \
  .venv-gpu/bin/vllm serve "$REFLECT_MODEL" --served-model-name reflect --host 0.0.0.0 --port $PORT \
  --max-model-len 65536 --gpu-memory-utilization 0.9 --seed 0 > "$OUT/reflect-server-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!

# The 4B model and the sandbox image from the node's own disk (see grpo.sh).
MODEL_DIR=hub/models--Qwen--Qwen3-4B-Thinking-2507
if mkdir -p /tmp/hf/hub && cp -a "$HF_HOME/$MODEL_DIR" /tmp/hf/hub/; then
  export HF_HOME=/tmp/hf
fi
SIF_SRC=$SCRATCH/sif/python311.sif
[ -f "$SIF_SRC" ] || SIF_SRC=$CODEGEN_SANDBOX_SIF
cp "$SIF_SRC" /tmp/python311.sif && export CODEGEN_SANDBOX_SIF=/tmp/python311.sif

URL=http://$REFLECT:$PORT/v1
for i in $(seq 1 90); do
  curl -sf "$URL/models" > /dev/null && break
  if ! kill -0 $SERVER 2>/dev/null; then
    echo "== $(date) reflection server exited"; tail -n 40 "$OUT/reflect-server-$SLURM_JOB_ID.log"; exit 1
  fi
  sleep 10
done
curl -sf "$URL/models" > /dev/null || { echo "== reflection server not up after 15 min"; tail -n 40 "$OUT/reflect-server-$SLURM_JOB_ID.log"; exit 1; }
echo "== $(date) reflection server up at $URL"

.venv-gpu/bin/python -m codegen.gepa_run --out "$OUT" --reflect-url "$URL" --deadline "$DEADLINE" "$@"
status=$?
kill $SERVER 2>/dev/null

mkdir -p "$WORK/gepa-grpo-runs/$NAME" "results/$NAME"
cp "$OUT"/gepa_summary.json "$OUT"/candidates.json "$OUT"/best_prompt.txt "$OUT"/run_log.txt "$OUT"/evals.jsonl "$OUT"/reflections.jsonl \
  "$WORK/gepa-grpo-runs/$NAME/" 2>/dev/null
cp "$OUT"/gepa_summary.json "$OUT"/best_prompt.txt "results/$NAME/" 2>/dev/null
echo "== $(date) done (exit $status); outputs in $OUT"
[ $status -eq 3 ] && echo "== stopped before the budget; continue with: OUT=$OUT sbatch tacc/gepa.sh $*"
exit $status
