#!/bin/bash
# GRPO + LoRA on grpo_train, 3 A100s (one trainer process + colocated vLLM per GPU).
# Extra arguments go to codegen.grpo, e.g. a smoke test that times a few steps:
#
#   cd $WORK/ls6/post-training/gepa-grpo-codegen
#   sbatch -p gpu-a100-dev -t 01:00:00 tacc/grpo.sh --max-steps 3
#   sbatch tacc/grpo.sh --max-steps 80
#   OUT=$SCRATCH/gepa-grpo/grpo-<id> sbatch tacc/grpo.sh --max-steps 80 --resume   # continue a run
#   sbatch -p gpu-a100-dev -t 02:00:00 --exclude=c301-002 tacc/grpo.sh --max-steps 40
#
# Time limits: training saves and stops ~12 min before the job's limit. If
# --max-steps isn't reached, the job resubmits itself with --resume (same queue,
# limit, output directory and W&B run), up to MAX_PARTS jobs (default 5); EXCLUDE
# keeps continuations off bad nodes (default: the first job's --exclude).
#
# Checkpoints (LoRA adapter + optimizer) go to $SCRATCH/gepa-grpo/grpo-<jobid>/;
# the final adapter and logs are copied to $WORK/gepa-grpo-runs/.
#
# Monitoring: per-step timing/memory (perf.rank*.jsonl), GPU samples (gpu.csv) and
# CPU samples (cpu.log) are written to the run directory; summarize them with
#   .venv-gpu/bin/python -m codegen.perf_report $SCRATCH/gepa-grpo/grpo-<jobid>
# Metrics also go to Weights & Biases (project $WANDB_PROJECT, default
# gepa-grpo-codegen; WANDB_PROJECT=none to turn off): live when `wandb login` has
# been run on a login node, otherwise offline, to upload later with
#   .venv-gpu/bin/wandb sync $SCRATCH/gepa-grpo/grpo-<jobid>/wandb/offline-run-*
#
#SBATCH -J grpo
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A CDA26008
#SBATCH -o logs/grpo-%j.out

cd "$SLURM_SUBMIT_DIR"
module load tacc-apptainer/1.4.1
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=$WORK/hf HF_HUB_OFFLINE=1
export CODEGEN_SANDBOX_SIF=$WORK/sif/python311.sif
export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false

export OUT=${OUT:-$SCRATCH/gepa-grpo/grpo-$SLURM_JOB_ID} PART=${PART:-1}
NAME=$(basename "$OUT")
MAX_PARTS=${MAX_PARTS:-5}
export EXCLUDE=${EXCLUDE:-$(scontrol show job "$SLURM_JOB_ID" | grep -o 'ExcNodeList=[^ ]*' | cut -d= -f2 | grep -v null)}
mkdir -p "$OUT"
# Minutes left ([D-]H:MM:SS); stop training 12 min before the end to save and copy.
LIMIT_STR=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
LEFT_MIN=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF == 4) print $1*1440 + $2*60 + $3; else if (NF == 3) print $1*60 + $2; else print $1 }')
DEADLINE=$(( $(date +%s) + 60 * (LEFT_MIN - 12) ))
echo "== $(date) $NAME part $PART/$MAX_PARTS on $(hostname) (limit $LIMIT_STR, stop by $(date -d @$DEADLINE +%H:%M)), args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "$(nvidia-smi -L | grep -c '^GPU')" -lt 3 ]; then
  echo "== $(date) only $(nvidia-smi -L | grep -c '^GPU') GPUs visible on $(hostname); resubmit with --exclude=$(hostname -s)"
  exit 1
fi

# Read the model from the node's disk: three processes loading it cold from $WORK took ~5 min.
# Copy it from $SCRATCH when a copy is there ($WORK read at ~4 MB/s on 2026-11-03: a
# 15 min copy, vs 43 s from $SCRATCH); scratch is purged, so fall back to $WORK.
MODEL_DIR=hub/models--Qwen--Qwen3-4B-Thinking-2507
SRC=$SCRATCH/hf
[ -d "$SRC/$MODEL_DIR/snapshots" ] || SRC=$HF_HOME
if mkdir -p /tmp/hf/hub && cp -a "$SRC/$MODEL_DIR" /tmp/hf/hub/; then
  export HF_HOME=/tmp/hf
  echo "== $(date) model copied to /tmp/hf from $SRC"
fi

WANDB=()
WANDB_PROJECT=${WANDB_PROJECT:-gepa-grpo-codegen}
if [ "$WANDB_PROJECT" != none ]; then
  # One W&B run across continuation jobs.
  export WANDB_NAME=${WANDB_NAME:-$NAME} WANDB_DIR=$OUT WANDB_RUN_ID=${WANDB_RUN_ID:-$NAME} WANDB_RESUME=allow
  # Ask wandb itself: newer versions store the key under a CoreWeave host in ~/.netrc.
  if ! .venv-gpu/bin/python -c "import sys, wandb; sys.exit(0 if wandb.Api().api_key else 1)" 2>/dev/null; then
    export WANDB_MODE=offline
    echo "== no wandb login found; logging offline to $OUT/wandb (upload later with wandb sync)"
  fi
  WANDB=(--wandb-project "$WANDB_PROJECT")
fi

# Sample GPUs and CPUs every 5 s for codegen.perf_report.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,nounits -l 5 >> "$OUT/gpu.csv" &
SMI=$!
vmstat -t 5 >> "$OUT/cpu.log" &
VMSTAT=$!

.venv-gpu/bin/accelerate launch --num_processes 3 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  -m codegen.grpo --out "$OUT" --deadline "$DEADLINE" "${WANDB[@]}" "$@"
status=$?
kill $SMI $VMSTAT 2>/dev/null
.venv-gpu/bin/python -m codegen.perf_report "$OUT"

# Keep the adapter, logs and run settings; optimizer checkpoints stay on scratch.
mkdir -p "$WORK/gepa-grpo-runs/$NAME" "results/$NAME"
cp -r "$OUT"/final "$WORK/gepa-grpo-runs/$NAME/" 2>/dev/null
state=$OUT/trainer_state.json  # written at the end; a run cut short only has checkpoint copies
[ -f "$state" ] || state=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)/trainer_state.json
cp "$state" "$OUT"/run_args.json "$OUT"/perf.rank*.jsonl "$OUT"/gpu.csv "results/$NAME/" 2>/dev/null
echo "== $(date) done (exit $status); outputs in $OUT"

# Stopped at the deadline with a checkpoint: continue in a new job.
if [ $status -eq 0 ] && [ ! -d "$OUT/final" ] && ls -d "$OUT"/checkpoint-* >/dev/null 2>&1; then
  ARGS=("$@")
  [[ " $* " == *" --resume "* ]] || ARGS+=(--resume)
  if [ "$PART" -ge "$MAX_PARTS" ]; then
    echo "== MAX_PARTS=$MAX_PARTS reached; continue with: OUT=$OUT sbatch -p $SLURM_JOB_PARTITION -t $LIMIT_STR tacc/grpo.sh ${ARGS[*]}"
  else
    echo "== $(date) submitting part $((PART + 1))"
    PART=$((PART + 1)) sbatch -p "$SLURM_JOB_PARTITION" -t "$LIMIT_STR" ${EXCLUDE:+--exclude="$EXCLUDE"} tacc/grpo.sh "${ARGS[@]}"
  fi
fi
exit $status
