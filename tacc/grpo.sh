#!/bin/bash
# GRPO + LoRA on grpo_train, 3 A100s (one trainer process + colocated vLLM per GPU).
# Extra arguments go to codegen.grpo, e.g. a smoke test that times a few steps:
#
#   cd $WORK/ls6/post-training/gepa-grpo-codegen
#   sbatch -p gpu-a100-dev -t 01:00:00 tacc/grpo.sh --max-steps 3
#   sbatch tacc/grpo.sh --max-steps 80
#   OUT=$SCRATCH/gepa-grpo/grpo-<id> sbatch tacc/grpo.sh --max-steps 80 --resume   # continue a run
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

OUT=${OUT:-$SCRATCH/gepa-grpo/grpo-$SLURM_JOB_ID}
NAME=$(basename "$OUT")
mkdir -p "$OUT"
echo "== $(date) $NAME on $(hostname), args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "$(nvidia-smi -L | grep -c '^GPU')" -lt 3 ]; then
  echo "== $(date) only $(nvidia-smi -L | grep -c '^GPU') GPUs visible on $(hostname); resubmit with --exclude=$(hostname -s)"
  exit 1
fi

WANDB=()
WANDB_PROJECT=${WANDB_PROJECT:-gepa-grpo-codegen}
if [ "$WANDB_PROJECT" != none ]; then
  export WANDB_NAME=${WANDB_NAME:-$NAME} WANDB_DIR=$OUT
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
  -m codegen.grpo --out "$OUT" "${WANDB[@]}" "$@"
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
exit $status
