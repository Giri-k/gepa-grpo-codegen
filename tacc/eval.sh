#!/bin/bash
# Evaluate one condition: generate on all 3 A100s (one vLLM per GPU), then score
# in the Apptainer sandbox. Settings come from environment variables:
#
#   SPLIT      split to evaluate (default eval)
#   N          samples per problem (default 4)
#   LIMIT      random subset of this many problems (default: all)
#   PROMPT     system prompt name in src/codegen/prompts (default original)
#   LORA       LoRA adapter directory (default: none, the base model)
#   NAME       condition name; the run is <name>-<first jobid> (default eval)
#   MAX_PARTS  most jobs one run may take (default 3)
#   EXCLUDE    nodes to keep continuation jobs off, e.g. one with a dead GPU
#
#   cd $WORK/ls6/post-training/gepa-grpo-codegen
#   NAME=c1 sbatch tacc/eval.sh                                    # C1: base + original
#   NAME=c1 sbatch -p gpu-a100-dev -t 02:00:00 tacc/eval.sh        # same, on the dev queue
#   NAME=c3 LORA=$WORK/gepa-grpo-runs/grpo-<id>/final sbatch tacc/eval.sh   # C3
#
# Generation saves after every chunk of problems and stops 15 minutes before the
# time limit. If problems are left, the job submits a continuation of itself
# (same queue and limit, same output directory) that picks up where it stopped;
# scoring runs in the part that finishes generation. To continue a run by hand:
#   RUN=c1-<first jobid> sbatch -p gpu-a100-dev -t 02:00:00 tacc/eval.sh
#
# Outputs: $SCRATCH/gepa-grpo/<run>/, copied to $WORK/gepa-grpo-runs/ and the
# summary to results/<run>/.
#
#SBATCH -J eval
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:30:00
#SBATCH -A CDA26008
#SBATCH -o logs/eval-%j.out

cd "$SLURM_SUBMIT_DIR"
module load tacc-apptainer/1.4.1
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=$WORK/hf HF_HUB_OFFLINE=1
export CODEGEN_SANDBOX_SIF=$WORK/sif/python311.sif
export CUDA_VISIBLE_DEVICES=0,1,2

SPLIT=${SPLIT:-eval}
N=${N:-4}
PROMPT=${PROMPT:-original}
export RUN=${RUN:-${NAME:-eval}-$SLURM_JOB_ID} PART=${PART:-1}
MAX_PARTS=${MAX_PARTS:-3}
EXCLUDE=${EXCLUDE:-$SBATCH_EXCLUDE}  # TACC's sbatch ignores SBATCH_EXCLUDE, so pass it on explicitly
OUT=$SCRATCH/gepa-grpo/$RUN
EXTRA=()
[ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")
[ -n "$LORA" ] && EXTRA+=(--lora "$LORA")

# Minutes left in the allocation ([D-]H:MM:SS; less than the limit when run by
# hand inside an idev session); leave 15 for scoring and copying.
LIMIT_STR=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
LEFT_MIN=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF == 4) print $1*1440 + $2*60 + $3; else if (NF == 3) print $1*60 + $2; else print $1 }')
GEN_MIN=$((LEFT_MIN - 15))

echo "== $(date) $RUN part $PART/$MAX_PARTS on $(hostname) ($SLURM_JOB_PARTITION, limit $LIMIT_STR):" \
  "split=$SPLIT n=$N prompt=$PROMPT lora=${LORA:-none} limit=${LIMIT:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv
# A node can lose a GPU without Slurm noticing (c301-002 showed 2 of 3 on 2026-10-27);
# fail before loading anything. Resubmit with --exclude=<node>.
if [ "$(nvidia-smi -L | grep -c '^GPU')" -lt 3 ]; then
  echo "== $(date) only $(nvidia-smi -L | grep -c '^GPU') GPUs visible on $(hostname); resubmit with --exclude=$(hostname -s)"
  exit 1
fi

echo "== $(date) generate (stops after $GEN_MIN min)"
.venv-gpu/bin/python -m codegen.generate --split "$SPLIT" --n "$N" --system-prompt "$PROMPT" --max-tokens 8192 \
  --gpus 3 --stop-after "$GEN_MIN" "${EXTRA[@]}" --out "$OUT"
gen=$?
grep -h "chunk .* done\|already done\|stopping" "$OUT"/shard*.log
if [ $gen -eq 3 ]; then
  if [ "$PART" -ge "$MAX_PARTS" ]; then
    echo "== $(date) stopped early and MAX_PARTS=$MAX_PARTS reached; continue with: RUN=$RUN sbatch -p $SLURM_JOB_PARTITION -t $LIMIT_STR tacc/eval.sh"
    exit 3
  fi
  echo "== $(date) generation incomplete; submitting part $((PART + 1))"
  PART=$((PART + 1)) sbatch -p "$SLURM_JOB_PARTITION" -t "$LIMIT_STR" ${EXCLUDE:+--exclude="$EXCLUDE"} tacc/eval.sh
  exit 0
elif [ $gen -ne 0 ]; then
  tail -n 30 "$OUT"/shard*.log
  exit 1
fi

echo "== $(date) score"
.venv/bin/python -m codegen.score --gen "$OUT"
status=$?

mkdir -p "$WORK/gepa-grpo-runs" "results/$RUN"
cp -r "$OUT" "$WORK/gepa-grpo-runs/"
cp "$OUT"/meta.json "$OUT"/summary.json "results/$RUN/" 2>/dev/null
echo "== $(date) done (score exit $status); full outputs in $WORK/gepa-grpo-runs/$RUN"
exit $status
