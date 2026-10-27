#!/bin/bash
# Evaluate one condition: generate on all 3 A100s (one vLLM per GPU), then score
# in the Apptainer sandbox. Settings come from environment variables:
#
#   SPLIT   split to evaluate (default eval)
#   N       samples per problem (default 4)
#   LIMIT   random subset of this many problems (default: all)
#   PROMPT  system prompt name in src/codegen/prompts (default original)
#   LORA    LoRA adapter directory (default: none, the base model)
#   NAME    run name (default <condition>-<jobid>)
#
#   cd $WORK/ls6/post-training/gepa-grpo-codegen
#   NAME=c1 sbatch tacc/eval.sh                                    # C1: base + original
#   NAME=c3 LORA=$WORK/gepa-grpo-runs/grpo-<id>/final sbatch tacc/eval.sh   # C3
#
# Outputs: $SCRATCH/gepa-grpo/<name>/, copied to $WORK/gepa-grpo-runs/ and the
# summary to results/<name>/.
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
NAME=${NAME:-eval}-$SLURM_JOB_ID
OUT=$SCRATCH/gepa-grpo/$NAME
EXTRA=()
[ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")
[ -n "$LORA" ] && EXTRA+=(--lora "$LORA")

echo "== $(date) $NAME on $(hostname): split=$SPLIT n=$N prompt=$PROMPT lora=${LORA:-none} limit=${LIMIT:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== $(date) generate"
.venv-gpu/bin/python -m codegen.generate --split "$SPLIT" --n "$N" --system-prompt "$PROMPT" --max-tokens 8192 \
  --gpus 3 "${EXTRA[@]}" --out "$OUT" || { tail -30 "$OUT"/shard*.log; exit 1; }

echo "== $(date) score"
.venv/bin/python -m codegen.score --gen "$OUT"
status=$?

mkdir -p "$WORK/gepa-grpo-runs" "results/$NAME"
cp -r "$OUT" "$WORK/gepa-grpo-runs/"
cp "$OUT"/meta.json "$OUT"/summary.json "results/$NAME/" 2>/dev/null
echo "== $(date) done (score exit $status); full outputs in $WORK/gepa-grpo-runs/$NAME"
exit $status
