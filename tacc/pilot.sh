#!/bin/bash
# Baseline pilot (condition C1 on dev): base Qwen3-4B-Thinking-2507 + original
# prompt, 50 dev problems x 8 samples, 8k output budget. Generates on one A100,
# then scores in the Apptainer sandbox.
#
#   cd $WORK/post-training/gepa-grpo-codegen && sbatch tacc/pilot.sh
#   cat logs/pilot-<jobid>.out
#
# Outputs: $SCRATCH/gepa-grpo/pilot-<jobid>/ (generations, scores, summary),
# copied to $WORK/gepa-grpo-runs/ (scratch is purged) and the summary to results/.
#
#SBATCH -J pilot
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A CDA26008
#SBATCH -o logs/pilot-%j.out

cd "$SLURM_SUBMIT_DIR"
module load tacc-apptainer/1.4.1
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=$WORK/hf HF_HUB_OFFLINE=1  # model was downloaded on a login node
export CODEGEN_SANDBOX_SIF=$WORK/sif/python311.sif
export CUDA_VISIBLE_DEVICES=0  # a 4B model fits one A100; the pilot also measures single-GPU throughput

NAME=pilot-$SLURM_JOB_ID
OUT=$SCRATCH/gepa-grpo/$NAME
echo "== $(date) $NAME on $(hostname), cpus: $(grep Cpus_allowed_list /proc/self/status | cut -f2)"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== $(date) generate"
.venv-gpu/bin/python -m codegen.generate --split dev --limit 50 --n 8 --max-tokens 8192 --out "$OUT" || exit 1

echo "== $(date) score"
.venv/bin/python -m codegen.score --gen "$OUT"
status=$?

mkdir -p "$WORK/gepa-grpo-runs" "results/$NAME"
cp -r "$OUT" "$WORK/gepa-grpo-runs/"
cp "$OUT"/meta.json "$OUT"/summary.json "results/$NAME/" 2>/dev/null
echo "== $(date) done (score exit $status); full outputs in $WORK/gepa-grpo-runs/$NAME"
exit $status
