#!/bin/bash
# Budget forcing for a finished run (see codegen.force_answer): samples cut off
# while reasoning get an early-stop sentence, a closed </think> and up to
# ANSWER_TOKENS (default 2048) more tokens for the answer; then the whole run is
# scored in the sandbox. One GH200.
#
#   cd $WORK/post-training/gepa-grpo-codegen
#   GEN=$WORK/../ls6/gepa-grpo-runs/c1-3538415 sbatch tacc/force.sh     # C1, budget-forced
#
# Outputs: $SCRATCH/gepa-grpo/<run>-forced/, copied to $WORK/gepa-grpo-runs/ and
# the summary to results/<run>-forced/.
#
#SBATCH -J force
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72  # Vista binds a 1-task job to one core otherwise
#SBATCH -t 01:00:00
#SBATCH -A CDA26008
#SBATCH -o logs/force-%j.out

cd "$SLURM_SUBMIT_DIR"
module load tacc-apptainer/1.4.1
unset PYTHONPATH PYTHONHOME
export CC=gcc CXX=g++  # Triton/inductor compile helpers with $CC; TACC sets it to nvc, which fails
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1
export CODEGEN_SANDBOX_SIF=$SCRATCH/sif/python311.sif
[ -f "$CODEGEN_SANDBOX_SIF" ] || export CODEGEN_SANDBOX_SIF=$WORK/sif/python311.sif

[ -n "$GEN" ] || { echo "set GEN to a finished run directory"; exit 1; }
NAME=$(basename "$GEN")-forced
OUT=$SCRATCH/gepa-grpo/$NAME
echo "== $(date) $NAME on $(hostname) from $GEN"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== $(date) force answers"
.venv-gpu/bin/python -m codegen.force_answer --gen "$GEN" --out "$OUT" --answer-tokens "${ANSWER_TOKENS:-2048}" || exit 1

echo "== $(date) score"
.venv/bin/python -m codegen.score --gen "$OUT"
status=$?

mkdir -p "$WORK/gepa-grpo-runs" "results/$NAME"
cp -r "$OUT" "$WORK/gepa-grpo-runs/"
cp "$OUT"/meta.json "$OUT"/summary.json "results/$NAME/" 2>/dev/null
echo "== $(date) done (score exit $status); full outputs in $WORK/gepa-grpo-runs/$NAME"
exit $status
