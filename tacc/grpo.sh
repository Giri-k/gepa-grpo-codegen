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
echo "== $(date) $NAME on $(hostname), args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv

.venv-gpu/bin/accelerate launch --num_processes 3 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  -m codegen.grpo --out "$OUT" "$@"
status=$?

# Keep the adapter, logs and run settings; optimizer checkpoints stay on scratch.
mkdir -p "$WORK/gepa-grpo-runs/$NAME" "results/$NAME"
cp -r "$OUT"/final "$WORK/gepa-grpo-runs/$NAME/" 2>/dev/null
state=$OUT/trainer_state.json  # written at the end; a run cut short only has checkpoint copies
[ -f "$state" ] || state=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)/trainer_state.json
cp "$state" "$OUT"/run_args.json "results/$NAME/" 2>/dev/null
echo "== $(date) done (exit $status); outputs in $OUT"
exit $status
