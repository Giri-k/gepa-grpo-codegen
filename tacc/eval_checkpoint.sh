#!/bin/bash
# Run on a login node (TACC refuses sbatch on compute nodes): wait for a GRPO
# checkpoint, then queue its dev eval (tacc/eval.sh) and the budget-forced
# rescoring of that eval (tacc/force.sh).
#
#   cd $WORK/post-training/gepa-grpo-codegen
#   nohup tacc/eval_checkpoint.sh $SCRATCH/gepa-grpo/grpo-<id> 40 > logs/watch-ckpt40.log 2>&1 &
#
# A checkpoint is complete once its trainer_state.json exists (the Trainer writes
# it last). Settings: SPLIT (default dev), N (default 4).

OUT=$1
STEP=$2
[ -n "$OUT" ] && [ -n "$STEP" ] || { echo "usage: $0 <grpo run dir> <step>"; exit 1; }
CKPT=$OUT/checkpoint-$STEP
SPLIT=${SPLIT:-dev}
NAME=$SPLIT-$(basename "$OUT")-ckpt$STEP
cd "$(dirname "$0")/.."

echo "== $(date) waiting for $CKPT/trainer_state.json"
until [ -f "$CKPT/trainer_state.json" ]; do
  [ -d "$OUT/final" ] && [ ! -d "$CKPT" ] && { echo "== $(date) run finished without $CKPT"; exit 1; }
  sleep 300
done
sleep 60
echo "== $(date) $CKPT is ready; submitting $NAME"
EVAL=$(NAME=$NAME SPLIT=$SPLIT N=${N:-4} LORA=$CKPT sbatch -t 01:30:00 tacc/eval.sh 2>&1 | grep -oE "[0-9]+$" | tail -1)
[ -n "$EVAL" ] || { echo "== $(date) eval submission failed"; exit 1; }
echo "== eval job $EVAL -> $WORK/gepa-grpo-runs/$NAME-$EVAL"
GEN=$WORK/gepa-grpo-runs/$NAME-$EVAL sbatch --dependency=afterok:"$EVAL" tacc/force.sh 2>&1 | tail -1
