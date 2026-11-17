#!/bin/bash
# One-time Vista (GH200, aarch64) setup check, run before the first GRPO job:
# builds the arm64 sandbox image (Apptainer is not allowed on login nodes), then
# checks CUDA torch, a short vLLM generation and the sandbox tests under Apptainer.
#
#   cd $WORK/post-training/gepa-grpo-codegen && sbatch tacc/setup_vista.sh
#   cat logs/setup-<jobid>.out          # ~15 min
#
# The venvs are built on a login node beforehand (aarch64 torch on PyPI is CPU-only,
# so torch comes from the PyTorch cu128 index):
#   uv venv .venv-gpu --python 3.12
#   uv pip install -p .venv-gpu vllm==0.11.2 trl==0.26.2 peft==0.21.2 transformers==4.57.6 \
#     accelerate==1.15.0 wandb==0.31.0 datasets==5.0.1 gepa==0.1.4 pytest -e .
#   uv pip install -p .venv-gpu --reinstall-package torch --index https://download.pytorch.org/whl/cu128 \
#     --index-strategy unsafe-best-match torch==2.9.0+cu128 torchvision==0.24.0 torchaudio==2.9.0
#   uv sync                                   # .venv, used for scoring
#
#SBATCH -J setup
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72  # Vista binds a 1-task job to one core otherwise
#SBATCH -t 00:30:00
#SBATCH -A CDA26008
#SBATCH -o logs/setup-%j.out

cd "$SLURM_SUBMIT_DIR"
module load tacc-apptainer/1.4.1
unset PYTHONPATH PYTHONHOME
export CC=gcc CXX=g++  # Triton/inductor compile helpers with $CC; TACC sets it to nvc, which fails
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1
export CODEGEN_SANDBOX_SIF=$WORK/sif/python311.sif
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer-cache
# Same digest as src/codegen/sandbox/Dockerfile (a multi-arch index, so this pulls arm64).
IMAGE=docker://python@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf

echo "== $(date) $(hostname) $(uname -m), $(nproc) cpus (affinity $(grep Cpus_allowed_list /proc/self/status | cut -f2)), max user processes $(ulimit -u)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep "CUDA Version"

echo "== sandbox image"
mkdir -p "$(dirname "$CODEGEN_SANDBOX_SIF")" "$SCRATCH/sif"
[ -f "$CODEGEN_SANDBOX_SIF" ] || apptainer pull "$CODEGEN_SANDBOX_SIF" "$IMAGE" || exit 1
cp "$CODEGEN_SANDBOX_SIF" "$SCRATCH/sif/"  # grpo.sh copies it to the node from here
apptainer exec "$CODEGEN_SANDBOX_SIF" python -c "import sys, platform; print(sys.version.split()[0], platform.machine())"

echo "== torch"
.venv-gpu/bin/python -c "
import torch
print(torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); print('matmul ok', (x @ x).float().abs().mean().item() > 0)
" || exit 1

echo "== vllm: short generation with Qwen3-4B-Thinking-2507"
.venv-gpu/bin/python - <<'EOF' || exit 1
import time
from vllm import LLM, SamplingParams
t = time.time()
llm = LLM("Qwen/Qwen3-4B-Thinking-2507", max_model_len=4096, gpu_memory_utilization=0.5, seed=0)
print(f"loaded in {time.time() - t:.0f}s")
t = time.time()
outs = llm.generate(["Write a Python one-liner that prints the sum of two integers read from stdin."] * 64,
                    SamplingParams(temperature=0.6, max_tokens=1024, seed=0))
tokens = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"64 completions, {tokens} tokens in {time.time() - t:.0f}s ({tokens / (time.time() - t):.0f} tok/s)")
EOF

echo "== trl / peft import"
.venv-gpu/bin/python -c "import trl, peft, accelerate; print('trl', trl.__version__, 'peft', peft.__version__, 'accelerate', accelerate.__version__)" || exit 1

echo "== sandbox under apptainer"
sleep 1800 & SENTINEL=$!
.venv/bin/python -c "
from codegen.sandbox import Sandbox
with Sandbox(workers=2, backend='apptainer') as sb:
    print('isolation:', sb.isolation)
    print(sb.run('print(int(input()) * 2)', ['21\n']))
" || exit 1
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_sandbox.py tests/test_judge.py tests/test_validate.py 2>&1 | tail -5
kill -0 $SENTINEL 2>/dev/null && echo "host sentinel after tests: alive (good)" || echo "host sentinel after tests: KILLED (unsafe)"
kill $SENTINEL 2>/dev/null
echo "== $(date) done"
