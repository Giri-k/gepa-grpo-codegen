#!/bin/bash
# Probe a Lonestar6 A100 node: hardware, which Apptainer isolation features work
# for a normal user, and the sandbox test suite under Apptainer (no Docker on TACC).
#
#   cd $WORK/post-training/gepa-grpo-codegen && sbatch tacc/probe_ls6.sh
#   cat logs/probe-<jobid>.out          # when the job finishes (~10 min)
#
#SBATCH -J probe
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A CDA26008
#SBATCH -o logs/probe-%j.out

# no `set -u`: TACC module scripts reference unset variables
cd "$SLURM_SUBMIT_DIR"
WORKER=$PWD/src/codegen/sandbox/worker.py
export CODEGEN_SANDBOX_SIF=$WORK/sif/python311.sif
SIF=$CODEGEN_SANDBOX_SIF
# Same digest as src/codegen/sandbox/Dockerfile (python:3.11-slim-bookworm). Apptainer rejects tag+digest.
IMAGE=docker://python@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf

echo "== node"; hostname; uname -m; nproc; grep Cpus_allowed_list /proc/self/status; lscpu | grep -E "^CPU\(s\)|Socket"; free -g | head -2; echo "max user processes: $(ulimit -u)"
echo "== gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep "CUDA Version"
echo "== filesystems"; echo "HOME=$HOME WORK=$WORK SCRATCH=$SCRATCH"; df -h "$WORK" "$SCRATCH" | tail -2; ldd --version | head -1

echo "== internet from compute node"
curl -sS -m 5 -o /dev/null -w "huggingface.co: %{http_code}\n" https://huggingface.co || echo "huggingface.co: unreachable"
curl -sS -m 5 -o /dev/null -w "pypi.org: %{http_code}\n" https://pypi.org/simple/ || echo "pypi.org: unreachable"

echo "== apptainer"; module load tacc-apptainer/1.4.1; apptainer --version
mkdir -p "$(dirname "$SIF")"
if [ ! -f "$SIF" ]; then time apptainer pull "$SIF" "$IMAGE"; fi
apptainer exec "$SIF" python -c "import sys, platform; print(sys.version.split()[0], platform.machine())"

# The flags the sandbox uses (src/codegen/sandbox/runner.py: _apptainer_cmd).
FLAGS=(--containall --no-home --cleanenv --pid --no-mount bind-paths,hostfs,cwd --bind "$WORKER:/sandbox/worker.py:ro")

echo "== host folders hidden?"
apptainer exec "${FLAGS[@]}" "$SIF" python -c \
  "import os; print('HOME visible:', os.path.exists('$HOME'), '| WORK visible:', os.path.exists('$WORK'), '| SCRATCH visible:', os.path.exists('$SCRATCH'), '| repo visible:', os.path.exists('$PWD'))"
echo "-- without --no-mount (what site bind paths would expose)"
apptainer exec --containall --no-home --cleanenv "$SIF" python -c \
  "import os; print('WORK visible:', os.path.exists('$WORK'), '| SCRATCH visible:', os.path.exists('$SCRATCH'))"

echo "== pid namespace (want: few pids)"
apptainer exec "${FLAGS[@]}" "$SIF" python -c \
  "import os; p=sorted(int(x) for x in os.listdir('/proc') if x.isdigit()); print('visible pids:', p); print('pid1:', open('/proc/1/cmdline','rb').read().replace(b'\0',b' ')[:80])"

echo "== /tmp writable, root fs read-only?"
apptainer exec "${FLAGS[@]}" "$SIF" sh -c "touch /tmp/x && echo tmp: writable; df -h /tmp | tail -1; touch /x 2>/dev/null && echo 'root: WRITABLE' || echo root: read-only; touch /sandbox/x 2>/dev/null && echo '/sandbox: WRITABLE' || echo /sandbox: read-only"

NET='import socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3); print("open")
except OSError as e:
    print("closed:", e)'
echo "== network: no flag (baseline)";            apptainer exec "${FLAGS[@]}" "$SIF" python -c "$NET" 2>&1 | tail -1
echo "== network: --net --network none";          apptainer exec "${FLAGS[@]}" --net --network none "$SIF" python -c "$NET" 2>&1 | tail -2
echo "== network: unshare -rn inside container";  apptainer exec "${FLAGS[@]}" "$SIF" unshare -rn python -c "$NET" 2>&1 | tail -2
echo "== cgroup limits: --pids-limit / --memory"
apptainer exec "${FLAGS[@]}" --pids-limit 64 "$SIF" true; echo "pids-limit exit: $? (0 = supported)"
apptainer exec "${FLAGS[@]}" --memory 2G "$SIF" true; echo "memory exit: $? (0 = supported)"

echo "== worker cleanup must not kill host processes"
sleep 600 & SENTINEL=$!
printf '%s\n' '{"code": "print(sum(map(int, input().split())))", "inputs": ["1 2\n", "3 4\n"], "timeout": 4, "budget": 30, "memory_mb": 1024, "output_bytes": 4000000}' \
  | apptainer exec "${FLAGS[@]}" "$SIF" python -s /sandbox/worker.py | cut -c1-300
kill -0 $SENTINEL 2>/dev/null && echo "host sentinel: alive (good)" || echo "host sentinel: KILLED (unsafe)"
kill $SENTINEL 2>/dev/null

echo "== container startup time (x5)"
time (for i in 1 2 3 4 5; do apptainer exec "${FLAGS[@]}" "$SIF" true; done)

echo "== sandbox setup chosen by runner.py, and the test suite under Apptainer"
sleep 1800 & SENTINEL=$!
.venv/bin/python -c "
from codegen.sandbox import Sandbox
with Sandbox(workers=2, backend='apptainer') as sb:
    print('isolation:', sb.isolation)
    print(sb.run('print(int(input()) * 2)', ['21\n']))
"
.venv/bin/python -m pytest -v tests/test_sandbox.py tests/test_judge.py tests/test_validate.py 2>&1 | tail -40
kill -0 $SENTINEL 2>/dev/null && echo "host sentinel after tests: alive (good)" || echo "host sentinel after tests: KILLED (unsafe)"
kill $SENTINEL 2>/dev/null

echo "== modules"; module -t avail cuda python3 gcc 2>&1 | head -20
echo "== done"
