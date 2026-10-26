#!/bin/bash
# Probe a Vista GPU node: hardware, filesystems, and which Apptainer isolation
# features work for a normal user. Read-only apart from a scratch folder.
#
#   cd $WORK/gepa-grpo-codegen && sbatch tacc/probe.sh
#   cat probe-<jobid>.out          # when the job finishes (~5 min)
#
#SBATCH -J probe
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -A CDA26008
#SBATCH -o probe-%j.out

set -u
echo "== node"; hostname; uname -m; nproc; free -g | head -2
echo "== gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep "CUDA Version"
echo "== filesystems"; echo "HOME=$HOME WORK=$WORK SCRATCH=$SCRATCH"; df -h "$WORK" "$SCRATCH" | tail -2

echo "== apptainer"; module load tacc-apptainer; apptainer --version
mkdir -p "$SCRATCH/probe" && cd "$SCRATCH/probe"

echo "== pull a small image from Docker Hub (needs internet on compute nodes)"
time apptainer pull -F py.sif docker://python:3.11-slim-bookworm
apptainer exec py.sif python -c "import sys, platform; print(sys.version.split()[0], platform.machine())"

echo "== are your folders hidden with --containall --no-home?"
apptainer exec --containall --no-home --cleanenv py.sif python -c \
  "import os; print('HOME visible:', os.path.exists('$HOME'), '| WORK visible:', os.path.exists('$WORK'), '| SCRATCH visible:', os.path.exists('$SCRATCH'))"

echo "== network: default (expect reachable)"
apptainer exec --containall py.sif python -c \
  "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3); print('reachable')" 2>&1 | tail -1

echo "== network: --net --network none (want: blocked)"
apptainer exec --containall --net --network none py.sif python -c \
  "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3); print('REACHABLE: not isolated')" 2>&1 | tail -2

echo "== memory limit flag (cgroups)"
apptainer exec --containall --memory 1G py.sif true; echo "exit code: $? (0 = supported)"

echo "== writable tmpfs"
apptainer exec --containall --writable-tmpfs py.sif sh -c "touch /x && echo writable-tmpfs: ok" 2>&1 | tail -1

echo "== modules"; module -t avail cuda python3 gcc 2>&1 | head -20
echo "== done"
