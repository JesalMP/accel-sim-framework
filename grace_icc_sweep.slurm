#!/bin/bash
#SBATCH --job-name=accel_icc
#SBATCH --array=0-25               # 13 schedulers x 2 ICC states = 26 jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00            # 4h: enough for 9 short-test benchmarks
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# ── Paths ────────────────────────────────────────────────────────────────────
BASE="/scratch/user/nkinnal/accel-sim-icc"
BINARY="${BASE}/gpu-simulator/bin/release/accel-sim.out"
TRACE_BASE="${BASE}/gpu-simulator/hw_run/rodinia_2.0-ft/9.1"
CONFIG_SRC="${BASE}/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100"
RESULTS_DIR="${BASE}/results"
WORKDIR="${BASE}/workdirs/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${RESULTS_DIR}" "${WORKDIR}" logs

module purge
module load GCC/12.2.0 CUDA/11.8
export CUDA_INSTALL_PATH=/sw/eb/sw/CUDA/11.8.0
export LD_LIBRARY_PATH=/scratch/user/nkinnal/accel-sim-icc/gpu-simulator/gpgpu-sim/lib/gcc-12.2.0/cuda-11080/release:${LD_LIBRARY_PATH}

# ── Scheduler / ICC mapping ───────────────────────────────────────────────────
SCHEDULERS=(
  "lrr"
  "gto"
  "two_level_rr:4"
  "two_level_rr:8"
  "two_level_rr:16"
  "lwm:64"
  "lwm:256"
  "lwm_two_level_adaptive:64"
  "lwm_two_level_adaptive:128"
  "lwm_two_level_adaptive:256"
  "two_level_gto:4"
  "two_level_gto:8"
  "two_level_gto:16"
)
ICC_VALS=(0 1)

SCHED_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
ICC_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))
SCHED="${SCHEDULERS[$SCHED_IDX]}"
ICC="${ICC_VALS[$ICC_IDX]}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: scheduler=${SCHED}  icc=${ICC}"

# ── Isolated config copy ──────────────────────────────────────────────────────
cp "${CONFIG_SRC}"/* "${WORKDIR}/"
sed -i "s/-gpgpu_scheduler .*/-gpgpu_scheduler ${SCHED}/"   "${WORKDIR}/gpgpusim.config"
sed -i "s/-gpgpu_icc_enable [01]/-gpgpu_icc_enable ${ICC}/" "${WORKDIR}/gpgpusim.config"

# ── Benchmarks ───────────────────────────────────────────────────────────────
BENCHMARKS=(backprop bfs hotspot kmeans lud nw pathfinder srad_v2 streamcluster)
SAFE_SCHED="${SCHED//:/_}"
OUTFILE="${RESULTS_DIR}/result_${SAFE_SCHED}_icc${ICC}.csv"

echo "scheduler,icc_enabled,benchmark,ipc,icc_merges,noc_pkts" > "${OUTFILE}"

for bench in "${BENCHMARKS[@]}"; do
  tracefile=$(find "${TRACE_BASE}" -maxdepth 2 -path "*/${bench}*/kernelslist.g" 2>/dev/null | head -1)
  if [ -z "$tracefile" ]; then
    echo "  [SKIP] ${bench}: trace not found"
    echo "${SCHED},${ICC},${bench},,," >> "${OUTFILE}"
    continue
  fi

  echo "  Running ${bench} ..."
  sim_out=$(cd "${WORKDIR}" && "${BINARY}" -config ./gpgpusim.config -trace "${tracefile}" 2>&1)

  ipc=$(     echo "$sim_out" | grep "gpu_tot_ipc"                 | tail -1 | awk '{print $NF}')
  merges=$(  echo "$sim_out" | grep "icc_merges"                  | tail -1 | awk '{print $NF}')
  noc_pkts=$(echo "$sim_out" | grep "icnt_total_pkts_simt_to_mem" | tail -1 | awk '{print $NF}')

  echo "    IPC=${ipc}  merges=${merges}  noc=${noc_pkts}"
  echo "${SCHED},${ICC},${bench},${ipc},${merges},${noc_pkts}" >> "${OUTFILE}"
done

echo "Done: ${OUTFILE}"
