#!/bin/bash
#SBATCH --job-name=accel_icc_r31
#SBATCH --array=0-25               # 13 schedulers x 2 ICC states = 26 jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00            # 6h: rodinia-3.1 large inputs run longer
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# ── Paths ────────────────────────────────────────────────────────────────────
BASE="/scratch/user/nkinnal/accel-sim-icc"
BINARY="${BASE}/gpu-simulator/bin/release/accel-sim.out"
TRACE_BASE="${BASE}/gpu-simulator/hw_run/rodinia-3.1/11.0"
CONFIG_SRC="${BASE}/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100"
RESULTS_DIR="${BASE}/results_rodinia31"
WORKDIR="${BASE}/workdirs/r31_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

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

# ── Benchmarks (rodinia-3.1 large inputs) ────────────────────────────────────
declare -A TRACES
TRACES[bfs]="bfs-rodinia-3.1/__data_graph1MW_6_txt/traces/kernelslist.g"
TRACES[backprop]="backprop-rodinia-3.1/65536/traces/kernelslist.g"
TRACES[hotspot]="hotspot-rodinia-3.1/1024_2_2___data_temp_1024___data_power_1024_output_out/traces/kernelslist.g"
TRACES[kmeans]="kmeans-rodinia-3.1/_o__i___data_819200_txt/traces/kernelslist.g"
TRACES[lud]="lud-rodinia-3.1/_i___data_512_dat/traces/kernelslist.g"
TRACES[gaussian]="gaussian-rodinia-3.1/_s_256/traces/kernelslist.g"
TRACES[nw]="nw-rodinia-3.1/2048_10/traces/kernelslist.g"
TRACES[pathfinder]="pathfinder-rodinia-3.1/100000_100_20___result_txt/traces/kernelslist.g"
TRACES[dwt2d]="dwt2d-rodinia-3.1/__data_rgb_bmp__d_1024x1024__f__5__l_3/traces/kernelslist.g"
TRACES[hybridsort]="hybridsort-rodinia-3.1/__data_500000_txt/traces/kernelslist.g"
TRACES[lavaMD]="lavaMD-rodinia-3.1/_boxes1d_10/traces/kernelslist.g"
TRACES[btree]="b+tree-rodinia-3.1/file___data_mil_txt_command___data_command_txt/traces/kernelslist.g"
TRACES[srad]="srad_v1-rodinia-3.1/100_0_5_502_458/traces/kernelslist.g"
TRACES[particlefilter]="particlefilter_float-rodinia-3.1/_x_128__y_128__z_10__np_1000/traces/kernelslist.g"

SAFE_SCHED="${SCHED//:/_}"
OUTFILE="${RESULTS_DIR}/result_${SAFE_SCHED}_icc${ICC}.csv"

echo "scheduler,icc_enabled,benchmark,ipc,icc_merges,noc_pkts" > "${OUTFILE}"

for bench in "${!TRACES[@]}"; do
  tracefile="${TRACE_BASE}/${TRACES[$bench]}"
  if [ ! -f "$tracefile" ]; then
    echo "  [SKIP] ${bench}: trace not found at ${tracefile}"
    echo "${SCHED},${ICC},${bench},,," >> "${OUTFILE}"
    continue
  fi

  echo "  Running ${bench} ..."
  sim_out=$(cd "${WORKDIR}" && "${BINARY}" -config ./gpgpusim.config -trace "${tracefile}" 2>&1)

  ipc=$(     echo "$sim_out" | grep "gpu_tot_ipc"                 | tail -1 | awk '{print $NF}')
  merges=$(  echo "$sim_out" | grep "icc_merges"                  | tail -1 | awk '{print $NF}')
  noc_pkts=$(echo "$sim_out" | grep "icnt_total_pkts_simt_to_mem" | tail -1 | awk -F= '{print $NF}')

  echo "    IPC=${ipc}  merges=${merges}  noc=${noc_pkts}"
  echo "${SCHED},${ICC},${bench},${ipc},${merges},${noc_pkts}" >> "${OUTFILE}"
done

echo "Done: ${OUTFILE}"
