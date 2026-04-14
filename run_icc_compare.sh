#!/bin/bash
# run_icc_compare.sh
# Sweeps all schedulers x {ICC off, ICC on} on rodinia_2.0-ft short tests.
# Run from inside Docker at the SM7_QV100 config directory:
#   cd /accel-sim/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100
#   bash /accel-sim/run_icc_compare.sh

set -euo pipefail

OUTFILE="/accel-sim/results_icc_compare.csv"
BINARY="/accel-sim/gpu-simulator/bin/release/accel-sim.out"
TRACE_BASE="/accel-sim/hw_run/rodinia_2.0-ft/9.1"
CONFIG="./gpgpusim.config"

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

BENCHMARKS=(backprop bfs hotspot kmeans lud nw pathfinder srad_v2 streamcluster)

echo "scheduler,icc_enabled,benchmark,ipc,icc_merges,noc_pkts" > "$OUTFILE"

for icc in 0 1; do
  echo ""
  echo "============================================================"
  echo "  ICC = ${icc}"
  echo "============================================================"

  # Set ICC flag in config
  sed -i "s/-gpgpu_icc_enable [01]/-gpgpu_icc_enable ${icc}/" "$CONFIG"

  for sched in "${SCHEDULERS[@]}"; do
    echo ""
    echo "--- Scheduler: ${sched}  ICC: ${icc} ---"

    # Set scheduler in config
    sed -i "s/-gpgpu_scheduler .*/-gpgpu_scheduler ${sched}/" "$CONFIG"

    for bench in "${BENCHMARKS[@]}"; do
      tracefile=$(find "${TRACE_BASE}/${bench}"* -name "kernelslist.g" 2>/dev/null | head -1)
      if [ -z "$tracefile" ]; then
        echo "  [SKIP] ${bench}: trace not found"
        echo "${sched},${icc},${bench},,,," >> "$OUTFILE"
        continue
      fi

      sim_out=$("$BINARY" -config "$CONFIG" -trace "$tracefile" 2>&1)

      ipc=$(echo "$sim_out"      | grep "gpu_tot_ipc"          | tail -1 | awk '{print $NF}')
      merges=$(echo "$sim_out"   | grep "icc_merges"            | tail -1 | awk '{print $NF}')
      noc_pkts=$(echo "$sim_out" | grep "icnt_total_injected_pkts\|n_simt_to_mem" | tail -1 | awk '{print $NF}')

      echo "  ${bench}: IPC=${ipc}  merges=${merges}  noc=${noc_pkts}"
      echo "${sched},${icc},${bench},${ipc},${merges},${noc_pkts}" >> "$OUTFILE"
    done
  done
done

echo ""
echo "Done. Results in: $OUTFILE"
