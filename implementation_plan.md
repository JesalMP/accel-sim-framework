# Implementation Plan: Intra-Cluster Coalescing (ICC)

**Paper:** *Intra-Cluster Coalescing to Reduce GPU NoC Pressure*  
**Goal:** Before a cluster injects a `mem_fetch` into the NoC, check if another in-flight read from a **different SM within the same cluster** is targeting the same cache line. If so, merge the two requests into a single NoC packet. When the reply returns, fan it out to all requesting SMs.

---

## Background: The Problem

In the current simulator, each SM independently sends its L1-miss `mem_fetch` packets to the L2 via the NoC. When two SMs in the same cluster miss on the **same cache-line address at the same time**, two identical packets traverse the NoC and two identical replies come back — wasting bandwidth.

The paper proposes a small **Intra-Cluster Coalescing Buffer (ICCB)** at the cluster level that:
1. Holds pending outbound read requests.
2. Checks each new request against the buffer.
3. If a match exists (same address → same L2 sub-partition), merges the new request into the existing one (records a second requester).
4. Sends only one packet to the NoC.
5. On reply, duplicates the response and delivers it to all requesting SMs.

---

## Current Architecture (What Exists)

```
SM0 ──┐
SM1 ──┤  simt_core_cluster::icnt_inject_request_packet()
SM2 ──┤  → ::icnt_push(cluster_id, destination, mf, size)
SM3 ──┘

shader_memory_interface::push(mf)        // shader.h:2790
  → m_cluster->icnt_inject_request_packet(mf)   // shader.cc:4690
      → ::icnt_push(...)                         // shader.cc:4707/4710

simt_core_cluster::icnt_cycle()          // shader.cc:4796
  → ::icnt_pop() → m_response_fifo.push_back(mf)
  → dispatch mf to correct SM via m_core[cid]->accept_ldst_unit_response(mf)
```

---

## Proposed Changes

### Component 1: `mem_fetch.h` — Track Multiple Requesters

**File:** [mem_fetch.h](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/mem_fetch.h)

The `mem_fetch` object needs to carry a list of additional SMs/warps waiting for the same reply.

#### [MODIFY] mem_fetch.h

Add to the **public** section (after line ~130):
```cpp
// ICC: additional requesters merged into this packet at the cluster level
// Each entry is (sid, wid) of a merged requester
std::vector<std::pair<unsigned, unsigned>> m_merged_requesters;

void add_merged_requester(unsigned sid, unsigned wid) {
    m_merged_requesters.push_back({sid, wid});
}
bool has_merged_requesters() const {
    return !m_merged_requesters.empty();
}
const std::vector<std::pair<unsigned, unsigned>>& get_merged_requesters() const {
    return m_merged_requesters;
}
```

> [!NOTE]
> The original `m_sid`/`m_wid` remain as the **primary** requester. Merged requesters are stored in the vector. This avoids breaking any existing code that reads `get_sid()`/`get_wid()`.

---

### Component 2: `shader.h` — ICCB Data Structure in `simt_core_cluster`

**File:** [shader.h](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h)

#### [MODIFY] `simt_core_cluster` class (line 2659–2721)

Add the ICCB as a protected member and expose methods:

```cpp
// --- ICC: Intra-Cluster Coalescing Buffer ---
// Maps cache-line address → pending outbound mem_fetch
// (only READ_REQUEST entries; writes are not coalesced)
std::unordered_map<new_addr_type, mem_fetch *> m_icc_buffer;
unsigned m_icc_buffer_size;   // configurable max entries (from config)
bool m_icc_enabled;           // toggled by config flag

// Returns the cache-line aligned address (block address)
new_addr_type icc_block_addr(new_addr_type addr) const;

// Try to merge mf into ICCB. Returns true if merged (don't inject to NoC).
// Returns false if no match or write request (inject normally).
bool icc_try_merge(mem_fetch *mf);

// Called when a reply arrives: fan out to all merged requesters
void icc_fanout_reply(mem_fetch *mf);

// Remove entry from ICCB when packet departs to NoC
void icc_mark_inflight(new_addr_type addr);
```

Add a new `m_icc_buffer_size` config param read (see `gpgpusim.config` section below).

---

### Component 3: `shader.cc` — ICCB Logic Implementation

**File:** [shader.cc](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc)

#### 3a. `icnt_inject_request_packet()` — Injection with Coalescing (line 4690)

Replace the current direct-inject logic with an ICCB check:

```cpp
void simt_core_cluster::icnt_inject_request_packet(class mem_fetch *mf) {
    // Only coalesce read requests (not writes, not atomics)
    if (m_icc_enabled && !mf->get_is_write() && !mf->isatomic()) {
        if (icc_try_merge(mf)) {
            // Merged into existing in-flight request; don't inject to NoC
            m_stats->icc_merges++;   // new stat counter
            return;
        }
        // No match — add to ICCB as new entry, then inject
        new_addr_type block_addr = icc_block_addr(mf->get_addr());
        m_icc_buffer[block_addr] = mf;
    }

    // Original injection path (unchanged)
    update_icnt_stats(mf);
    unsigned int packet_size = mf->get_ctrl_size();  // reads: ctrl only
    m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
    unsigned destination = mf->get_sub_partition_id();
    mf->set_status(IN_ICNT_TO_MEM, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    ::icnt_push(m_cluster_id, m_config->mem2device(destination),
                (void *)mf, mf->get_ctrl_size());
}
```

#### 3b. `icc_try_merge()` — The Merge Function (new, in shader.cc)

```cpp
bool simt_core_cluster::icc_try_merge(mem_fetch *mf) {
    new_addr_type block_addr = icc_block_addr(mf->get_addr());
    auto it = m_icc_buffer.find(block_addr);
    if (it == m_icc_buffer.end()) return false;

    // Found a pending request for the same cache line
    mem_fetch *leader = it->second;
    leader->add_merged_requester(mf->get_sid(), mf->get_wid());

    // Free the redundant mem_fetch (the leader will carry the reply)
    // But keep a back-pointer so the reply can be delivered
    // We need to store the full mf for the return-path fan-out
    // so instead of freeing, store it in a side table keyed by leader
    m_icc_waiters[leader].push_back(mf);
    return true;
}
```

This requires adding:
```cpp
// Maps leader mf ptr → list of merged (waiting) mf ptrs
std::unordered_map<mem_fetch *, std::vector<mem_fetch *>> m_icc_waiters;
```

#### 3c. `icnt_cycle()` — Fan-Out on Reply (line 4796)

After popping a reply from the NoC (`::icnt_pop()`), check if it was a leader with merged waiters:

```cpp
void simt_core_cluster::icnt_cycle() {
    // ... existing response fifo dispatch code (unchanged) ...

    if (m_response_fifo.size() < m_config->n_simt_ejection_buffer_size) {
        mem_fetch *mf = (mem_fetch *)::icnt_pop(m_cluster_id);
        if (!mf) return;
        assert(mf->get_tpc() == m_cluster_id);

        // ICC: remove from ICCB now that reply has arrived
        if (m_icc_enabled) {
            new_addr_type block_addr = icc_block_addr(mf->get_addr());
            m_icc_buffer.erase(block_addr);

            // Fan-out to merged waiters
            icc_fanout_reply(mf);
        }

        // ... existing push to m_response_fifo (unchanged) ...
    }
}
```

#### 3d. `icc_fanout_reply()` — Deliver Reply to All Waiters (new)

```cpp
void simt_core_cluster::icc_fanout_reply(mem_fetch *leader) {
    auto it = m_icc_waiters.find(leader);
    if (it == m_icc_waiters.end()) return;

    for (mem_fetch *waiter : it->second) {
        // Clone the reply data into the waiter's mf
        // (the waiter mf tracks the original requesting SM/warp)
        waiter->set_reply();
        waiter->set_status(IN_CLUSTER_TO_SHADER_QUEUE,
                           m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        // Inject directly into response fifo (bypass NoC)
        m_response_fifo.push_back(waiter);
        m_stats->icc_fanouts++;
    }
    m_icc_waiters.erase(it);
}
```

#### 3e. `icc_block_addr()` — Address Alignment (new helper)

```cpp
new_addr_type simt_core_cluster::icc_block_addr(new_addr_type addr) const {
    // Align to cache line size (128 bytes on V100)
    unsigned line_sz = m_config->m_L1D_config.get_line_sz(); // or hardcode 128
    return (addr / line_sz) * line_sz;
}
```

---

### Component 4: `gpu-sim.h` / `shader_core_config` — Config Parameters

**File:** [gpu-sim.h](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h)

In `shader_core_config`, add:
```cpp
bool icc_enabled;                // -gpgpu_icc_enable 0/1
unsigned icc_buffer_size;        // -gpgpu_icc_buffer_size N (max pending entries)
```

Register in the option parser (wherever other `-gpgpu_*` options are parsed, likely `gpu-sim.cc` around the `option_parser_register` block):
```cpp
option_parser_register(opp, "-gpgpu_icc_enable", OPT_BOOL, &icc_enabled,
                       "Enable intra-cluster coalescing", "0");
option_parser_register(opp, "-gpgpu_icc_buffer_size", OPT_INT32, &icc_buffer_size,
                       "Max entries in ICC buffer per cluster", "16");
```

---

### Component 5: `gpgpusim.config` — Enable the Feature

**File:** [gpgpusim.config](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config)

Add new lines (e.g., after the interconnect section):
```ini
# Intra-Cluster Coalescing (Narasiman et al.)
-gpgpu_icc_enable 1
-gpgpu_icc_buffer_size 16
```

Set to `0` for the baseline (default) run, `1` for the ICC run.

---

### Component 6: New Statistics

Add to `shader_core_stats` (in `shader.h` or `stats.h`):
```cpp
unsigned long long icc_merges;    // # requests merged (saved NoC packets)
unsigned long long icc_fanouts;   // # synthetic replies generated
```

Print in the simulation summary (in `gpu-sim.cc` where other stats are printed):
```cpp
fprintf(stdout, "ICC merges = %llu\n", m_stats->icc_merges);
fprintf(stdout, "ICC fanouts = %llu\n", m_stats->icc_fanouts);
```

---

## Data Flow Diagram

```
SM0: L1 miss addr=0xABC0 ──► icnt_inject_request_packet()
                                  │
                              icc_try_merge()
                                  │ buffer empty → ADD to m_icc_buffer[0xABC0]
                                  │
                              ::icnt_push() → NoC → L2

SM1: L1 miss addr=0xABC0 ──► icnt_inject_request_packet()
                                  │
                              icc_try_merge()
                                  │ HIT: m_icc_buffer[0xABC0] exists
                                  │ → add SM1/warp to m_icc_waiters[leader]
                                  │ → return (NO icnt_push)
                                  │
                              (packet NOT injected to NoC) ✅

L2 reply arrives ──────────► icnt_cycle() → ::icnt_pop()
                                  │
                              icc_fanout_reply(leader_mf)
                                  │ → push leader_mf to m_response_fifo (SM0)
                                  │ → push waiter_mf to m_response_fifo (SM1)
                                  │   (SM1 gets its own copy, free of charge)
                                  │
                              Both SMs receive their cache line ✅
```

---

## Files Summary

| File | Change Type | What Changes |
|---|---|---|
| [mem_fetch.h](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/mem_fetch.h) | MODIFY | Add `m_merged_requesters` vector + accessors |
| [shader.h](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h) | MODIFY | Add `m_icc_buffer`, `m_icc_waiters`, helper method declarations to `simt_core_cluster` |
| [shader.cc](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) | MODIFY | Modify `icnt_inject_request_packet()`, `icnt_cycle()`; add `icc_try_merge()`, `icc_fanout_reply()`, `icc_block_addr()` |
| [gpu-sim.h](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h) | MODIFY | Add `icc_enabled`, `icc_buffer_size` to `shader_core_config` |
| [gpu-sim.cc](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) | MODIFY | Register config options; add stat printout |
| [gpgpusim.config](file:///d:/projects/proj/Evaluating-Two-Level-Warp-Scheduling-with-AccelSim/accel-sim-framework/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config) | MODIFY | Add `-gpgpu_icc_enable` and `-gpgpu_icc_buffer_size` |

---

## Open Questions

> [!IMPORTANT]
> **1. Write/atomic handling**: The paper only coalesces reads. Writes and atomics are passed through unchanged. This is the safe default — confirm this is your intended scope.

> [!IMPORTANT]
> **2. ICCB size**: The paper uses a small buffer (8–16 entries per cluster). Too large → complex hardware; too small → poor coverage. 16 is a reasonable starting point, but this should be swept experimentally.

> [!WARNING]
> **3. Sector-granularity vs. line-granularity**: The V100 L2 uses 128-byte cache lines divided into 32-byte sectors. The MSHR on the SM side deals in sectors. The icc_block_addr() should align to 128-byte lines (not 32-byte sectors) to maximize coalescing opportunity, but this needs to match how `get_sub_partition_id()` routes requests.

> [!WARNING]
> **4. In-flight vs. pending**: The current plan only merges requests while the leader is **still in the ICCB** (not yet injected to NoC). Once `::icnt_push()` is called, the entry should remain in `m_icc_buffer` (marked inflight) to also catch requests that arrive after departure but before reply. This requires a second state (`pending` vs `inflight`) in the ICCB to avoid re-injecting a second NoC packet.

> [!NOTE]
> **5. Correctness of fan-out**: The waiter's `mem_fetch` object retains its original `m_sid`/`m_wid`, which is what `icnt_cycle()` uses to route the reply back to the correct SM core. No changes are needed to the dispatch logic in `icnt_cycle()` as long as the waiter mf is pushed directly into `m_response_fifo`.

---

## Verification Plan

1. **Build sanity**: compile with `icc_enabled=0` — all existing tests should be identical to baseline.
2. **Functional check**: run with `icc_enabled=1` and verify `gpu_tot_sim_insn` is identical to baseline (same work done).
3. **Stat check**: `icc_merges > 0` confirms coalescing is firing; `icc_fanouts == icc_merges` confirms every merge is followed by a fan-out.
4. **NoC pressure**: compare `total dram reads` and L2 TOTAL_ACCESS between baseline and ICC — you should see a reduction in NoC packets without a change in DRAM reads.
5. **IPC**: run `get_stats.py` to compare `gpu_ipc` between default, 2LRR, and 2LRR+ICC.
