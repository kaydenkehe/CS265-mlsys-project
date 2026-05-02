"""
phase2.py — μ-TWO Activation Checkpointing Algorithm

Consumes a profiled GraphProfiler (Phase 1) and decides which activations to
discard in the forward pass and recompute in the backward pass, subject to a
target memory budget. The output is a dict consumed by Phase 3 to rewrite the
graph. Keys:
  - to_recompute:           Set[fx.Node]                — activations to evict
  - recompute_subgraphs:    Dict[fx.Node, List[fx.Node]] — A → forward nodes to replay
  - recompute_inputs:       Dict[fx.Node, List[fx.Node]] — A → checkpointed inputs
  - peak_bytes:             int                          — estimated peak after eviction
  - recompute_overhead_ms:  float                        — estimated total recompute time

Three pieces:
  1. compute_recompute_subgraph(target, checkpointed)
       Static analysis. Backward DFS over forward predecessors of `target`,
       stopping at any node in `checkpointed`. Returns the ordered set of
       forward nodes to replay and the inputs it reads.

  2. simulate_peak_memory(profiler, evicted)
       Linear-pass memory simulator. Models each tensor's liveness intervals,
       reallocating evicted activations at their first backward use. Used both
       as the greedy loop's stopping criterion and as a candidate-scoring oracle.

  3. run_mutwo_algorithm(profiler, memory_budget_bytes)
       While peak > budget, pick the highest size/recompute_time
       activation, mark it for recomputation, simulate, repeat.

Approximations (first cut, marked TODO inline):
  - Recompute intermediates not modeled in the simulator (only the target
    activation is re-allocated; its subgraph's intermediate tensors are not).
  - Negative mem_deltas (in-place frees) ignored.
  - Single re-allocation lifetime for evicted activations (no overlap modeling
    if multiple recomputed activations spike together).
"""

from typing import List, Tuple, Set
import torch.fx as fx
from graph_prof import GraphProfiler, OP


# =============================================================================
# Recompute subgraph extraction
# =============================================================================

def compute_recompute_subgraph(
    profiler: GraphProfiler,
    target: fx.Node,
    checkpointed: Set[fx.Node],
) -> Tuple[List[fx.Node], List[fx.Node]]:
    """
    Backward DFS from `target` over forward predecessors, stopping at any node
    in `checkpointed`. Returns:
      subgraph: nodes to replay to regenerate target (topo order, includes target)
      inputs:   nodes the subgraph reads but does not define (subset of checkpointed)

    Caller must ensure target is NOT in checkpointed (otherwise the subgraph is
    empty — the target is already available).
    """
    visited = set() # nodes in the subgraph (including target)
    inputs = set() # values relied on by subgraph

    def dfs(node: fx.Node) -> None:
        # base cases: already visited, or a checkpointed node
        if node in visited or node in inputs:
            return
        # if we hit a checkpointed node, we don't need to go further back
        if node in checkpointed:
            inputs.add(node)
            return
        # otherwise, visit this node and continue searching on its inputs
        visited.add(node)
        for inp in node.all_input_nodes:
            dfs(inp)

    dfs(target)

    # topo sort the visited nodes and inputs according to their order in the original graph
    idx = profiler.node_index # dict that maps node to its position
    subgraph = sorted(visited, key=idx.__getitem__)
    input_list = sorted(inputs, key=idx.__getitem__)
    return subgraph, input_list


# sum per-node average execution time in ms for subgraph
# (total time it would take to recompute a subgraph)
def recompute_time_ms(profiler: GraphProfiler, subgraph: List[fx.Node]) -> float:
    return sum(profiler.avg_times.get(n.name, 0.0) for n in subgraph)


# =============================================================================
# Memory simulator
# =============================================================================

def simulate_peak_memory(profiler: GraphProfiler, evicted: Set[fx.Node]) -> int:
    """
    Walk the graph in topological order, tracking the sum of live tensor sizes
    at each step. Returns the maximum.

    Liveness model per tensor T (with mem_delta(T) > 0 and at least one user):
      Non-evicted:     alive on [creation_step, last_user_step]
      Evicted (act A): alive on [creation_step, last_fwd_use_step]
                              and [first_bwd_use_step, last_bwd_use_step]

    Breaks ties at the same step in favor of allocations
    """
    events = []  # (step, signed_delta)

    for node in profiler.module.graph.nodes:
        # proxy for the size of the tensor produced by a node
        size = profiler.avg_mem_deltas.get(node.name, 0)

        # we may slightly overestimate the peak by ignoring negative mem deltas
        # avg mem delta is an imperfect proxy, and including negative delta nodes
        # would add complexity and double-count frees
        # we ignore nodes with no users since they're immediately freed
        if size <= 0 or not node.users:
            continue

        creation = profiler.node_index[node]
        last_user = max(profiler.node_index[u] for u in node.users)

        # i.e., if we recompute
        if node in evicted:
            last_fwd = profiler.node_index[profiler.last_forward_access[node]]
            first_bwd = profiler.node_index[profiler.first_backward_access[node]]

            # Forward lifetime
            events.append((creation, +size))
            events.append((last_fwd + 1, -size))

            # Recomputed lifetime in backward
            # TODO: doesn't model the recompute subgraph's intermediate tensors,
            # which briefly occupy memory near first_bwd. For tight budgets this
            # under-estimates peak; refine by simulating each evicted A's subgraph.
            events.append((first_bwd, +size))
            events.append((last_user + 1, -size))
        else:
            events.append((creation, +size))
            events.append((last_user + 1, -size))

    # sort by step, breaking ties in favor of smaller -e[1] (i.e., +alloc before -free)
    events.sort(key=lambda e: (e[0], -e[1]))

    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        if current > peak:
            peak = current
    return peak


# =============================================================================
# Greedy algorithm — mu-two 
# =============================================================================

def run_mutwo_algorithm(
    profiler: GraphProfiler,
    memory_budget_bytes: int,
) -> dict:
    """
    Iteratively pick the activation with the best size/recompute_time ratio,
    mark it for recomputation, re-simulate, and stop once peak ≤ budget (or no
    more candidates remain).
    """

    placeholders = {n for n in profiler.module.graph.nodes if n.op == OP.PLACEHOLDER}
    activations = set(profiler.activation_nodes)

    evicted = set()
    candidates = set(activations)
    peak = simulate_peak_memory(profiler, evicted)

    while candidates and peak > memory_budget_bytes:
        # we keep all placeholders and all activations that haven't been evicted yet
        kept_base = placeholders | (activations - evicted)

        best = None # best candidate for eviction
        best_ratio = -1.0 # size / recompute_time ratio of the best candidate

        for cand in candidates:
            # again, mem delta is an imperfect proxy for size
            size = profiler.avg_mem_deltas.get(cand.name, 0)
            if size <= 0:
                continue

            kept = kept_base - {cand} # cand is no longer kept
            subgraph, _ = compute_recompute_subgraph(profiler, cand, kept)
            recompute_time = recompute_time_ms(profiler, subgraph)

            # avoid div by 0, potential weird cuda timing noise
            if recompute_time <= 0:
                continue

            ratio = size / recompute_time  # bytes saved per ms of recompute (higher is better)
            if ratio > best_ratio:
                best_ratio = ratio
                best = cand

        # we can't make progress
        if best is None:
            break

        evicted.add(best)
        candidates.remove(best)
        peak = simulate_peak_memory(profiler, evicted)

    # Finalize: compute each evicted activation's subgraph against the FINAL
    # kept set. This is what Phase 3 will paste into the backward pass.
    final_kept = placeholders | (activations - evicted)
    subgraphs = {}
    inputs = {}
    overhead = 0.0
    for a in evicted:
        subgraph, input_list = compute_recompute_subgraph(profiler, a, final_kept)
        subgraphs[a] = subgraph
        inputs[a] = input_list
        overhead += recompute_time_ms(profiler, subgraph)

    return {
        "to_recompute": evicted,
        "recompute_subgraphs": subgraphs,
        "recompute_inputs": inputs,
        # estimates
        "peak_bytes": peak,
        "recompute_overhead_ms": overhead,
    }


# =============================================================================
# Reporting
# =============================================================================

def print_ac_decision(decision: dict, profiler: GraphProfiler) -> None:
    n_total = len(profiler.activation_nodes)
    n_evicted = len(decision["to_recompute"])

    print(f"\n{' AC DECISION ':=^90}")
    print(f"Activations evicted: {n_evicted} / {n_total}")
    print(f"Estimated peak memory: {decision['peak_bytes'] / 1024 / 1024:.1f} MB")
    print(f"Estimated recompute overhead: {decision['recompute_overhead_ms']:.2f} ms")

    if not n_evicted:
        return

    print()
    print(f"{'Activation':<25} {'Size (MB)':>10} {'Recompute (ms)':>16} {'Subgraph nodes':>16} {'Inputs':>8}")
    print('-' * 90)
    for a in sorted(decision["to_recompute"], key=lambda n: profiler.node_index[n]):
        size_mb = profiler.avg_mem_deltas.get(a.name, 0) / 1024 / 1024
        sub = decision["recompute_subgraphs"][a]
        rt = recompute_time_ms(profiler, sub)
        print(f"{a.name:<25} {size_mb:>10.3f} {rt:>16.3f} {len(sub):>16} {len(decision['recompute_inputs'][a]):>8}")
    print()

