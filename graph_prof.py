"""
graph_prof.py — Computational Graph Profiler (Phase 1)

Extends torch.fx.Interpreter to execute a traced training graph node-by-node,
collecting per-operator timing, memory usage, and tensor classification.

This is the main file you need to implement for Phase 1. The GraphProfiler class
has placeholder methods that need to be filled in:

  __init__():        Static analysis — identify forward/backward boundary (sep nodes),
                     classify nodes as PARAM/ACT/GRAD/OTHER, track activation lifetimes.
  run_node():        Per-node profiling — measure execution time (CUDA events) and
                     memory delta (torch.cuda.memory_allocated) for each operation.
  reset_stats():     Clear accumulated measurements after warm-up iterations.
  aggregate_stats(): Average measurements over the profiling iterations.
  print_stats():     Output profiling results (per-op table, activation lifetimes,
                     peak memory breakdown).

The profiler is invoked by graph_transformation() in starter_code.py / benchmarks.py.
"""

from enum import Enum
from typing import Dict, Any
from datetime import datetime
import json
import os
import torch
import torch.fx as fx


class OP(str, Enum):
    """
    Maps to torch.fx node operation types (node.op).

    In an FX graph, every node has an 'op' field indicating what kind of operation it is:
    - PLACEHOLDER:    An input to the graph. These are the function arguments.
                      In our case: parameters, buffers, optimizer states, input batch, etc.
    - CALL_FUNCTION:  A call to a Python function or torch operator.
                      e.g., torch.ops.aten.mm.default, torch.ops.aten.relu.default
                      This is where the actual computation happens.
    - CALL_MODULE:    A call to a submodule (e.g., self.layer1(x)). Rare after tracing
                      since make_fx decomposes modules into individual ops.
    - CALL_METHOD:    A call to a method on a tensor (e.g., x.sum()).
    - GET_ATTR:       Accessing an attribute of the module (e.g., self.weight).
    - OUTPUT:         The return value of the graph. Always the last node.
    """
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    Classification for tensors flowing through the graph.
    Every tensor (node output) in the graph falls into one of these categories:

    - PARAM: Model parameters (weights and biases). These are PLACEHOLDER nodes
             that appear in the optimizer's parameter list. They persist across
             iterations and are updated by the optimizer.

    - ACT:   Activations / intermediate feature maps. Created during the forward
             pass and consumed during the backward pass. These are the PRIMARY
             target for memory optimization via activation checkpointing, since
             they account for ~70-85% of peak memory.

    - GRAD:  Gradients of the loss w.r.t. parameters. Created during the backward
             pass and consumed by the optimizer. Can be identified from the
             _fused_adam node's argument at position 1.

    - OTHER: Everything else — optimizer states (exp_avg, exp_avg_sq, step),
             input batch data, loss scalars, intermediate backward computations
             that aren't activations, etc.
    """
    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3


# =============================================================================
# GraphProfiler
#
# Extends fx.Interpreter to execute the traced graph node-by-node.
#
# fx.Interpreter works by:
#   1. Setting up an "environment" dict mapping each node → its runtime value
#   2. For each node in topological order, calling run_node(node)
#   3. run_node() looks up the node's inputs in the environment, executes the
#      operation, stores the result in the environment, and returns it
#
# By overriding run_node(), you can inject profiling logic (timing, memory
# measurement) around each operation's execution.
#
# Key fx.Node attributes you'll use:
#   node.op:              One of the OP enum values (placeholder, call_function, etc.)
#   node.target:          The actual function/op being called (e.g., torch.ops.aten.mm.default)
#   node.name:            A unique string name (e.g., "mm", "relu", "sep")
#   node.args:            The arguments passed to this op (other nodes or constants)
#   node.all_input_nodes: List of all Node objects that are inputs to this node
#   node.users:           Dict of {node: None} for all nodes that use this node's output
# =============================================================================


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # used run_node
        self.cumulative_mem = 0 # running sum of memory deltas (increase then decrease)
        self.peak_mems = [] # list of _run_peak values
        self.peak_nodes = [] # list of _run_peak_node values

        # used in run_node and aggregate_stats
        self.node_times = {}
        self.node_mem_deltas = {}

        # used in aggregate_stats
        self.avg_peak_mem = 0 # avg of peak_mems
        self.peak_node = None # node at peak
        self.avg_times = {}
        self.avg_mem_deltas = {}

        # used in run
        self.actual_peak_mem = 0 # actual GPU peak. only taken at last run

        # node position map
        self.node_index = {}
        for i, node in enumerate(self.module.graph.nodes):
            self.node_index[node] = i

        # identify sep nodes to determine forward/backward boundary
        self.sep_node = None
        self.sep_backward_node = None
        for node in self.module.graph.nodes:
            if node.op == OP.CALL_FUNCTION:
                if node.target == torch.ops.separator.sep.default:
                    self.sep_node = node
                elif node.target == torch.ops.separator.sep_backward.default:
                    self.sep_backward_node = node

        # identify parameter and gradient nodes
        # sets for constant time membership checking
        self.param_nodes = set()
        self.grad_nodes = set()
        for node in self.module.graph.nodes:
            if node.op == OP.CALL_FUNCTION and node.target == torch.ops.aten._fused_adam.default:
                self.param_nodes.update(node.args[0])
                self.grad_nodes.update(node.args[1])

        # identify activations, track their lifetimes (from last fwd to first bwd use)
        self.activation_nodes = []
        self.last_forward_access = {}
        self.first_backward_access = {}
        self.sep_idx = self.node_index[self.sep_node]
        self.sep_back_idx = self.node_index[self.sep_backward_node]

        for node in self.module.graph.nodes:
            idx = self.node_index[node]

            # only observing forward-pass computation nodes
            if idx >= self.sep_idx:
                continue
            if node.op == OP.PLACEHOLDER or node.op == OP.OUTPUT:
                continue

            last_fwd_user = None
            first_bwd_user = None

            for user in node.users:
                user_idx = self.node_index[user]

                # last user in forward region
                if user_idx < self.sep_idx:
                    if last_fwd_user is None or user_idx > self.node_index[last_fwd_user]:
                        last_fwd_user = user

                # first user in backward region
                if user_idx > self.sep_back_idx:
                    if first_bwd_user is None or user_idx < self.node_index[first_bwd_user]:
                        first_bwd_user = user

            # needs one backward user to be considered an activation
            if first_bwd_user is not None:
                self.activation_nodes.append(node)
                self.last_forward_access[node] = last_fwd_user
                self.first_backward_access[node] = first_bwd_user


    def run(self, *args, initial_env: Dict[fx.Node, Any] | None = None, enable_io_processing: bool = True):
        """
        Executes the entire graph once. Called multiple times:
        - warm_up_iters times (to warm up GPU caches/kernels)
        - profile_iters times (actual measurement)

        Each call iterates through all nodes in topological order,
        calling run_node() for each one.
        """
        self.cumulative_mem = 0
        self._run_peak = 0 # highest cumulative_mem for one run
        self._run_peak_node = None # associated node

        torch.cuda.reset_peak_memory_stats()
        result = super().run(*args, initial_env=initial_env, enable_io_processing=enable_io_processing)

        self.peak_mems.append(self._run_peak)
        self.peak_nodes.append(self._run_peak_node)
        self.actual_peak_mem = torch.cuda.max_memory_allocated()

        return result


    def run_node(self, n: fx.Node) -> Any:

        # setup
        orig_mem = torch.cuda.memory_allocated()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = super().run_node(n) # execute actual op
        end_event.record()
        torch.cuda.synchronize() # wait for GPU to finish

        # measurements
        elapsed_ms = start_event.elapsed_time(end_event)
        mem_after = torch.cuda.memory_allocated()
        mem_delta = mem_after - orig_mem

        # recording
        self.cumulative_mem += mem_delta
        if self.cumulative_mem > self._run_peak:
            self._run_peak = self.cumulative_mem
            self._run_peak_node = n

        if n.name not in self.node_times:
            self.node_times[n.name] = []
            self.node_mem_deltas[n.name] = []

        self.node_times[n.name].append(elapsed_ms)
        self.node_mem_deltas[n.name].append(mem_delta)

        return result


    def aggregate_stats(self) -> None:
        for name in self.node_times:
            self.avg_times[name] = sum(self.node_times[name]) / len(self.node_times[name])
        for name in self.node_mem_deltas:
            self.avg_mem_deltas[name] = sum(self.node_mem_deltas[name]) / len(self.node_mem_deltas[name])

        self.avg_peak_mem = sum(self.peak_mems) / len(self.peak_mems) if self.peak_mems else 0
        self.peak_node = self.peak_nodes[-1] if self.peak_nodes else None



    def _short_target(self, node):
        if node.op == OP.PLACEHOLDER:
            return "input"
        if node.op == OP.OUTPUT:
            return "output"
        target = str(node.target)
        # "torch.ops.aten.mm.default" → "aten.mm"
        if "aten." in target:
            parts = target.split(".")
            aten_idx = parts.index("aten")
            return f"aten.{parts[aten_idx + 1]}"
        if "separator." in target:
            parts = target.split(".")
            sep_idx = parts.index("separator")
            return f"sep.{parts[sep_idx + 1]}"
        return target


    def print_stats(self, save_prefix: str = "results", model_name: str = None, batch_size: int = None) -> None:

        # node table
        print(f"\n{' NODE PROFILING ':=^90}")
        print(f"{'Name':<30} {'Op':<22} {'Type':<8} {'Time (ms)':>12} {'Mem Delta (kb)':>18}")
        print('-' * 90)

        for node in self.module.graph.nodes:
            if node.op == OP.OUTPUT:
                continue

            time = self.avg_times.get(node.name, 0)
            mem_del = self.avg_mem_deltas.get(node.name, 0)

            if node in self.param_nodes:
                node_type = 'PARAM'
            elif node in self.grad_nodes:
                node_type = 'GRAD'
            elif node in self.activation_nodes:
                node_type = 'ACT'
            else:
                node_type = 'OTHER'

            print(f"{node.name:<30} {self._short_target(node):<22} {node_type:<8} {time:>12.3f} {mem_del/1024:>18.3f}")

        # forward / backward timing
        fwd_time = 0
        bwd_time = 0

        for node in self.module.graph.nodes:
            time = self.avg_times.get(node.name, 0)
            idx = self.node_index[node]
            if idx < self.sep_idx:
                fwd_time += time
            elif idx > self.sep_back_idx:
                bwd_time += time

        print(f"\n{' SUMMARY ':=^90}")
        print(f"Forward: {fwd_time:.3f} ms")
        print(f"Backward: {bwd_time:.3f} ms")
        print(f"Total: {fwd_time + bwd_time:.3f} ms")
        print(f"\nPeak graph memory: {self.avg_peak_mem / 1024 / 1024:.3f} MB (at node: {self.peak_node.name if self.peak_node else 'N/A'})")
        print(f"Peak GPU memory: {self.actual_peak_mem / 1024 / 1024:.3f} MB")

        # memory by node type
        param_mem = 0
        act_mem = 0
        grad_mem = 0
        other_mem = 0

        for node in self.module.graph.nodes:
            mem_del = self.avg_mem_deltas.get(node.name, 0)
            if mem_del <= 0:
                continue
            if node in self.param_nodes:
                param_mem += mem_del
            elif node in self.grad_nodes:
                grad_mem += mem_del
            elif node in self.activation_nodes:
                act_mem += mem_del
            else:
                other_mem += mem_del

        total_mem = param_mem + act_mem + grad_mem + other_mem

        print(f"\n{' MEMORY BREAKDOWN ':=^90}")
        if total_mem > 0:
            print(f"Parameters: {param_mem/1024/1024:.3f} MB ({param_mem/total_mem*100:.1f}%)")
            print(f"Activations: {act_mem/1024/1024:.3f} MB ({act_mem/total_mem*100:.1f}%)")
            print(f"Gradients: {grad_mem/1024/1024:.3f} MB ({grad_mem/total_mem*100:.1f}%)")
            print(f"Other: {other_mem/1024/1024:.3f} MB ({other_mem/total_mem*100:.1f}%)")
            print(f"Total: {total_mem/1024/1024:.3f} MB")
        else:
            print("No memory allocations recorded.")

        # activation lifetimes
        print(f"\n{' ACTIVATION LIFETIMES ':=^90}")
        print(f"{'Activation':<25} {'Size (MB)':>12} {'Last Fwd Use':<22} {'First Bwd Use':<22} {'Idle Gap':>9}")
        print('-' * 90)

        for node in self.activation_nodes:
            last_fwd = self.last_forward_access.get(node)
            first_bwd = self.first_backward_access.get(node)
            size = self.avg_mem_deltas.get(node.name, 0) / 1024 / 1024
            gap = 0
            if last_fwd and first_bwd:
                gap = self.node_index[first_bwd] - self.node_index[last_fwd]

            print(f"{node.name:<25} {size:>12.3f} {last_fwd.name if last_fwd else 'N/A':<22} {first_bwd.name if first_bwd else 'N/A':<22} {gap:>9}")

        print()

        self.save_stats(prefix=save_prefix, model_name=model_name, batch_size=batch_size)


    def save_stats(self, prefix: str = "results", model_name: str = None, batch_size: int = None) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{prefix}_bs{batch_size}_{timestamp}.json"

        # compute summary values
        fwd_time = 0
        bwd_time = 0
        for node in self.module.graph.nodes:
            t = self.avg_times.get(node.name, 0)
            idx = self.node_index[node]
            if idx < self.sep_idx:
                fwd_time += t
            elif idx > self.sep_back_idx:
                bwd_time += t

        param_mem = 0
        act_mem = 0
        grad_mem = 0
        other_mem = 0
        for node in self.module.graph.nodes:
            m = self.avg_mem_deltas.get(node.name, 0)
            if m <= 0:
                continue
            if node in self.param_nodes:
                param_mem += m
            elif node in self.grad_nodes:
                grad_mem += m
            elif node in self.activation_nodes:
                act_mem += m
            else:
                other_mem += m

        # per-node data
        nodes_data = []
        for node in self.module.graph.nodes:
            if node.op == OP.OUTPUT:
                continue
            if node in self.param_nodes:
                ntype = "PARAM"
            elif node in self.grad_nodes:
                ntype = "GRAD"
            elif node in self.activation_nodes:
                ntype = "ACT"
            else:
                ntype = "OTHER"
            nodes_data.append({
                "name": node.name,
                "op": self._short_target(node),
                "type": ntype,
                "time_ms": self.avg_times.get(node.name, 0),
                "mem_delta_bytes": self.avg_mem_deltas.get(node.name, 0),
            })

        # activation lifetime data
        activations_data = []
        for node in self.activation_nodes:
            last_fwd = self.last_forward_access.get(node)
            first_bwd = self.first_backward_access.get(node)
            gap = 0
            if last_fwd and first_bwd:
                gap = self.node_index[first_bwd] - self.node_index[last_fwd]
            activations_data.append({
                "name": node.name,
                "size_bytes": self.avg_mem_deltas.get(node.name, 0),
                "last_forward_use": last_fwd.name if last_fwd else None,
                "first_backward_use": first_bwd.name if first_bwd else None,
                "idle_gap": gap,
            })

        data = {
            "timestamp": timestamp,
            "summary": {
                "model_name": model_name,
                "batch_size": batch_size,
                "forward_time_ms": fwd_time,
                "backward_time_ms": bwd_time,
                "total_time_ms": fwd_time + bwd_time,
                "peak_graph_memory_bytes": self.avg_peak_mem,
                "peak_graph_memory_node": self.peak_node.name if self.peak_node else None,
                "peak_gpu_memory_bytes": self.actual_peak_mem,
            },
            "memory_breakdown": {
                "param_bytes": param_mem,
                "activation_bytes": act_mem,
                "gradient_bytes": grad_mem,
                "other_bytes": other_mem,
            },
            "nodes": nodes_data,
            "activations": activations_data,
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {filename}")


    def reset_stats(self) -> None:
        self.peak_mems.clear()
        self.peak_nodes.clear()
        self.cumulative_mem = 0

        self.node_times.clear()
        self.node_mem_deltas.clear()






