"""
activation_checkpoint.py — Activation Checkpointing Reference Example (Phases 2 & 3)

Demonstrates activation checkpointing on a simple 2-layer network (mm → relu → mm
→ relu → sum → backward). This is a HARDCODED example showing the mechanics — for
the project, you'll generalize it using profiler data and the μ-TWO algorithm.

The three-step AC process demonstrated here:
  1. DECIDE which activations to recompute vs. checkpoint (retain in memory).
     Here: recompute 'relu', checkpoint 'relu_1'. In your project: use μ-TWO.
  2. EXTRACT the subgraph needed to recompute discarded activations using
     _extract_graph_with_inputs_outputs().
  3. INSERT the extracted recomputation subgraph into the backward pass, just
     before the first node that needs the discarded activation.

Also provides reusable utility functions:
  - replace_subsequent_uses_of(): redirect backward-pass references to recomputed nodes
  - remove_detach_nodes(): clean up autograd-inserted detach ops
  - get_name_to_node_map(): build a name→node lookup dict

Run standalone with: python activation_checkpoint.py
Requires CUDA GPU. Verifies correctness by comparing gradients with/without AC.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from graph_tracer import SEPFunction


# =============================================================================
# REFERENCE EXAMPLE: Activation Checkpointing on a Simple Function
#
# This file demonstrates the Phase 3 concept on a tiny 2-layer network.
# It is NOT the real implementation — it's hardcoded for this specific example.
# For the project, you'll generalize this using profiler data + μ-TWO algorithm.
#
# The traced graph of custom_fn looks like:
#
#   FORWARD:
#     w1_1, w2_1, x_1  (placeholders — inputs)
#     mm      = w1 @ x           (intermediate, but not an "activation" we track)
#     relu    = relu(mm)          ← ACTIVATION: created in fwd, needed in bwd
#     mm_1    = relu @ w2         (intermediate)
#     relu_1  = relu(mm_1)        ← ACTIVATION: created in fwd, needed in bwd
#     sum_1   = relu_1.sum()      (loss)
#     sep     = separator(sum_1)  ← FORWARD/BACKWARD BOUNDARY
#
#   BACKWARD:
#     sep_backward = separator_backward(...)
#     ... gradient computations that USE relu and relu_1 ...
#
#   Without AC: both relu and relu_1 are stored in GPU memory from forward to backward.
#   With AC: we discard relu after forward and recompute it from w1, x just before
#            it's needed in backward. relu_1 is kept (checkpointed).
# =============================================================================


def custom_fn(w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    A simple 2-layer linear network with ReLU, traced to demonstrate AC.
    Returns gradients of w1 and w2 (the backward pass is included in the trace).
    """
    z = torch.mm(w1, x)          # Linear layer 1: z = W1 @ X
    z = nn.functional.relu(z)    # Activation 1 (this is "relu" in the graph)
    z = torch.mm(z, w2)          # Linear layer 2: z = z @ W2
    z = nn.functional.relu(z)    # Activation 2 (this is "relu_1" in the graph)
    z = z.sum()                  # Loss function (scalar)
    z = SEPFunction.apply(z)     # Insert forward/backward boundary marker
    z.backward()                 # Backward pass — all grad ops get traced too
    return w1.grad, w2.grad      # Return the computed gradients


# =============================================================================
# GRAPH UTILITY FUNCTIONS
# These are reusable helpers for manipulating FX graphs.
# You'll use these in your real AC implementation too.
# =============================================================================

def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
) -> None:
    """
    Replace all uses of old_node with new_node, but ONLY for nodes that come
    AFTER new_node in the graph. This is important because:
    - new_node is a recomputed version of old_node inserted in the backward pass
    - We only want backward-pass nodes to use the recomputed value
    - Forward-pass nodes should keep using the original (they computed it)

    Walks the graph in REVERSE order, replacing uses until we hit new_node.
    """
    old_node_users = old_node.users
    for node in reversed(graph.nodes):
        if node == new_node:
            break  # Stop — don't replace uses that come before the recomputation
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def remove_detach_nodes(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Remove all detach nodes from the graph. Autograd inserts detach() calls
    during tracing, but they're identity ops that just add noise to the graph.
    We redirect all users of detach(x) to use x directly, then erase detach.
    """
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
    gm.graph.lint()     # Validate graph integrity
    gm.recompile()      # Regenerate forward() from modified graph
    return gm


def get_name_to_node_map(gm: fx.GraphModule) -> Dict[str, fx.Node]:
    """Build a dict mapping node name strings to node objects for easy lookup."""
    name_to_node = {}
    for node in gm.graph.nodes:
        name_to_node[node.name] = node
    return name_to_node


# =============================================================================
# ACTIVATION CHECKPOINTING (hardcoded example)
#
# This demonstrates the 3-step process for activation checkpointing:
#   1. DECIDE what to recompute vs. checkpoint (here: hardcoded)
#      In your project: use profiler data + μ-TWO algorithm
#   2. EXTRACT the subgraph needed to recompute the discarded activations
#   3. INSERT the recomputation subgraph into the backward pass
#
# The key constraint: to recompute an activation, you need its inputs to still
# be available. Those inputs must be either:
#   - Placeholder nodes (model params, input data — always available), OR
#   - Other activations that are CHECKPOINTED (not discarded)
# =============================================================================

def activation_checkpointing(gm: fx.GraphModule) -> fx.GraphModule:
    # ---- Step 1: Decide what to recompute (hardcoded for this example) ----

    # The traced graph has 2 intermediate activations: relu and relu_1
    # Decision: recompute relu, checkpoint (keep) relu_1
    # This saves memory because relu's tensor is freed after the forward pass
    # and only recomputed when the backward pass needs it.

    name_to_node = get_name_to_node_map(gm)

    # The first node in the backward pass that needs 'relu' as input.
    # This is where we'll insert the recomputation — just before this node.
    first_back_access = name_to_node["t"]

    # The activation we want to recompute instead of storing
    node_to_recompute = [name_to_node["relu"]]
    node_to_recompute_names = ["relu"]

    # The inputs needed to recompute relu: relu = relu(mm(w1, x))
    # We need w1 and x (both are placeholders, so always available).
    # NOTE: we can't use 'mm' as input because mm is ALSO an intermediate
    # that would have been freed. We must go back to checkpoint-safe inputs.
    nodes_required_to_recompute = [name_to_node["w1_1"], name_to_node["x_1"]]

    # ---- Step 2: Extract the recomputation subgraph ----
    # _extract_graph_with_inputs_outputs pulls out the minimal subgraph that
    # computes 'outputs' from 'inputs'. In this case: w1, x → mm → relu
    recompute_subgraph = _extract_graph_with_inputs_outputs(
        joint_graph=gm.graph,
        inputs=nodes_required_to_recompute,
        outputs=node_to_recompute,
    )
    print("Extracted recomputation sub-graph: ")
    recompute_subgraph.print_tabular()

    # ---- Step 3: Insert recomputation into the backward pass ----
    # We place the new nodes just BEFORE the first backward node that needs relu.
    # This way, relu is recomputed right when it's needed, not stored from forward.
    with gm.graph.inserting_before(first_back_access):
        for n in recompute_subgraph.nodes:
            # Skip placeholder and output nodes — we only want the computation nodes
            if n.op == "placeholder" or n.op == "output":
                continue

            # Copy each node from the subgraph into the main graph.
            # arg_transform maps the subgraph's placeholder references back to
            # the corresponding nodes in the main graph (by matching names).
            new_node = gm.graph.node_copy(
                n, arg_transform=lambda arg: name_to_node[arg.name]
            )

            if n.name in node_to_recompute_names:
                old_node = name_to_node[n.name]
                # Replace all SUBSEQUENT uses of the old relu with the new
                # recomputed relu. "Subsequent" = only backward pass nodes,
                # since new_node is placed in the backward region.
                replace_subsequent_uses_of(
                    gm.graph, old_node=old_node, new_node=new_node
                )

            # Update the name→node map so later iterations can find this node
            name_to_node[n.name] = new_node

    gm.graph.lint()     # Validate the modified graph
    gm.recompile()      # Regenerate forward() from the modified graph
    return gm


# =============================================================================
# MAIN: End-to-end demonstration of activation checkpointing
# =============================================================================

if __name__ == "__main__":
    # Create test tensors: two weight matrices (require gradients) and input data
    w1 = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    w2 = torch.randn(2048, 512, device="cuda", requires_grad=True)
    x = torch.randn(1024, 2048, device="cuda")

    # Trace custom_fn into an FX graph — this captures forward + backward
    # make_fx runs the function with proxy tensors and records every op
    graph_module = make_fx(custom_fn)(w1, w2, x)
    graph_module = remove_detach_nodes(graph_module)  # Clean up autograd noise
    print("Original graph of custom fn (fwd+bwd): ")
    graph_module.graph.print_tabular()

    # Run the ORIGINAL graph (no AC) to get reference gradients
    # no_grad because the graph already contains explicit backward ops
    with torch.no_grad():
        old_grads = graph_module(w1, w2, x)

    # Apply activation checkpointing — this modifies the graph to recompute
    # relu instead of storing it. A new node 'relu_2' (the recomputation)
    # appears in the backward section of the graph.
    new_graph_module = activation_checkpointing(graph_module)
    print("Modified graph of custom fn (fwd+bwd+activation_checkpointing): ")
    new_graph_module.graph.print_tabular()

    # Run the MODIFIED graph (with AC) to get new gradients
    with torch.no_grad():
        new_grads = new_graph_module(w1, w2, x)

    # VERIFICATION: gradients with AC must match gradients without AC
    # If activation checkpointing is correct, the recomputed activations
    # produce identical gradients. Any mismatch indicates a bug.
    print("Result verification")
    for old_grad, new_grad in zip(old_grads, new_grads):
        print(torch.allclose(old_grad, new_grad))  # Should print True for both
