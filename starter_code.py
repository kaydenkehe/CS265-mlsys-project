"""
starter_code.py — Entry Point and Example Pipeline

Demonstrates the end-to-end flow on a simple DummyModel (stacked Linear + ReLU):

  1. Create a model and Adam optimizer
  2. Pre-initialize optimizer states (so they exist during tracing)
  3. compile(train_step, graph_transformation) wraps the training step:
     - First call: traces train_step into an FX graph → runs graph_transformation
       (which profiles the graph using GraphProfiler) → caches the compiled graph
     - Subsequent calls: execute the cached graph directly
  4. graph_transformation() is where profiling happens, and later where you'll
     also call activation_checkpointing() to rewrite the graph.

Run with: python starter_code.py
Requires CUDA GPU.
"""

import logging
import os
from functools import wraps
from typing import Any

import torch
import torch.fx as fx
import torch.multiprocessing as mp
import torch.nn as nn

from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile


# =============================================================================
# DUMMY MODEL
# A simple feedforward network for testing the tracing/profiling pipeline.
# In the real project, you'll use ResNet-152 and BERT (see benchmarks.py).
# =============================================================================

class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            # Each "layer" is a Linear transform followed by ReLU activation
            # Linear(dim, dim) means input and output have the same dimension
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        # nn.Sequential chains the layers: input → Linear → ReLU → Linear → ReLU → ...
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


# =============================================================================
# TRAIN STEP
# This is the function that gets TRACED by graph_tracer.compile().
# After tracing, the entire body of this function (forward + backward +
# optimizer) becomes a single FX graph.
#
# The SEPFunction.apply(loss) call is critical: it inserts marker nodes
# (sep / sep_backward) into the graph so you can locate the forward/backward
# boundary during profiling.
# =============================================================================

def train_step(
    model: torch.nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor
):
    # Forward pass: run input through the model, compute scalar loss
    loss = model(batch).sum()

    # Insert the separator marker — this is an identity op that marks:
    #   sep         → end of forward pass
    #   sep_backward → start of backward pass (inserted automatically by autograd)
    loss = SEPFunction.apply(loss)

    # Backward pass: computes gradients for all parameters
    loss.backward()

    # Optimizer step: updates parameters using gradients (e.g., Adam update rule)
    optim.step()

    # Zero out gradients to prepare for next iteration
    optim.zero_grad()


# =============================================================================
# GRAPH TRANSFORMATION
# This callback is invoked by compile() on the FIRST call only.
# It receives the fully traced FX graph (containing forward + backward +
# optimizer ops) and can analyze or modify it before it's used for execution.
#
# The workflow:
#   1. Print the graph for inspection
#   2. Create a GraphProfiler to run the graph node-by-node
#   3. Run warm-up iterations (GPU kernel caching, memory allocation settling)
#   4. Reset stats (discard warm-up measurements)
#   5. Run measurement iterations (collect actual timing/memory data)
#   6. Aggregate and print the statistics
#   7. Return the (possibly modified) graph
#
# In later phases, you'll also call activation_checkpointing() here to
# rewrite the graph before returning it.
# =============================================================================

def graph_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
    # Print the graph structure (shows all nodes and their connections)
    print(gm.graph)

    # Create the profiler — this runs static analysis in __init__
    graph_profiler = GraphProfiler(gm)

    warm_up_iters, profile_iters = 2, 3

    # torch.no_grad(): we already have backward ops in the graph, so we
    # don't need autograd to build another backward graph on top
    with torch.no_grad():
        # Warm-up: run a few iterations so GPU kernels are compiled/cached
        # and memory allocator patterns stabilize
        for _ in range(warm_up_iters):
            graph_profiler.run(*args)

        # Discard warm-up measurements — they include one-time costs
        graph_profiler.reset_stats()

        # Actual measurement: collect timing and memory stats
        for _ in range(profile_iters):
            graph_profiler.run(*args)

    # Average the measurements over profile_iters runs
    graph_profiler.aggregate_stats()

    # Print the profiling results
    graph_profiler.print_stats(
        save_prefix="results_DummyModel",
        model_name="DummyModel",
        batch_size=1000,
    )

    return gm


# =============================================================================
# EXPERIMENT: End-to-end setup and execution
#
# Flow:
#   1. Create model and optimizer
#   2. Pre-initialize optimizer states (so they exist during tracing)
#   3. compile() wraps train_step:
#      - First call: traces → graph_transformation → caches compiled graph
#      - Subsequent calls: runs cached graph directly
# =============================================================================

def experiment():
    logging.getLogger().setLevel(logging.DEBUG)
    torch.manual_seed(20)
    batch_size = 1000
    layers = 10
    dim = 100
    num_iters = 5

    device_str = 'cuda:0'
    model = DummyModel(dim=dim, layers=layers).to(device_str)
    batch = torch.randn(batch_size, dim).to(device_str)

    # Adam optimizer with:
    #   foreach=True: uses fused foreach operations (faster, but traced as grouped ops)
    #   capturable=True: makes the optimizer compatible with CUDA graph capture and tracing
    #     (without this, optimizer internals use Python-side logic that can't be traced)
    optim = torch.optim.Adam(
        model.parameters(), lr=0.01,
        foreach=True,  # fused=True,  # fused would use _fused_adam kernel
        capturable=True
    )

    # PRE-INITIALIZE OPTIMIZER STATES
    # Adam lazily creates its states (exp_avg, exp_avg_sq, step) on the first
    # call to .step(). We need these to exist BEFORE tracing so they become
    # placeholder nodes in the graph. Without this, the tracer wouldn't see
    # optimizer state tensors.
    for param in model.parameters():
        if param.requires_grad:
            # Create dummy gradients so optimizer.step() has something to work with
            param.grad = torch.rand_like(param, device=device_str)

    # Run one optimizer step to initialize internal Adam states
    optim.step()
    optim.zero_grad()  # Clear the dummy gradients

    # compile() returns a wrapper that:
    # - On first call: traces train_step → applies graph_transformation → caches result
    # - On later calls: directly executes the compiled graph
    compiled_fn = compile(train_step, graph_transformation)

    # This first call triggers tracing + graph_transformation (profiling)
    compiled_fn(model, optim, batch)


if __name__ == "__main__":
    experiment()

