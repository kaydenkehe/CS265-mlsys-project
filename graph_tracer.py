"""
graph_tracer.py — Compilation and Graph Tracing Engine

Traces an entire PyTorch training step (forward + backward + optimizer) into a
single torch.fx graph. This is the foundation that all other components build on:
the profiler analyzes this graph, and activation checkpointing rewrites it.

Key exports:
  - SEPFunction:  Autograd function that inserts separator markers (sep /
                  sep_backward) into the graph to identify the forward/backward
                  boundary.
  - compile():    Takes a train_step function and a graph_transformation callback.
                  On first call, traces train_step into an FX graph, passes it to
                  graph_transformation (for profiling/optimization), and caches the
                  result. Subsequent calls execute the cached compiled graph.

How tracing works:
  1. All model parameters, buffers, and optimizer states are "lifted" out of the
     module and passed as explicit function arguments (making the function stateless).
  2. make_fx() runs the function with FakeTensors (shape-only, no real data) and
     records every operation into an fx.Graph.
  3. The decomposition table (from utils.py) breaks fused ops into individual
     operations so each one is a separate, profileable graph node.
  4. Cleanup removes noise nodes (detach, tag_grad) from the traced graph.
"""

from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Union
from utils import SPMD_DECOMP_TABLE

import torch

# This import has a side effect: it registers distributed collective ops
# (like all_reduce) so they are available as torch operators during tracing.
import torch.distributed._functional_collectives
import torch.nn as nn
import torch.optim as optim
import torch.utils._pytree as pytree  # pytree: utilities for flattening/unflattening nested Python structures (dicts, lists, tuples) of tensors
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode  # FakeTensorMode: allows tracing with "fake" tensors that track shapes/dtypes but hold no data — no GPU memory used during tracing
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.tensor import DTensor  # DTensor: distributed tensor abstraction for SPMD (Single Program Multiple Data) parallelism
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.fx.experimental.proxy_tensor import make_fx  # make_fx: the main tracing function — runs a Python function with proxy tensors and records all ops into an fx.Graph
from torch.fx.graph import CodeGen, _PyTreeCodeGen, _PyTreeInfo
from torch.nn.utils import stateless  # stateless: allows running a module with externally-provided parameters instead of its own .parameters()
from torch.utils.hooks import RemovableHandle


# =============================================================================
# SEPARATOR OPERATORS
# These are identity functions registered as custom torch operators.
# They do NOTHING computationally (just return their input unchanged).
# Their purpose: when the training loop is traced into an FX graph, these
# ops appear as nodes that MARK THE BOUNDARY between forward and backward.
# This lets the profiler know which nodes belong to forward vs backward.
# =============================================================================

def sep(x: torch.Tensor) -> torch.Tensor:
    """Forward separator — identity function that marks END of forward pass."""
    return x


def sep_backward(grad: torch.Tensor) -> torch.Tensor:
    """Backward separator — identity function that marks START of backward pass."""
    return grad


# Register these as proper torch library ops so they survive tracing.
# "DEF" means we're defining new ops in the "separator" namespace.
# After this, they are callable as torch.ops.separator.sep() and
# torch.ops.separator.sep_backward().
separator_lib = torch.library.Library("separator", "DEF")
separator_lib.define("sep(Tensor x) -> Tensor")
separator_lib.impl("sep", sep, "CompositeExplicitAutograd")  # CompositeExplicitAutograd: works with both eager and compiled modes
separator_lib.define("sep_backward(Tensor x) -> Tensor")
separator_lib.impl("sep_backward", sep_backward, "CompositeExplicitAutograd")


# =============================================================================
# SHARDING PROPAGATION RULES (for distributed training with DTensor)
# When using SPMD parallelism, each op needs a rule that says how its output
# should be sharded given how its inputs are sharded. Since sep/sep_backward
# are identity ops, the output sharding is just the same as the input sharding.
# =============================================================================

def _identity_prop_rule(op_schema: OpSchema) -> OutputSharding:
    """Propagation rule: output has same sharding as input (identity)."""
    (x,) = op_schema.args_schema
    assert isinstance(x, DTensorSpec), f"expecting DTensorSpec but got {x}"
    return OutputSharding(output_spec=DTensorSpec(x.mesh, x.placements))

def _prop_sepm(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)

def _prop_sepm_backward(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)

# Register the sharding rules so DTensor knows how to handle these ops
DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(torch.ops.separator.sep.default, _prop_sepm)
DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(torch.ops.separator.sep_backward.default, _prop_sepm_backward)



# =============================================================================
# SEPFunction: The autograd glue that inserts separator markers into the graph.
#
# Usage in train_step:
#   loss = model(batch).sum()
#   loss = SEPFunction.apply(loss)   # <-- inserts sep (fwd) and sep_backward (bwd)
#   loss.backward()
#
# When make_fx traces this, the graph will contain:
#   ... forward ops ... → sep → loss ops → sep_backward → ... backward ops ...
# This is how you find the forward/backward boundary in the profiler.
# =============================================================================

class SEPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        # Calls the custom sep op — appears in the traced graph as:
        # %sep = call_function[target=torch.ops.separator.sep.default]
        return torch.ops.separator.sep(x)

    @staticmethod
    def backward(ctx: Any, grad_x: torch.Tensor) -> torch.Tensor:
        # Calls the custom sep_backward op — appears in the traced graph as:
        # %sep_backward = call_function[target=torch.ops.separator.sep_backward.default]
        return torch.ops.separator.sep_backward(grad_x)


# =============================================================================
# GRADIENT TAGGING
# Another identity op ("dummy.tag_grad") used to mark gradient tensors during
# tracing. When hooks are attached to parameters, their gradients flow through
# this op, making gradients identifiable in the graph. These dummy nodes are
# cleaned up after tracing (see _compile, lines ~267-271).
# =============================================================================
_spmd_lib_def = torch.library.Library("dummy", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

_spmd_lib_impl = torch.library.Library("dummy", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")


# =============================================================================
# PYTREE INPUT FLATTENING
#
# When make_fx traces a function, it expects nested structures (dicts of
# params, dicts of optimizer states, etc.) as inputs. The traced graph uses
# _PyTreeCodeGen to automatically flatten/unflatten these structures.
#
# After tracing, we want the graph to accept a FLAT list of tensors as input
# (i.e., the caller is responsible for flattening). This is simpler and
# matches how we pass flat_state + args to the compiled graph.
# =============================================================================

class _PyTreeCodeGenOutputsOnly(_PyTreeCodeGen):
    """Modified code generator that skips input flattening (caller does it)
    but still handles output unflattening."""
    # pyre-ignore[3]
    def process_inputs(self, *args: Any) -> Any:
        # Don't flatten inputs — they're already flat from the caller
        return args

    # pyre-ignore[2, 3]
    def gen_fn_def(self, free_vars, maybe_return_annotation):
        # Use the base CodeGen's function definition (no pytree input spec)
        return CodeGen.gen_fn_def(self, free_vars, maybe_return_annotation)


def _to_caller_flattened_graph_module(gm: fx.GraphModule) -> fx.GraphModule:
    """Move the responsibility of flattening the input arguments from the
    graph module to the caller.

    Before: gm({"layer1.weight": tensor, "layer1.bias": tensor, ...})
    After:  gm(tensor, tensor, ...)  ← caller flattens the dict first

    This is needed because at runtime we pass flat_state (a list of all
    parameter/buffer/optimizer-state tensors) directly to the graph.
    """
    # pyre-ignore[16]
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(
        pytree_info=_PyTreeInfo(
            # pyre-ignore[6]
            orig_args=None,  # type: ignore[arg-type]  # No input spec needed
            # pyre-ignore[6]
            in_spec=None,  # type: ignore[arg-type]    # No input spec needed
            # pyre-ignore[16]
            out_spec=gm._graph._codegen.pytree_info.out_spec,  # Keep the output spec for unflattening
        )
    )
    gm.graph.eliminate_dead_code()  # Remove any nodes that are no longer used
    gm.recompile()  # Regenerate the forward() method from the modified graph
    return gm


# =============================================================================
# CONTEXT MANAGERS FOR TRACING
# These set up the environment so make_fx can trace through the full
# training loop, including parameter access, gradient hooks, and optimizer.
# =============================================================================

@contextmanager
def gradients_tagging(params: Dict[str, nn.Parameter]):
    """
    Registers backward hooks on all parameters that tag their gradients
    with the dummy.tag_grad op. During tracing, this makes gradient tensors
    identifiable in the graph (they flow through tag_grad nodes).

    The tag_grad nodes are removed after tracing in _compile() — they're
    only needed to mark which tensors are gradients vs other intermediates.
    """
    tagging_hooks: List[RemovableHandle] = []
    try:
        for p in params.values():
            # register_hook: called every time a gradient is computed for this param
            # The hook calls tag_grad (an identity op) so the gradient gets a
            # recognizable node in the traced graph
            h = p.register_hook(lambda grad: torch.ops.dummy.tag_grad(grad))
            tagging_hooks.append(h)
        yield
    finally:
        # Clean up hooks after tracing is done
        for h in tagging_hooks:
            h.remove()


@contextmanager
def _rematerialize_optimizer(
    opt: optim.Optimizer,
    named_states: Dict[str, Any],
    params: Dict[str, nn.Parameter],
):
    """
    Temporarily replaces the optimizer's internal state and parameter references
    with proxy tensors so that optimizer.step() gets traced into the graph.

    Problem: PyTorch optimizers store state keyed by Parameter objects.
    During tracing, we use proxy tensors (not real Parameters), so the
    optimizer can't find its states. This context manager swaps in the
    proxy-tensor-keyed states temporarily.

    After tracing, it restores the original state so the optimizer is unmodified.
    """
    assert opt is not None

    # Save original state, then replace with proxy-tensor-keyed states
    orig_states = copy(opt.state)
    for n in named_states:
        # Key is the proxy Parameter, value is the proxy optimizer state (exp_avg, exp_avg_sq, etc.)
        opt.state[params[n]] = named_states[n]  # type: ignore[index]

    # FIXME: support multiple parameter groups
    # Replace the parameter list with proxy parameters
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    param_group["params"] = params.values()

    try:
        yield
    finally:
        # Restore original parameters and state after tracing
        param_group["params"] = orig_params
        opt.state = orig_states


@contextmanager
def _enable_compile():
    """
    Monkey-patches torch._utils.is_compiling() to return True during tracing.

    Why: PyTorch's optimizer checks is_compiling() to decide its code path.
    When compiling, the optimizer uses a "capturable" path that records all
    operations as graph nodes (instead of doing in-place updates that can't
    be traced). Without this patch, optimizer.step() would not appear in
    the traced graph.
    """
    def f_true():
        return True

    orig_is_compiling_code = torch._utils.is_compiling.__code__
    torch._utils.is_compiling.__code__ = f_true.__code__
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code


# =============================================================================
# COMPILATION: The core of the tracing system
# =============================================================================

@dataclass
class _CompiledResult:
    """Holds everything needed to run the compiled training step."""
    gm: fx.GraphModule       # The traced FX graph (contains ALL ops: forward + backward + optimizer)
    mod: nn.Module            # The original model (for accessing parameters at runtime)
    opt: Optional[torch.optim.Optimizer]  # The original optimizer
    flat_state: List[torch.Tensor]        # Flattened list of [params, buffers, optimizer_states] — passed as graph inputs at runtime


def _compile(func: Callable, *args: Any, **kwargs: Any):
    """
    Traces a training step function into a single FX graph.

    The key challenge: PyTorch's autograd normally builds the backward graph
    dynamically during forward execution. Here, we use make_fx to capture
    BOTH forward and backward (and optimizer) into a single static graph.

    Steps:
      1. Extract the model and optimizer from the function arguments
      2. "Lift" all stateful tensors (params, buffers, optimizer states) into
         explicit function arguments — this makes the function stateless so
         make_fx can trace through everything
      3. Trace with FakeTensors (no actual computation, just shape tracking)
      4. Clean up the resulting graph (remove detach/tag_grad noise)
    """

    # ---- Step 1: Find the nn.Module and Optimizer from the args ----
    # pytree.tree_flatten recursively unwraps nested structures to find all leaf values
    mod, opt = None, None
    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, nn.Module):
            assert mod is None, "Only support single nn.Module for now"
            mod = arg
        if isinstance(arg, optim.Optimizer):
            assert opt is None, "Only support single Optimizer for now"
            opt = arg
    assert mod is not None, "Couldn't find nn.Module instances from the arguments."

    # ---- Step 2: Lift state into function arguments (make it "stateless") ----
    # Why stateless? make_fx traces by running the function with proxy tensors.
    # If parameters live inside the module, make_fx can't see operations on them.
    # By lifting them into function args, every read/write becomes a visible graph op.

    params = dict(mod.named_parameters(remove_duplicate=False))   # e.g. {"mod.0.weight": tensor, "mod.0.bias": tensor, ...}
    buffers = dict(mod.named_buffers(remove_duplicate=False))     # e.g. {"mod.0.running_mean": tensor, ...} (for BatchNorm etc.)

    # Extract optimizer states (exp_avg, exp_avg_sq, step for Adam) keyed by param name
    named_states: Dict[str, nn.Parameter] = {}
    for n, p in params.items():
        if p in opt.state:
            # Adam stores: {param_tensor: {"step": ..., "exp_avg": ..., "exp_avg_sq": ...}}
            # We re-key by param name so we can match them with proxy tensors during tracing
            named_states[n] = opt.state[p]

    def stateless_func(
        func: Callable,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, nn.Parameter],
        args: Any,
        kwargs: Any,
    ):
        """
        Wrapper that runs the original train_step with externalized state.

        _reparametrize_module: temporarily replaces module's params/buffers
            with the ones passed as arguments (proxy tensors during tracing)
        _rematerialize_optimizer: temporarily replaces optimizer's state
            with proxy-tensor versions
        gradients_tagging: adds hooks to tag gradients for identification
        """
        with stateless._reparametrize_module(
            mod, {**params, **buffers}
        ), _rematerialize_optimizer(
            opt, named_states, params
        ) if opt else nullcontext():
            with gradients_tagging(params):
                ret = func(*args, **kwargs)  # Runs: forward → SEP → backward → optimizer.step()

            # Return the original output PLUS the updated parameters and optimizer states.
            # This ensures the graph captures how params/states are modified by the optimizer.
            return ret, list(mod.parameters()), list(named_states.values())

    # ---- Step 3: Trace with FakeTensors ----
    # FakeTensors track shapes and dtypes without allocating real GPU memory.
    # This lets us trace models that wouldn't fit in memory during tracing.
    tracing_mode = "fake"
    fake_mode = FakeTensorMode()

    def _get_fake_args(arg: torch.Tensor) -> torch.Tensor:
        """Convert a real tensor to a FakeTensor (same shape/dtype, no data)."""
        fake_arg = fake_mode.from_tensor(arg)
        return fake_arg

    # Convert ALL tensor arguments to fake tensors
    args = pytree.tree_map_only(torch.Tensor, _get_fake_args, args)
    kwargs = pytree.tree_map_only(torch.Tensor, _get_fake_args, kwargs)

    # Actually trace the function:
    # - _enable_compile(): makes optimizer use the traceable code path
    # - detect_anomaly(check_nan=False): enables autograd anomaly detection without NaN checks
    # - decomposition_table=SPMD_DECOMP_TABLE: tells the tracer to decompose fused ops
    #   (e.g., _fused_adam_ → individual tensor ops) so every operation is a separate node
    with _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
        gm = make_fx(
            partial(stateless_func, func),
            tracing_mode=tracing_mode,
            decomposition_table=SPMD_DECOMP_TABLE,
            _allow_non_fake_inputs=False,
        )(params, buffers, named_states, args, kwargs)
    # At this point, gm.graph contains nodes for EVERY operation:
    # placeholder inputs → forward ops → sep → loss → sep_backward → backward ops → optimizer ops → output

    # ---- Step 4: Prepare the flat state for runtime execution ----
    # At runtime, we pass all state tensors + original args as a flat list to the graph
    params_and_buffers: Dict[str, Union[torch.Tensor, nn.Parameter]] = {
        **params,
        **buffers,
    }
    flat_state, _ = pytree.tree_flatten([params_and_buffers, named_states])
    # flat_state = [param1, param2, ..., buffer1, ..., opt_state1, opt_state2, ...]

    # ---- Step 5: Clean up the traced graph ----
    # Remove detach nodes: autograd inserts detach() calls that are noise in our graph.
    # Remove tag_grad nodes: we only needed them to identify gradients, not for execution.
    # Both are identity ops, so we just redirect their users to their inputs.
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)  # All users of detach(x) now use x directly
            if len(node.users) == 0:
                gm.graph.erase_node(node)  # Safe to remove since nobody uses it
        if node.target == torch.ops.dummy.tag_grad.default:
            grad_node = node.all_input_nodes[0]
            node.replace_all_uses_with(grad_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)

    # Switch to flat-input calling convention (caller flattens, graph doesn't)
    gm = _to_caller_flattened_graph_module(gm)

    return _CompiledResult(gm, mod, opt, flat_state)


# =============================================================================
# PUBLIC API: compile()
#
# This is what users call. It wraps a train_step function so that:
# - On the FIRST call: traces the function into an FX graph, applies the
#   user's graph_transformation (profiling, AC, etc.), and caches the result
# - On SUBSEQUENT calls: just runs the cached compiled graph (fast path)
# =============================================================================

COMPILED_OBJECT_KEY = "_compiled_obj"  # Key for storing compiled result on the wrapper function


def compile(func: Callable, gm_transformation: Callable):
    """
    Args:
        func: The training step function (e.g., train_step(model, optim, batch))
        gm_transformation: A callback that receives the traced graph and its inputs.
            This is where you do profiling, activation checkpointing, etc.
            Signature: (gm: fx.GraphModule, args: List[Tensor]) -> fx.GraphModule

    Returns:
        A wrapper function with the same signature as func.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        first_iter = False
        # Check if we've already compiled (stored on the wrapper function itself)
        compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
        if compiled_obj is None:
            first_iter = True
            # Trace the training step into an FX graph
            compiled_obj = _compile(func, *args, **kwargs)
            wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj

        # Build the flat input list: [all state tensors] + [original function args]
        # This matches the graph's placeholder nodes in order
        flat_inps = compiled_obj.flat_state + pytree.tree_flatten([args, kwargs])[0]

        if first_iter and gm_transformation:
            # Apply the user's transformation (profiling, optimization, etc.)
            # This only runs once — the transformed graph is cached for future calls
            compiled_obj.gm = gm_transformation(compiled_obj.gm, flat_inps)

        # Run the compiled graph. torch.no_grad() because the graph already
        # contains explicit backward ops — we don't want autograd to record
        # a second backward graph on top of it.
        # [0] indexes into the output tuple: (return_value, updated_params, updated_states)
        with torch.no_grad():
            output = compiled_obj.gm(*flat_inps)[0]

        return output

    return wrapper
