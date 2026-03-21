# =============================================================================
# OPERATOR DECOMPOSITION TABLE
#
# When make_fx traces a training step, PyTorch uses many "fused" and "in-place"
# operators for performance. For example:
#   - _foreach_add_.List: adds two lists of tensors element-wise, IN-PLACE
#   - _fused_adam_: performs the entire Adam optimizer update as one fused kernel
#
# Problem: these fused ops appear as SINGLE nodes in the graph, hiding the
# individual operations inside them. For profiling and graph rewriting, we
# need every operation to be a separate, visible node.
#
# Solution: the SPMD_DECOMP_TABLE tells make_fx how to "decompose" each fused
# op into simpler, non-in-place operations. Each decomposition:
#   1. Calls the non-in-place version of the op (which creates a NEW tensor)
#   2. Explicitly copies the result back into the original tensor with .copy_()
#
# This makes every operation visible as a separate graph node, at the cost of
# some extra copy operations (which is fine for profiling/analysis).
#
# The table is passed to make_fx via decomposition_table=SPMD_DECOMP_TABLE
# in graph_tracer._compile().
# =============================================================================

from functools import partial
import torch
from torch._decomp.decompositions import native_layer_norm_backward
aten = torch.ops.aten  # Shorthand for accessing ATen (PyTorch's C++ tensor library) operators


# --- foreach decompositions ---
# PyTorch's _foreach_* ops operate on LISTS of tensors simultaneously.
# The in-place versions (ending with _) modify tensors in place.
# We decompose them into: non-in-place op → explicit copy_ back.

def _foreach_add_decomp(self, other, alpha=1):
    """Decompose _foreach_add_.List (in-place list addition) into
    non-in-place _foreach_add.List + explicit copy."""
    self_updated = aten._foreach_add.List(self, other, alpha=alpha)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)  # Copy result back into original tensor


def _foreach_unaop_decomp(op, self):
    """Decompose in-place unary foreach ops (neg, reciprocal, sqrt)."""
    self_updated = op(self)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_binop_list_decomp(op, self, other):
    """Decompose in-place binary foreach ops that take two lists (e.g., div)."""
    self_updated = op(self, other)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_binop_scalar_decomp(op, self, scalar=1):
    """Decompose in-place binary foreach ops that take a scalar (e.g., mul, div)."""
    self_updated = op(self, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_addcop_scalar_decomp(op, self, tensor1, tensor2, scalar=1):
    """Decompose in-place addcmul/addcdiv foreach ops.
    These compute: self += scalar * (tensor1 OP tensor2) for each element."""
    self_updated = op(self, tensor1, tensor2, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# --- Adam optimizer decomposition ---
# _fused_adam_ is a single fused kernel that does the entire Adam update.
# We decompose it into the non-in-place _fused_adam + explicit copies.
# This is critical because it makes the optimizer's internal operations
# (moment updates, parameter updates) visible as separate graph nodes.

def _fused_adam_decomp(
    self,           # List of parameter tensors
    grads,          # List of gradient tensors
    exp_avgs,       # List of first moment (mean) estimates
    exp_avg_sqs,    # List of second moment (variance) estimates
    max_exp_avg_sqs,  # List of max second moments (for AMSGrad)
    state_steps,    # List of step counters
    *,
    lr=1,           # Learning rate
    beta1=1,        # First moment decay rate
    beta2=1,        # Second moment decay rate
    weight_decay=1, # L2 regularization
    eps=1,          # Numerical stability epsilon
    amsgrad=True,   # Whether to use AMSGrad variant
    maximize=True,  # Whether to maximize (vs minimize) the objective
    grad_scale=None,
    found_inf=None,
):
    orig_tuple = (self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)

    # Call the non-in-place version — returns NEW tensors with updated values
    updated_tuple = aten._fused_adam.default(
        self,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )

    # Copy updated values back into the original tensors
    for idx, (orig, updated) in enumerate(zip(orig_tuple, updated_tuple)):
        if idx == 1:
            # Skip index 1 (gradients) — we don't need to write back to gradient
            # tensors since they'll be zeroed out by optim.zero_grad() anyway
            continue
        for o, u in zip(orig, updated):
            o.copy_(u)


# =============================================================================
# THE DECOMPOSITION TABLE
# Maps: in-place fused op → decomposition function
# Passed to make_fx(decomposition_table=SPMD_DECOMP_TABLE) during tracing.
# =============================================================================

SPMD_DECOMP_TABLE = {
    # foreach list addition: param += lr * grad
    aten._foreach_add_.List: _foreach_add_decomp,

    # foreach scalar addition: tensor += scalar
    aten._foreach_add_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_add.Scalar
    ),

    # foreach addcdiv: self += scalar * (tensor1 / tensor2) — used in Adam for param update
    aten._foreach_addcdiv_.Scalar: partial(
        _foreach_addcop_scalar_decomp, aten._foreach_addcdiv.Scalar
    ),

    # foreach addcmul: self += scalar * (tensor1 * tensor2) — used in Adam for moment updates
    aten._foreach_addcmul_.Scalar: partial(
        _foreach_addcop_scalar_decomp, aten._foreach_addcmul.Scalar
    ),

    # foreach division by list
    aten._foreach_div_.List: partial(
        _foreach_binop_list_decomp, aten._foreach_div.List
    ),

    # foreach scalar multiplication
    aten._foreach_mul_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_mul.Scalar
    ),

    # foreach scalar division
    aten._foreach_div_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_div.Scalar
    ),

    # foreach negation
    aten._foreach_neg_.default: partial(
        _foreach_unaop_decomp, aten._foreach_neg.default
    ),

    # foreach reciprocal (1/x)
    aten._foreach_reciprocal_.default: partial(
        _foreach_unaop_decomp, aten._foreach_reciprocal.default
    ),

    # foreach square root
    aten._foreach_sqrt_.default: partial(
        _foreach_unaop_decomp, aten._foreach_sqrt.default
    ),

    # foreach scalar subtraction
    aten._foreach_sub_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_sub.Scalar
    ),

    # Fused Adam optimizer — the big one
    aten._fused_adam_.default: _fused_adam_decomp,

    # Layer norm backward — decompose into individual gradient computations
    aten.native_layer_norm_backward.default: native_layer_norm_backward,
}