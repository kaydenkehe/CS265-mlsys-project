# =============================================================================
# BENCHMARKS
#
# This file provides an Experiment class for running the tracing/profiling
# pipeline on real models (Transformer, ResNet18, ResNet50) instead of the
# DummyModel in starter_code.py.
#
# For the final project deliverables, you'll run experiments on ResNet-152
# and BERT. This file currently supports ResNet18/50 and a small Transformer.
# You'll need to extend it for the required models.
# =============================================================================

import importlib
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fx as fx
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,       # Configuration dataclass for the Transformer model
    Transformer,     # A basic Transformer implementation from PyTorch internals
    )
from torchvision.models import resnet18, resnet50, resnet152
from transformers import BertForSequenceClassification, BertConfig
from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile
from phase2 import run_mutwo_algorithm, print_ac_decision, simulate_peak_memory


# Available model names for experiments
model_names: List[str] = [
    "Transformer",
    "Resnet18",
    "Resnet50",
    "Resnet152",
    "Bert",
]

# Default batch sizes per model — chosen to fit in typical GPU memory
# Larger models get smaller batches to avoid OOM
model_batch_sizes: Dict[str, int] = {
    "Transformer": 4,
    "Resnet18": 16,
    "Resnet50": 4,
    "Resnet152": 4,
    "Bert": 4,
}


class Experiment:
    """
    Encapsulates everything needed to run a training experiment:
    model, optimizer, data, loss function, train step, and graph transformation.

    Usage:
        exp = Experiment("Resnet18", batch_size=16)
        exp.init_opt_states()           # Initialize optimizer states for tracing
        compiled_fn = compile(exp.train_step, exp.graph_transformation)
        compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
    """

    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        assert model_name in model_names, f"Model {model_name} not found in model names {model_names}"
        dev = torch.device("cuda")
        self.model_name = model_name
        self.batch_size = batch_size

        if self.model_name == "Transformer":
            # ---- Transformer setup ----
            # A small Transformer with 8 layers, 4 attention heads, vocab of 2048
            # Input: token indices [batch_size, seq_len]
            # Output: logits [batch_size, seq_len, vocab_size]
            vocab_size = 2048
            bsz, seq_len = self.batch_size, 256
            with torch.device(dev):  # All tensors created in this block go to GPU
                model_args = ModelArgs(
                    n_layers=8,
                    n_heads=4,
                    vocab_size=vocab_size,
                    max_seq_len=seq_len,
                    dropout_p=0.1,
                )
                self.model = Transformer(model_args)

            # Random token sequences for source and target
            src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            self.example_inputs = (src, tgt)

            def transformer_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                # Forward: model(src) → logits, then cross-entropy loss against tgt
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)  # Insert forward/backward boundary
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = transformer_train_step
            # fused=True: uses a single CUDA kernel for the Adam update (faster)
            # capturable=True: makes Adam compatible with graph tracing/CUDA graphs
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, fused=True, capturable=True)

        elif self.model_name in ["Resnet18", "Resnet50", "Resnet152"]:
            # ---- ResNet setup ----
            # Input: images [batch_size, 3, 224, 224] (3 channels, 224x224 pixels)
            # Output: class logits [batch_size, 1000] (ImageNet classes)
            inp = torch.randn(self.batch_size, 3, 224, 224, device=dev)
            num_classes = 10
            target = torch.randint(0, num_classes, (self.batch_size,), device=dev)
            self.example_inputs = (inp, target)

            with torch.device(dev):
                if self.model_name == "Resnet18":
                    self.model = resnet18()
                elif self.model_name == "Resnet50":
                    self.model = resnet50()
                elif self.model_name == "Resnet152":
                    self.model = resnet152()

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                # Forward: model(images) → logits, then cross-entropy loss against targets
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)  # Insert forward/backward boundary
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, foreach=True, capturable=True)
            self.train_step = resnet_train_step

        elif self.model_name == "Bert":
            num_classes = 10 # we're using BERT as a sequence classifier
            seq_len = 128
            config = BertConfig(
                num_labels=num_classes,
                # we want deterministic behavior
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
            )
            with torch.device(dev):
                self.model = BertForSequenceClassification(config).to(dev)

            # same as transformer above
            input_ids = torch.randint(0, config.vocab_size, (self.batch_size, seq_len), device=dev)
            # BERT's output is [batch_size, num_classes], so targets are just class indices (not sequences)
            target = torch.randint(0, num_classes, (self.batch_size,), device=dev)
            self.example_inputs = (input_ids, target)

            def bert_train_step(model: nn.Module, optim: optim.Optimizer, example_inputs: Any):
                output = model(example_inputs[0])
                loss = self.loss_fn(output.logits, example_inputs[1])
                loss = SEPFunction.apply(loss)  # Insert forward/backward boundary
                loss.backward()
                optim.step()
                optim.zero_grad()
                
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, foreach=True, capturable=True)
            self.train_step = bert_train_step

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Cross-entropy loss that handles different output shapes.
        view(-1, ...) flattens batch and sequence dimensions for the loss computation.
        ignore_index=-1 skips padding tokens (relevant for Transformer).
        """
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def init_opt_states(self):
        """
        Pre-initialize optimizer states by running a dummy step.
        Adam lazily creates exp_avg, exp_avg_sq, and step tensors on the first
        call to .step(). We need these to exist BEFORE tracing so they appear
        as placeholder nodes in the traced graph.
        """
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)  # Fake gradients
        self.optimizer.step()       # Triggers Adam state initialization
        self.optimizer.zero_grad()  # Clean up fake gradients

    def graph_transformation(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        """
        Callback for compile() — receives the traced graph and profiles it.
        Same workflow as starter_code.graph_transformation:
          1. Print graph structure
          2. Profile with warm-up + measurement iterations
          3. Aggregate and print statistics
        """
        print(gm.graph.print_tabular())  # Tabular format: shows op, target, args for each node

        warm_up_iters, profile_iters = 2, 3
        graph_profiler = GraphProfiler(gm)

        with torch.no_grad():
            # Warm-up: stabilize GPU kernel caches and memory allocator
            for _ in range(warm_up_iters):
                graph_profiler.run(*args)
            graph_profiler.reset_stats()  # Discard warm-up measurements

            # Measurement: collect actual profiling data
            for _ in range(profile_iters):
                graph_profiler.run(*args)
            graph_profiler.aggregate_stats()  # Average over measurement runs
            graph_profiler.print_stats(
                save_prefix=f"results_{self.model_name}",
                model_name=self.model_name,
                batch_size=self.batch_size,
            )  # Output the results

            # Sanity: simulator should roughly match the measured peak with no eviction
            sim_peak, peak_step = simulate_peak_memory(graph_profiler, set())
            print(f"Simulator (no AC): {sim_peak / 1024**2:.1f} MB (peak at node index {peak_step})")
            print(f"Measured peak:     {graph_profiler.actual_peak_mem / 1024**2:.1f} MB")

            # Run AC with a budget below the measured peak
            budget = int(graph_profiler.actual_peak_mem * 0.6)
            print(f"Budget:            {budget / 1024**2:.1f} MB")
            decision = run_mutwo_algorithm(graph_profiler, budget)
            print_ac_decision(decision, graph_profiler)

        return gm

    def run(self):
        """Run a single training step without compilation (for sanity checking)."""
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


# =============================================================================
# MAIN: Run a benchmark experiment
#
# Usage:
#   python benchmarks.py <model> <batch_size>
#   python benchmarks.py Resnet152 4
#   python benchmarks.py Bert 8
#
# Available models: Transformer, Resnet18, Resnet50, Resnet152, Bert
# If no arguments, defaults to Resnet18 with its default batch size.
# =============================================================================

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "Resnet18"
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else model_batch_sizes[model]
    print(f"Running {model} with batch size {bs}")
    exp = Experiment(model, bs)
    exp.init_opt_states()
    compiled_fn = compile(exp.train_step, exp.graph_transformation)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
