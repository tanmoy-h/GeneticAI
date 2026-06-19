"""
Smoke test for learnable theta_low (Option B REINFORCE).

Verifies:
  1. is_latent_step() populates _step_decisions
  2. compute_theta_loss() returns a finite tensor with grad_fn
  3. Gradient flows to theta_low_param via backward()
  4. Optimizer step actually moves theta_low_param
  5. warmup_scale=0 suppresses decisions and loss
  6. Gradient sync hook can be registered

Uses ast to extract LatentSpControllerOptB and the sync hook from
adaptive_thinking_residual_w9_optB.py without importing the whole module
(avoids needing trl, peft, evo2, etc.).

Usage (no GPU required):
  python week11tests/test_theta_low_reinforce.py
"""

import ast
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from transformers import TrainerCallback

sys.path.insert(0, ".")

# ── Extract and compile just the relevant classes from the production file ─────
_SRC = open("adaptive_thinking_residual_w9_optB.py", encoding="utf-8").read()
_TREE = ast.parse(_SRC)

_EXTRACT = {"LatentSpControllerOptB", "LatentSpWarmupCallbackOptB",
            "PreferLatestOnTieCallback", "ThinkingResidualGRPOTrainer_OptB"}

_nodes = [
    node for node in ast.walk(_TREE)
    if isinstance(node, ast.ClassDef) and node.name in _EXTRACT
]

_globs = {
    "__builtins__": __builtins__,
    "os": os, "torch": torch, "nn": nn, "F": F,
    "Optional": Optional, "List": List, "Tuple": Tuple,
    "dataclass": dataclass, "field": field,
    "TrainerCallback": TrainerCallback,
    # stubs for base classes that aren't being tested
    "ThinkingResidualGRPOTrainer": object,
    "GRPOScriptArgumentsW9": object,
    "make_thinking_residual_param_groups": lambda **kw: [],
    "ThinkingResidualGate": object,
    "DNAHiddenInjector": object,
}

for _node in _nodes:
    _code = compile(ast.Module(body=[_node], type_ignores=[]), "<extracted>", "exec")
    exec(_code, _globs)

LatentSpControllerOptB        = _globs["LatentSpControllerOptB"]
PreferLatestOnTieCallback     = _globs["PreferLatestOnTieCallback"]
ThinkingResidualGRPOTrainer_OptB = _globs["ThinkingResidualGRPOTrainer_OptB"]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_decisions_populated():
    ctrl = LatentSpControllerOptB(theta_low_init=1.0, theta_high=3.0, alpha=5.0)
    ctrl._warmup_scale = 1.0

    for entropy in [0.3, 0.7, 1.2, 2.0, 3.5]:
        ctrl.is_latent_step(entropy, consecutive=0)

    assert len(ctrl._step_decisions) == 5, (
        f"Expected 5 decisions, got {len(ctrl._step_decisions)}"
    )
    print(f"  [PASS] decisions={ctrl._step_decisions}")


def test_theta_loss_has_grad():
    ctrl = LatentSpControllerOptB(theta_low_init=1.0, theta_high=3.0, alpha=5.0)
    ctrl._warmup_scale = 1.0

    for entropy in [0.5, 1.0, 1.5]:
        ctrl.is_latent_step(entropy, consecutive=0)

    loss = ctrl.compute_theta_loss(adv_mean=1.0)
    assert loss is not None, "compute_theta_loss returned None"
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert loss.requires_grad, "loss has no grad_fn — gradient won't flow"
    print(f"  [PASS] theta_loss={loss.item():.6f}  grad_fn={loss.grad_fn}")


def test_gradient_flows():
    ctrl = LatentSpControllerOptB(theta_low_init=1.0, theta_high=3.0, alpha=5.0)
    ctrl._warmup_scale = 1.0

    for entropy in [0.5, 0.8, 1.2]:
        ctrl.is_latent_step(entropy, consecutive=0)

    loss = ctrl.compute_theta_loss(adv_mean=2.0)
    loss.backward()

    assert ctrl.theta_low_param.grad is not None, "theta_low_param.grad is None"
    assert torch.isfinite(ctrl.theta_low_param.grad), "gradient not finite"
    print(f"  [PASS] grad={ctrl.theta_low_param.grad.item():.6f}")


def test_optimizer_moves_param():
    ctrl = LatentSpControllerOptB(theta_low_init=1.0, theta_high=3.0, alpha=5.0)
    ctrl._warmup_scale = 1.0
    init_val = ctrl.theta_low_param.item()

    opt = torch.optim.AdamW([ctrl.theta_low_param], lr=1e-2)
    for _ in range(5):
        ctrl.clear_decisions()
        for entropy in [0.4, 0.9, 1.5]:
            ctrl.is_latent_step(entropy, consecutive=0)
        loss = ctrl.compute_theta_loss(adv_mean=1.0)
        if loss is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

    final_val = ctrl.theta_low_param.item()
    assert final_val != init_val, (
        f"theta_low_param did not move: {init_val} -> {final_val}"
    )
    print(f"  [PASS] theta_low: {init_val:.6f} -> {final_val:.6f}  "
          f"(delta={final_val - init_val:+.6f})")


def test_warmup_suppresses():
    ctrl = LatentSpControllerOptB(theta_low_init=1.0, alpha=5.0)
    ctrl._warmup_scale = 0.0

    for entropy in [0.5, 1.0, 2.0]:
        result = ctrl.is_latent_step(entropy, consecutive=0)
        assert result is False, "is_latent_step must return False during warmup"

    assert len(ctrl._step_decisions) == 0, "No decisions should be recorded during warmup"
    loss = ctrl.compute_theta_loss(adv_mean=1.0)
    assert loss is None, "compute_theta_loss should return None during warmup"
    print("  [PASS] warmup_scale=0 -> no decisions, loss=None")


def test_theta_clamped():
    ctrl = LatentSpControllerOptB(theta_low_init=1.0, alpha=5.0)
    ctrl._warmup_scale = 1.0

    with torch.no_grad():
        ctrl.theta_low_param.fill_(10.0)
    assert ctrl.theta_low < 8.0 + 1e-6, (
        f"theta_low not clamped at 8.0: got {ctrl.theta_low}"
    )
    with torch.no_grad():
        ctrl.theta_low_param.fill_(-5.0)
    assert ctrl.theta_low > 0.05 - 1e-6, (
        f"theta_low not clamped at 0.05: got {ctrl.theta_low}"
    )
    print("  [PASS] theta_low_param clamped to [0.05, 8.0]")


def test_sync_hook_registered():
    ctrl = LatentSpControllerOptB(theta_low_init=1.0)
    ctrl.theta_low_param.register_hook(ThinkingResidualGRPOTrainer_OptB._sync_theta_grad)
    assert len(ctrl.theta_low_param._backward_hooks) > 0, "Sync hook not registered"
    print("  [PASS] gradient sync hook registered")


if __name__ == "__main__":
    tests = [
        ("decisions populated",        test_decisions_populated),
        ("theta_loss has grad_fn",      test_theta_loss_has_grad),
        ("gradient flows to param",     test_gradient_flows),
        ("optimizer moves param",       test_optimizer_moves_param),
        ("warmup_scale=0 suppresses",   test_warmup_suppresses),
        ("theta_low_param clamped",     test_theta_clamped),
        ("grad sync hook registered",   test_sync_hook_registered),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
