"""Reusable HGM tree-search primitives.

**In-band observability.** ``thompson_sample_index`` and
``should_expand`` emit OpenTelemetry spans carrying the full math
state (cooling scale, posterior alphas/betas, theta draws, selection
index) so a downstream checker can verify recorded behavior against
the paper's equations. The OTel import is guarded by try/except --
when HGM is used standalone without a tracer provider, span calls
are zero-cost NoOps and no new dependency is required from HGM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

try:
    from opentelemetry import trace as _otel_trace

    _tracer = _otel_trace.get_tracer("hgm.hgm_search")
except ImportError:  # pragma: no cover -- HGM may be used without otel installed
    _tracer = None

__all__ = [
    "cooling_scale",
    "descendant_utility_measures",
    "should_expand",
    "thompson_sample_index",
]


class _BetaSampler(Protocol):
    def beta(self, a, b): ...


def cooling_scale(
    *,
    n_task_evals: int,
    max_task_evals: int,
    beta: float,
    cool_down: bool,
) -> float:
    if not cool_down:
        return 1.0
    if n_task_evals >= max_task_evals:
        return 10000.0
    return max_task_evals**beta / (max_task_evals - n_task_evals) ** beta


def thompson_sample_index(
    evals: Sequence[Sequence[int | float]],
    *,
    n_task_evals: int,
    max_task_evals: int,
    beta: float,
    cool_down: bool,
    rng: _BetaSampler | None = None,
) -> int:
    """One Thompson draw over the per-candidate Beta posteriors.

    Paper: ``theta_a ~ Beta(tau * (1 + s_a), tau * (1 + f_a))``
    where ``tau = cooling_scale(...)``. The highest theta is selected.

    Emits one OTel span ``hgm.thompson_sample_index`` per call with
    every math input and output as a span attribute (cooling_scale,
    posterior_alphas, posterior_betas, theta draws, selected_index).
    Span emission is a NoOp when no tracer provider is initialized.
    """
    if not evals:
        raise ValueError("evals must contain at least one candidate")

    alphas = np.array([1 + np.sum(candidate) for candidate in evals], dtype=float)
    betas = np.array(
        [1 + len(candidate) - np.sum(candidate) for candidate in evals],
        dtype=float,
    )
    scale = cooling_scale(
        n_task_evals=n_task_evals,
        max_task_evals=max_task_evals,
        beta=beta,
        cool_down=cool_down,
    )
    sampler = rng or np.random
    posterior_alphas = alphas * scale
    posterior_betas = betas * scale
    thetas = sampler.beta(posterior_alphas, posterior_betas)
    selected = int(np.argmax(thetas))

    if _tracer is not None:
        with _tracer.start_as_current_span("hgm.thompson_sample_index") as span:
            span.set_attributes(
                {
                    "hgm.n_task_evals": int(n_task_evals),
                    "hgm.max_task_evals": int(max_task_evals),
                    "hgm.beta": float(beta),
                    "hgm.cool_down": bool(cool_down),
                    "hgm.cooling_scale": float(scale),
                    "hgm.candidate_count": int(len(evals)),
                    "hgm.candidate_alphas": posterior_alphas.tolist(),
                    "hgm.candidate_betas": posterior_betas.tolist(),
                    "hgm.candidate_thetas": thetas.tolist(),
                    "hgm.selected_index": selected,
                    "hgm.selected_alpha": float(posterior_alphas[selected]),
                    "hgm.selected_beta": float(posterior_betas[selected]),
                    "hgm.selected_theta": float(thetas[selected]),
                }
            )

    return selected


def should_expand(
    *,
    n_task_evals: int,
    node_count: int,
    n_pending_expands: int,
    alpha: float,
) -> bool:
    """UCB-Air expansion gate. Paper: expansion permitted when ``N_t^alpha >= |T_t|``.

    The implementation generalizes the paper's inequality to parallel
    dispatch by adding ``n_pending_expands`` to the RHS, treating
    in-flight (claimed but not yet completed) expansions as if they
    had already landed. Under sequential dispatch ``n_pending_expands``
    is always 0 and the gate reduces to the paper's exact form.
    """
    lhs = float(n_task_evals) ** alpha
    rhs = node_count - 1 + n_pending_expands
    decision = lhs >= rhs

    if _tracer is not None:
        with _tracer.start_as_current_span("hgm.should_expand") as span:
            span.set_attributes(
                {
                    "hgm.n_task_evals": int(n_task_evals),
                    "hgm.node_count": int(node_count),
                    "hgm.n_pending_expands": int(n_pending_expands),
                    "hgm.alpha": float(alpha),
                    "hgm.lhs": lhs,
                    "hgm.rhs": float(rhs),
                    "hgm.decision_expand": decision,
                }
            )

    return decision


def descendant_utility_measures(
    node,
    *,
    num_pseudo_descendant_evals: int,
) -> list[int | float]:
    measures = list(node.get_pseudo_descendant_evals(num_pseudo_descendant_evals))
    for descendant in node.get_sub_tree()[1:]:
        measures.extend(descendant.utility_measures)
    return measures
