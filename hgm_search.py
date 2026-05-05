"""Reusable HGM tree-search primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

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
    if max_task_evals == n_task_evals:
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
    thetas = sampler.beta(alphas * scale, betas * scale)
    return int(np.argmax(thetas))


def should_expand(
    *,
    n_task_evals: int,
    node_count: int,
    n_pending_expands: int,
    alpha: float,
) -> bool:
    return n_task_evals**alpha >= node_count - 1 + n_pending_expands


def descendant_utility_measures(
    node,
    *,
    num_pseudo_descendant_evals: int,
) -> list[int | float]:
    measures = list(node.get_pseudo_decendant_evals(num_pseudo_descendant_evals))
    for descendant in node.get_sub_tree()[1:]:
        measures.extend(descendant.utility_measures)
    return measures
