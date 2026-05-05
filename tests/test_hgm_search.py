from __future__ import annotations

import numpy as np
import pytest

import hgm_utils
from hgm_search import (
    cooling_scale,
    descendant_utility_measures,
    should_expand,
    thompson_sample_index,
)
from tree import Node


class RecordingRng:
    def __init__(self, draws):
        self.draws = draws
        self.alphas = None
        self.betas = None

    def beta(self, alphas, betas):
        self.alphas = np.array(alphas)
        self.betas = np.array(betas)
        return np.array(self.draws)


def teardown_function():
    hgm_utils.nodes.clear()


def test_thompson_sample_uses_hgm_beta_parameters_without_cooldown():
    rng = RecordingRng([0.1, 0.9])

    selected = thompson_sample_index(
        [[1, 0, 1], [0]],
        n_task_evals=4,
        max_task_evals=100,
        beta=1.0,
        cool_down=False,
        rng=rng,
    )

    assert selected == 1
    assert rng.alphas.tolist() == [3.0, 1.0]
    assert rng.betas.tolist() == [2.0, 2.0]


def test_thompson_sample_applies_hgm_cooldown_scale():
    rng = RecordingRng([0.7])

    thompson_sample_index(
        [[1, 0]],
        n_task_evals=50,
        max_task_evals=100,
        beta=2.0,
        cool_down=True,
        rng=rng,
    )

    assert rng.alphas.tolist() == [8.0]
    assert rng.betas.tolist() == [8.0]
    assert cooling_scale(n_task_evals=100, max_task_evals=100, beta=2.0, cool_down=True) == 10000.0
    assert cooling_scale(n_task_evals=101, max_task_evals=100, beta=2.0, cool_down=True) == 10000.0


def test_thompson_sample_rejects_empty_candidates():
    with pytest.raises(ValueError, match="at least one candidate"):
        thompson_sample_index(
            [],
            n_task_evals=0,
            max_task_evals=10,
            beta=1.0,
            cool_down=False,
        )


def test_should_expand_matches_hgm_ucb_air_gate():
    assert should_expand(n_task_evals=4, node_count=3, n_pending_expands=0, alpha=0.5)
    assert not should_expand(n_task_evals=1, node_count=4, n_pending_expands=0, alpha=0.5)
    assert not should_expand(n_task_evals=4, node_count=3, n_pending_expands=2, alpha=0.5)


def test_descendant_utility_measures_uses_hgm_node_subtree_semantics():
    root = Node("root", utility_measures=[1, 0], id=0)
    child = Node("child", utility_measures=[1], parent_id=0, id=1)
    grandchild = Node("grandchild", utility_measures=[0, 1], parent_id=1, id=2)
    root.add_child(child)
    child.add_child(grandchild)

    assert descendant_utility_measures(root, num_pseudo_descendant_evals=10) == [
        1,
        0,
        1,
        0,
        1,
    ]
    assert root.utility_measures == [1, 0]


def test_node_descendant_eval_helpers_do_not_mutate_self_evals():
    root = Node("root", utility_measures=[1, 0], id=0)
    child = Node("child", utility_measures=[1], parent_id=0, id=1)
    root.add_child(child)

    assert root.get_decendant_evals(num_pseudo=10) == [1, 0, 1]
    assert root.get_descendant_evals(num_pseudo=10) == [1, 0, 1]
    assert root.utility_measures == [1, 0]
