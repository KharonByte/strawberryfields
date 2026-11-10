# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Count-presence conversion must not materialize one entry per photon."""

import networkx as nx
import numpy as np
import pytest

from strawberryfields.apps import sample

pytestmark = pytest.mark.apps


@pytest.mark.parametrize(
    "counts, expected",
    [
        ([0, 0, 0], []),
        ([2, 0, 3], [0, 2]),
        ([False, True, -2, 0.9, 1.9, "2"], [1, 4, 5]),
        ([np.int64(2), np.float64(0.5), np.float64(3)], [0, 2]),
        ([], []),
    ],
)
def test_count_presence(counts, expected):
    """Retain the existing integer conversion without adding validation."""
    graph = nx.empty_graph(len(counts))
    assert sample.to_subgraphs([counts], graph) == [expected]


def test_sparse_mode_order():
    """Preserve the existing set iteration order rather than sorting modes."""
    counts = [0, 2, 0, 0, 0, 0, 0, 0, 3]
    assert sample.to_subgraphs([counts], nx.empty_graph(9)) == [list({1, 8})]


@pytest.mark.parametrize("labels", [[30, 10, 20], [2, 0, 1], ["c", "a", "b"]])
def test_node_insertion_mapping(labels):
    """Map mode positions through node insertion order, then sort labels."""
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    assert sample.to_subgraphs([[2, 0, 3]], graph) == [sorted([labels[0], labels[2]])]


def test_no_photon_expansion(monkeypatch):
    """A bounded count input must bypass the per-photon helper entirely."""

    def reject_expansion(_counts):
        raise AssertionError("to_subgraphs must not expand photon counts")

    monkeypatch.setattr(sample, "modes_from_counts", reject_expansion)
    assert sample.to_subgraphs([[1000, 0, 100]], nx.empty_graph(3)) == [[0, 2]]


@pytest.mark.parametrize(
    "count, error",
    [(None, TypeError), ("bad", ValueError), (np.nan, ValueError), (np.inf, OverflowError)],
)
def test_conversion_errors(count, error):
    with pytest.raises(error):
        sample.to_subgraphs([[count]], nx.empty_graph(1))


def test_existing_width_behavior():
    """Do not introduce graph-width validation in this bounded fix."""
    graph = nx.empty_graph(2)
    assert sample.to_subgraphs([[1]], graph) == [[0]]
    assert sample.to_subgraphs([[0, 0, 1]], graph) == [[2]]
    mapped = nx.relabel_nodes(graph, {0: 10, 1: 20})
    assert sample.to_subgraphs([[1, 0, 0]], mapped) == [[10]]
    with pytest.raises(IndexError):
        sample.to_subgraphs([[0, 0, 1]], mapped)


def test_conversion_precedes_mapping():
    """All sample counts are converted before any node lookup can fail."""
    graph = nx.Graph()
    graph.add_node(10)
    with pytest.raises(ValueError):
        sample.to_subgraphs([[0, 1], ["bad"]], graph)


def test_empty_samples():
    assert sample.to_subgraphs([], nx.empty_graph(0)) == []
    assert sample.to_subgraphs([[]], nx.empty_graph(0)) == [[]]


def test_samples_and_graph_unchanged():
    counts = [[2, 0, 1], [0, 0, 0], [2, 0, 1]]
    graph = nx.Graph()
    graph.add_nodes_from([30, 10, 20])
    assert sample.to_subgraphs(counts, graph) == [[20, 30], [], [20, 30]]
    assert counts == [[2, 0, 1], [0, 0, 0], [2, 0, 1]]
    assert list(graph.nodes) == [30, 10, 20]


def test_conversion_order():
    calls = []

    class Count:
        def __init__(self, value):
            self.value = value

        def __int__(self):
            calls.append(self.value)
            return self.value

    counts = [[Count(0), Count(-1)], [Count(2), Count(3)]]
    assert sample.to_subgraphs(counts, nx.empty_graph(2)) == [[], [0, 1]]
    assert calls == [0, -1, 2, 3]


def test_incomparable_labels():
    graph = nx.Graph()
    graph.add_nodes_from([0, "a"])
    with pytest.raises(TypeError):
        sample.to_subgraphs([[1, 1]], graph)


def test_modes_from_counts_retains_multiplicity():
    assert sample.modes_from_counts([0, 1, 0, 1, 2, 0]) == [1, 3, 4, 4]
