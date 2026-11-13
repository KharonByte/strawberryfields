# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Count-presence conversion must not materialize one entry per photon."""

from collections import defaultdict
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from strawberryfields.apps import sample, subgraph

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


LAW_ROWS = [(0, 0, 0), (2, 0, 0), (1, 0, 0), (0, 1, 0), (1, 2, 0), (0, 0, 3)]
LAW_WEIGHTS = [Fraction(n, 8) for n in (1, 2, 1, 1, 2, 1)]
LAW_SUPPORTS = [[], ["zeta"], ["zeta"], ["alpha"], ["alpha", "zeta"], ["mu"]]


def law_graph():
    graph = nx.Graph()
    graph.add_nodes_from(["zeta", "alpha", "mu"])
    graph.add_edges_from([("zeta", "alpha"), ("alpha", "mu")])
    return graph


def support_masses(outputs):
    masses = defaultdict(Fraction)
    for support, weight in zip(outputs, LAW_WEIGHTS):
        masses[frozenset(support)] += weight
    return dict(masses)


def test_exact_terminal_support_law():
    """Independent finite law: exact support masses, moments and selection order."""
    graph = law_graph()
    outputs = sample.to_subgraphs(LAW_ROWS, graph)
    assert outputs == LAW_SUPPORTS
    assert support_masses(outputs) == {
        frozenset(): Fraction(1, 8),
        frozenset(["zeta"]): Fraction(3, 8),
        frozenset(["alpha"]): Fraction(1, 8),
        frozenset(["alpha", "zeta"]): Fraction(1, 4),
        frozenset(["mu"]): Fraction(1, 8),
    }
    assert sum(w * sum(row) for row, w in zip(LAW_ROWS, LAW_WEIGHTS)) == Fraction(15, 8)
    assert sum(w * len(row) for row, w in zip(outputs, LAW_WEIGHTS)) == Fraction(9, 8)
    assert [nx.density(graph.subgraph(row)) for row in outputs] == [0, 0, 0, 0, 1, 0]
    photon_rows = sample.postselect(LAW_ROWS, 2, 2)
    assert photon_rows == [(2, 0, 0)]
    assert sample.to_subgraphs(photon_rows, graph) == [["zeta"]]
    occupancy = [[int(n > 0) for n in row] for row in LAW_ROWS]
    click_rows = sample.postselect(occupancy, 2, 2)
    assert click_rows == [[1, 1, 0]]
    assert sample.to_subgraphs(click_rows, graph) == [["alpha", "zeta"]]
    assert sum(w for row, w in zip(LAW_ROWS, LAW_WEIGHTS) if sum(row) == 2) == Fraction(1, 4)
    assert sum(w for row, w in zip(occupancy, LAW_WEIGHTS) if sum(row) == 2) == Fraction(1, 4)


def test_terminal_row_order_and_multiplicity():
    rows = [(0, 0, 3), (2, 0, 0), (0, 0, 0), (1, 2, 0), (0, 0, 3)]
    assert sample.to_subgraphs(rows, law_graph()) == [
        ["mu"],
        ["zeta"],
        [],
        ["alpha", "zeta"],
        ["mu"],
    ]


@pytest.mark.parametrize("row", [[0, 0, 1, 2], [0, 0, 1, 1]])
def test_count_support_search_composition(row):
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 2), (2, 3)])
    supports = sample.to_subgraphs([row], graph)
    assert subgraph.search(supports, graph, 2, 2, 1) == {2: [(1.0, [2, 3])]}


@pytest.mark.parametrize("defect", ["equals_one", "values", "sorted_nodes", "reverse", "dedup"])
def test_support_oracle_rejects_mutants(defect):
    """Local defective doubles must fail the intended equality, not setup."""
    nodes = ["zeta", "alpha", "mu"]
    rows = [(0, 0, 3), (2, 0, 0), (0, 0, 0), (1, 2, 0), (0, 0, 3)]
    expected = [["mu"], ["zeta"], [], ["alpha", "zeta"], ["mu"]]
    if defect == "sorted_nodes":
        nodes = sorted(nodes)
    wrong = [
        sorted(nodes[i] for i, c in enumerate(row) if (c == 1 if defect == "equals_one" else c > 0))
        for row in rows
    ]
    if defect == "values":
        wrong = [sorted(set(row)) for row in rows]
    if defect == "reverse":
        wrong.reverse()
    if defect == "dedup":
        wrong = [list(row) for row in dict.fromkeys(tuple(row) for row in wrong)]
    with pytest.raises(AssertionError):
        assert wrong == expected


def test_selection_and_coercion_mutants():
    wrong_selection = sample.to_subgraphs([[1, 1, 0]], law_graph())
    with pytest.raises(AssertionError):
        assert wrong_selection == [["zeta"]]
    row = [0.7, 1.9, -0.7, -2]
    assert sample.to_subgraphs([row], nx.empty_graph(4)) == [[1]]
    wrong_coercion = [i for i, c in enumerate(row) if c > 0]
    with pytest.raises(AssertionError):
        assert wrong_coercion == [1]


def test_helper_mutant_is_rejected(monkeypatch):
    def reject_expansion(_row):
        raise AssertionError("per-photon expansion")

    monkeypatch.setattr(sample, "modes_from_counts", reject_expansion)
    with pytest.raises(AssertionError, match="per-photon expansion"):
        list(set(sample.modes_from_counts([1000, 0, 100])))
