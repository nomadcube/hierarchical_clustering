import math

import numpy as np

from ward_heapq import Node, HierarchicalTree, Pair


class TestPair:
    def test_val(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        p = Pair(0, 0, math.sqrt(sum((a - b) ** 2)))
        assert p.val == 0


class TestNode:
    def test_init(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        clu = Node([a, b], 2)
        assert isinstance(clu.sample_data, np.ndarray)
        assert clu.sample_data.shape == (2, 2)

    def test_len(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        clu = Node([a, b], 2)
        assert len(clu) == 2

    def test_variance(self):
        a = np.array([2, 2])
        b = np.array([0, 2])
        clu = Node([a, b], 2)
        assert clu.variance() == 1.0
        a_clu = Node([a], 1)
        assert a_clu.variance() == 0.0

    def test_union(self):
        a = np.array([2, 2])
        b = np.array([0, 2])
        a_clu = Node([a], 1)
        b_clu = Node([b], 1)
        c_clu = a_clu.union(b_clu)
        assert c_clu.variance() == 1.0
        assert a_clu.ward(b_clu) == 1.0


class TestHierarchicalTree:
    def test_init(self):
        all_leaf = [np.array([k, k ** 2]) for k in range(3)]
        ht = HierarchicalTree(all_leaf)
        assert len(ht) == 2 * 3 - 1
        assert len(ht._ward_heap_list) == 3

    def test_node_ward(self):
        all_data = [np.array([k, k ** 2], dtype='int8') for k in range(3)]
        ht = HierarchicalTree(all_data)
        i = 2 + 0
        j = 2 + 1
        w = ht._all_node[i].ward(ht._all_node[j])
        assert isinstance(w, float)
        assert w == 0.5

    def test_heappop(self):
        all_data = [np.array([k, k ** 2]) for k in range(3)]
        ht = HierarchicalTree(all_data)
        merge_pair = ht._nodes_to_be_merged()
        assert isinstance(merge_pair, Pair)
        assert merge_pair.i == 2
        assert merge_pair.j == 3
        assert merge_pair.val == 0.5
