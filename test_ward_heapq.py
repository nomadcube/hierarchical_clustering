import math

import numpy as np

from ward_heapq import Pair, Node, HierarchicalTree


class TestPair:
    def test_val(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        p = Pair(0, 0, math.sqrt(sum((a - b) ** 2)))
        assert p.val == 0


class TestCluster:
    def test_init(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        clu = Node([a, b])
        assert isinstance(clu.sample_data, np.ndarray)
        assert len(clu.sample_data) == 2

    def test_len(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        clu = Node([a, b])
        assert len(clu) == 2

    def test_variance(self):
        a = np.array([2, 2])
        b = np.array([0, 2])
        clu = Node([a, b])
        assert clu.variance() == 1.0
        a_clu = Node([a])
        assert a_clu.variance() == 0.0

    def test_union(self):
        a = np.array([2, 2])
        b = np.array([0, 2])
        a_clu = Node([a])
        b_clu = Node([b])
        c_clu = a_clu.union(b_clu)
        assert len(c_clu) == 2
        assert c_clu.variance() == 1.0
        assert a_clu.ward(b_clu) == 1.0


class TestHierarchicalTree:
    def test_add_node(self):
        ht = HierarchicalTree()
        all_data = [np.array([k, k ** 2]) for k in range(3)]
        for data in all_data:
            ht.add_node(data)
        assert len(ht) == 3

    def test_ward_heapq(self):
        ht = HierarchicalTree()
        all_data = [np.array([k, k ** 2], dtype='int8') for k in range(3)]
        for data in all_data:
            ht.add_node(data)
        assert isinstance(ht.ward_heapq(), list)
