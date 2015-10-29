import numpy as np

from ward_heapq import Pair, Node


class TestPair:
    def pytest_funcarg__two_pair(self):
        pair_1 = 0, 0, 0.0
        pair_2 = 0, 1, 1.0
        return pair_1, pair_2

    def test_function(self, two_pair):
        assert len(two_pair) == 2

    def test_init(self, two_pair):
        pair = Pair(*two_pair[0])
        assert isinstance(pair, Pair)

    def test_lt(self, two_pair):
        pair_1 = Pair(*two_pair[0])
        pair_2 = Pair(*two_pair[1])
        assert pair_1 < pair_2


class TestNode:
    def pytest_funcarg__original_sample_size(self):
        return 3

    def pytest_funcarg__original_sample(self, original_sample_size):
        return np.array([0, 0, 1, 1, 2, 5]).reshape((original_sample_size, -1))

    def pytest_funcarg__two_node(self):
        node_0 = Node(0, containing_original_sample_index=[0])
        node_1 = Node(1, containing_original_sample_index=[1])
        return node_0, node_1

    def test_init(self, two_node, original_sample, original_sample_size):
        assert isinstance(two_node[0], Node)
        assert original_sample_size == 3
        assert original_sample.shape == (3, 2)

    def test_combine(self, two_node):
        cmb_original_sample_index = two_node[0].combine(two_node[1])
        assert isinstance(cmb_original_sample_index, list)
        assert len(cmb_original_sample_index) == 2
        assert cmb_original_sample_index == [0, 1]

    def test_variance(self, two_node, original_sample):
        var_1 = two_node[0].variance(original_sample)
        var_2 = two_node[1].variance(original_sample)
        assert var_1 == 0
        assert var_2 == 0

    def test_ward(self, two_node, original_sample):
        ward_pair = two_node[0].ward(two_node[1], original_sample)
        assert isinstance(ward_pair, Pair)
