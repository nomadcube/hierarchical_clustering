from heapq import heapify, heappop

import numpy as np

from tree_base import MutableLinkedBinaryTree


class Pair:
    def __init__(self, i, j, val):
        self.i = i
        self.j = j
        self.val = val

    def __lt__(self, other):
        return self.val < other.val


class Node:
    def __init__(self, sample_data, sample_size):
        try:
            if not isinstance(sample_data, np.ndarray):
                raise TypeError()
            self.sample_data = sample_data
        except TypeError:
            self.sample_data = np.array(sample_data)
        self.sample_data = self.sample_data.reshape((sample_size, -1))

    def variance(self):
        if self.sample_data.ndim == 1:
            return 0.0
        else:
            n = self.sample_data.shape[0]
            sample_sum = 0
            for row_index in range(n):
                sample_sum += self.sample_data[row_index]
            sample_average = sample_sum / float(n)
            tmp_variance = 0.0
            for sample in self.sample_data:
                tmp_variance += sum((sample - sample_average) ** 2)
            return tmp_variance / float(n)

    def union(self, other):
        if not isinstance(other, Node):
            raise TypeError()
        return Node(np.concatenate((self.sample_data, other.sample_data)), len(self) + len(other))

    def __len__(self):
        return self.sample_data.shape[0]

    def ward(self, other):
        if not isinstance(other, Node):
            raise TypeError()
        if not isinstance(other.sample_data, np.ndarray):
            raise TypeError()
        new_cluster = self.union(other)
        if not isinstance(new_cluster.sample_data, np.ndarray):
            raise TypeError()
        return new_cluster.variance() - (self.variance() + other.variance())


class HierarchicalTree:
    def __init__(self, leaf_node_data):
        self.n_leaf = len(leaf_node_data)
        self._all_possible_nodes = [None] * (2 * self.n_leaf - 1)
        for i in range(self.n_leaf):
            self._all_possible_nodes[self.n_leaf - 1 + i] = Node(leaf_node_data[i], 1)
        self._ward_heap_list = []
        self._ward_heapify()

    def __len__(self):
        return len(self._all_possible_nodes)

    def _nodes_to_be_merged(self):
        result = heappop(self._ward_heap_list)
        return result

    def _ward_heapify(self):
        for i in range(len(self)):
            for j in range(i + 1, len(self)):
                node_i = self._all_possible_nodes[i]
                node_j = self._all_possible_nodes[j]
                if node_i is None or node_j is None:
                    continue
                self._ward_heap_list.append(Pair(i, j, node_i.ward(node_j)))
        heapify(self._ward_heap_list)


class ClusterForest:
    def __init__(self, leaf_data):
        self._all_tree = []
        for leaf in leaf_data:
            new_tree = MutableLinkedBinaryTree()
            new_tree._add_root(Node(leaf, 1))
            self._all_tree.append(new_tree)

    def size(self):
        return len(self._all_tree)
