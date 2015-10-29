from heapq import heapify, heappop
from copy import deepcopy

import numpy as np


class Pair:
    """Represent a pair of tree nodes."""

    def __init__(self, i, j, val):
        """
        Parameter i and j means index of these two nodes.
        Parameter val means the value of WARD(node_i, node_j).
        """
        self.i = i
        self.j = j
        self.val = val

    def __lt__(self, other):
        """Redefine __lt__ function for heapify."""
        return self.val < other.val


class Node:
    """Represent one Node in Hierarchical Tree."""

    def __init__(self, element, left=None, right=None, containing_original_sample_index=None, is_used=False):
        """
        :param element: Index in all sample data of this node itself.
        :param left: Index of its left child.
        :param right: Index of its right child.
        :param containing_original_sample_index: All leaf index contained by this node.
        :param is_used: Determine if this node has been contained by some other node.
        """
        if not isinstance(element, int):
            raise TypeError('Element must be of Type int.')
        self.element = element
        self.left = left
        self.right = right
        if not isinstance(containing_original_sample_index, list):
            raise TypeError('Original sample index must be of Type list.')
        self.containing_original_sample_index = containing_original_sample_index
        self.is_used = is_used

    def combine(self, other):
        """Union all leaf index contained by two node together."""
        combined_list = deepcopy(self.containing_original_sample_index)
        combined_list.extend(other.containing_original_sample_index)
        return combined_list

    def variance(self, all_sample, selected_sample_index=None):
        # todo: too ugly, needs modifying.
        if selected_sample_index is None:
            if self.containing_original_sample_index is None:
                return 0.0
            else:
                sample_data = np.array([all_sample[_] for _ in self.containing_original_sample_index])
        else:
            sample_data = [all_sample[i] for i in selected_sample_index]
            sample_data = np.array(sample_data).reshape((len(selected_sample_index), -1))
        if sample_data.ndim == 1:
            return 0.0
        else:
            n = sample_data.shape[0]
            sample_sum = 0
            for row_index in range(n):
                sample_sum += sample_data[row_index]
            sample_average = sample_sum / float(n)
            tmp_variance = 0.0
            for sample in sample_data:
                tmp_variance += sum((sample - sample_average) ** 2)
            return tmp_variance / float(n)

    def ward(self, other, original_sample):
        combined_list = self.combine(other)
        ward = self.variance(combined_list, original_sample) - (
        self.variance(original_sample) + other.variance(original_sample))
        new_pair = Pair(self.element, other.element, ward)
        return new_pair


class HierarchicalTree:
    def __init__(self, original_sample, sample_size):
        self.original_sample = np.array(original_sample).reshape((sample_size, -1))
        self._all_node = [Node(i) for i in range(sample_size)]
        self._ward_heap_list = []
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                self._ward_heap_list.append(Pair(i, j, self._all_node[i].ward(self._all_node[j], self.original_sample)))
        heapify(self._ward_heap_list)


    def select_two_node(self):
        selected_pair = heappop(self._ward_heap_list)
        return selected_pair.i, selected_pair.j

    def generate_new_node(self):
        left, right = self.select_two_node()
        index = len(self._all_node)
        original_sample_index = self._all_node[left].combine(self._all_node[right])
        node = Node(index, left, right, original_sample_index)
        self._all_node[left].is_used = True
        self._all_node[right].is_used = True
        self._all_node.append(node)
        new_heap_list = []
        for i in range(len(self._all_node)):
            if self._all_node[i].is_used:
                continue
            for j in range(i + 1, len(self._all_node)):
                if self._all_node[j].is_used:
                    continue
                new_heap_list.append(Pair(i, j, self._all_node[i].ward(self._all_node[j], self.original_sample)))
        self._ward_heap_list = new_heap_list
        heapify(self._ward_heap_list)
