from heapq import heapify, heappop
from copy import deepcopy

import numpy as np


class Pair:
    def __init__(self, i, j, val):
        self.i = i
        self.j = j
        self.val = val

    def __lt__(self, other):
        return self.val < other.val


class Node:
    def __init__(self, element, left=None, right=None, original_sample_index=list(), is_used=False):
        self.element = element
        self.left = left
        self.right = right
        self.original_sample_index = original_sample_index
        self.is_used = is_used

    def combine(self, other):
        combined_list = deepcopy(self.original_sample_index)
        combined_list.extend(other.original_sample_index)
        return combined_list

    def variance(self, selected_sample_index=None):
        if selected_sample_index is None:
            sample_data = self.original_sample_index
        else:
            sample_data = [self.original_sample_index[i] for i in range(len(self.original_sample_index)) if
                           i in selected_sample_index]
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

    def ward(self, other):
        combined_list = self.combine(other)
        ward = self.variance(combined_list) - (self.variance() + other.variance())
        new_pair = Pair(self.element, other.element, ward)
        return new_pair


class HierarchicalTree:
    def __init__(self, original_sample, sample_size):
        self.original_sample = np.array(original_sample).reshape((sample_size, -1))
        self._all_node = [Node(i) for i in range(sample_size)]
        self._ward_heap_list = []
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                self._ward_heap_list.append(Pair(i, j, self._all_node[i].ward(self._all_node[j])))
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
                new_heap_list.append(Pair(i, j, self._all_node[i].ward(self._all_node[j])))
        self._ward_heap_list = new_heap_list
        heapify(self._ward_heap_list)
