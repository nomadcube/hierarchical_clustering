from heapq import heapify

import numpy as np


class Pair:
    def __init__(self, i, j, val):
        self.i = i
        self.j = j
        self.val = val

    def __lt__(self, other):
        return self.val < other.val


class Node:
    def __init__(self, sample_data):
        try:
            if not isinstance(sample_data, np.ndarray):
                raise TypeError()
            self.sample_data = sample_data
        except TypeError:
            self.sample_data = np.array(sample_data)

    def variance(self):
        n = len(self.sample_data)
        sample_sum = 0
        for sample in self.sample_data:
            sample_sum += sample
        sample_average = sample_sum / float(n)
        tmp_variance = 0.0
        for sample in self.sample_data:
            tmp_variance += sum((sample - sample_average) ** 2)
        return tmp_variance / float(n)

    def union(self, other):
        if not isinstance(other, Node):
            raise TypeError()
        return Node(np.concatenate((self.sample_data, other.sample_data)))

    def __len__(self):
        return len(self.sample_data)

    def ward(self, other):
        if not isinstance(other, Node):
            raise TypeError()
        new_cluster = self.union(other)
        return new_cluster.variance() - (self.variance() + other.variance())


class HierarchicalTree:
    def __init__(self):
        self._all_nodes = []

    def add_node(self, data):
        self._all_nodes.append(Node(data))

    def __len__(self):
        return len(self._all_nodes)

    def nodes_to_be_merged(self):
        pass

    def ward_heapq(self):
        # todo : this is not right.
        heap_list = []
        for i in range(len(self)):
            for j in range(i + 1, len(self)):
                heap_list.append(Pair(i, j, self._all_nodes[i].ward(self._all_nodes[j])))
        heapify(heap_list)
        return heap_list


class MyList(list):
    def __init__(self):
        super(MyList, self).__init__()
        print('TestList initiate OK.')

if __name__ == '__main__':
    tl = MyList()
