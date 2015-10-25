from math import sqrt
from heapq import heapify, heappop
from random import randrange

import numpy as np


class Pair:
    def __init__(self, i, j, val):
        self.i = i
        self.j = j
        self.val = val

    def __lt__(self, other):
        return self.val <= other.val


def euclidead_dist(x, y):
    if (not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray)):
        raise TypeError('x and y should be of ndarray Type.')
    return sqrt(sum((x - y) ** 2))


if __name__ == '__main__':
    samples = [np.array([randrange(10), randrange(10)]) for _ in range(10)]
    dist_list = list()
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            item = Pair(i, j, euclidead_dist(samples[i], samples[j]))
            dist_list.append(item)
    heapify(dist_list)
    first = heappop(dist_list)
    print(samples[first.i], samples[first.j], first.val)
