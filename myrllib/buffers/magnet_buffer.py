import numpy as np


class MagnetBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = []
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.capacity:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.capacity
        else:
            self.storage.append(data)

    def normalize(self):
        mean_list = np.mean(self.storage, axis=0)
        std_list = np.std(self.storage, axis=0)

        return mean_list, std_list
