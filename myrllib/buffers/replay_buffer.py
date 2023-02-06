import numpy as np


class Replay_buffer(object):
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

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s, s_, a, g, r, d = [], [], [], [], [], []

        for i in ind:
            # (state, next_state, action, goal, reward, done)
            state, next_state, action, goal, reward, done = self.storage[i]

            s.append(np.array(state, copy=False))
            s_.append(np.array(next_state, copy=False))
            a.append(np.array(action, copy=False))
            g.append(np.array(goal, copy=False))
            r.append(np.array(reward, copy=False))
            d.append(np.array(done, copy=False))

        return np.array(s), np.array(s_), np.array(a), np.array(g), \
               np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)