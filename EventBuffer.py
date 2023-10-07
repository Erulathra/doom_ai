import numpy as np


class EventBuffer:
    def __init__(self, n, capacity=100, event_clip=0.01):
        self.n = n
        self.capacity = capacity
        self.idx = 0
        self.events = []
        self.event_clip = event_clip

    def record_events(self, events):
        if len(self.events) < self.capacity:
            self.events.append(events)
        else:
            self.events[self.idx] = events
            if self.idx + 1 < self.capacity:
                self.idx += 1
            else:
                self.idx = 0

    def intrinsic_reward(self, events):
        if len(self.events) == 0:
            return 0

        mean = np.mean(self.events, axis=0)
        clip = np.clip(mean, self.event_clip, np.max(mean))
        div = np.divide(np.ones(self.n), clip)
        mul = np.multiply(div, events)

        return np.sum(mul)

    def get_event_mean(self):
        if len(self.events) == 0:
            return np.zeros(self.n)
        mean = np.mean(self.events, axis=0)
        return mean
