import random
import sys

class Experience:
    def __init__(self, afterstate, reward, next_state, next_actions, done):
        self.afterstate = afterstate
        self.reward = reward
        self.next_state = next_state
        self.next_actions = next_actions
        self.done = done

    def get_size(self):
        return (sys.getsizeof(self.afterstate.indices().storage())
            + sys.getsizeof(self.afterstate.values().storage())
            + sys.getsizeof(self.next_state.indices().storage())
            + sys.getsizeof(self.next_state.values().storage())
            + sys.getsizeof(self.next_actions.indices().storage())
            + sys.getsizeof(self.next_actions.values().storage())
            + sys.getsizeof(self.afterstate)
            + sys.getsizeof(self.reward)
            + sys.getsizeof(self.done)
            + sys.getsizeof(self.next_state)
            + sys.getsizeof(self.next_actions))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
