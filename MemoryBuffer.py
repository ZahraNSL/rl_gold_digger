import numpy as np

class MemoryBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.order_buffer = []
    
    def add(self, state, action, reward, done, next_state):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
            self.order_buffer.pop(0)
        self.buffer.append((state, action, reward, done, next_state))
        self.order_buffer.append(len(self.buffer)-1)
    
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        idx = np.sort(idx)
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for i in idx:
            state, action, reward, done, next_state = self.buffer[self.order_buffer[i]]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
        return states, actions, rewards, dones, next_states
