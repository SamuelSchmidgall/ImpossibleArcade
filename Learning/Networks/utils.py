import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def identity(x):
    return x

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Adam:
    def __init__(self, params, stepsize, epsilon=1e-08, beta1=0.99, beta2=0.999):
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.params = params
        self.dim = params.size
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.m = np.zeros(params.size, dtype=np.float32)
        self.v = np.zeros(params.size, dtype=np.float32)

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.params
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        new_theta = self.params + step
        self.params = new_theta
        return ratio, new_theta

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


