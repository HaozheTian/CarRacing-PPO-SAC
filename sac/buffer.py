import torch
import numpy as np

class Transition():
    def __init__(self, obs: torch.Tensor, 
                 obs_next: torch.Tensor, 
                 act: torch.Tensor, 
                 reward: torch.Tensor, 
                 done: torch.Tensor):
        self.obs = obs
        self.obs_next = obs_next
        self.act = act
        self.reward = reward
        self.done = done

class ReplayBuffer():
    def __init__(self, env, buffer_size, device):
        self.device = device

        self.buffer_size = buffer_size
        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape

        self.obs_array = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.obs_next_array = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.act_array = np.zeros((self.buffer_size, *self.act_shape), dtype=np.float32)
        self.rew_array = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.done_array = np.zeros((self.buffer_size, 1), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs: np.ndarray, obs_next: np.ndarray, act: np.ndarray, rew: float, done: int):
        self.obs_array[self.ptr] = np.array(obs)
        self.obs_next_array[self.ptr] = np.array(obs_next)
        self.act_array[self.ptr] = np.array(act)
        self.rew_array[self.ptr] = np.array(rew)
        self.done_array[self.ptr] = np.array(done)

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True
            self.ptr = 0
    
    def sample(self, batch_size):
        if self.full:
            batch_idxs = (np.random.randint(1, self.buffer_size, size=batch_size) + self.ptr) % self.buffer_size
        else:
            batch_idxs = np.random.randint(0, self.ptr, size=batch_size)
        return self._get_samples(batch_idxs)
    
    def _get_samples(self, batch_idxs):
        data = (
            self.index(self.obs_array, batch_idxs),
            self.index(self.obs_next_array, batch_idxs),
            self.index(self.act_array, batch_idxs),
            self.index(self.rew_array, batch_idxs),
            self.index(self.done_array, batch_idxs),
        )
        return Transition(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray):
        return torch.tensor(array, device=self.device)
    
    def index(self, x, idxs):
        return x[idxs] if x.ndim==1 else x[idxs, :]