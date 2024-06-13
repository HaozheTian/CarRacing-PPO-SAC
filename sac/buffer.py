import gymnasium
from gymnasium import spaces
import numpy as np
import torch
from typing import Union



class Samples():
    def __init__(self, obs: torch.Tensor, 
                 obs_next: torch.Tensor, 
                 act: torch.Tensor, 
                 rew: torch.Tensor, 
                 done: torch.Tensor):
        self.obs = obs
        self.obs_next = obs_next
        self.act = act
        self.rew = rew
        self.done = done



class ReplayBuffer():
    def __init__(self, env: gymnasium.Env,
                 buffer_size: int,
                 device: torch.device):
        self.buffer_size = buffer_size
        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape

        self.ptr = 0
        self.full = False
        self.device = device

        self.obs_array = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.uint8)
        self.obs_next_array = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.uint8)
        self.act_array = np.zeros((self.buffer_size, *self.act_shape), dtype=np.float32)
        self.rew_array = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.done_array = np.zeros((self.buffer_size, 1), dtype=np.float32)


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
    

    def sample(self, batch_size: int) -> Samples:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.ptr) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.ptr, size=batch_size)
        return self._get_samples(batch_inds)
    

    def _get_samples(self, batch_inds):
        data = (
            self.index(self.obs_array, batch_inds),
            self.index(self.obs_next_array, batch_inds),
            self.index(self.act_array, batch_inds),
            self.index(self.rew_array, batch_inds),
            self.index(self.done_array, batch_inds),
        )
        return Samples(*tuple(map(self.to_torch, data)))


    def to_torch(self, array: np.ndarray):
        return torch.tensor(array, device=self.device)
    

    def index(self, x, inds):
        return x[inds] if x.ndim==1 else x[inds, :]
