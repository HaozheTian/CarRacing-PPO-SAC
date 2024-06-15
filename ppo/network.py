import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F



def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class Encoder(nn.Module):
    def __init__(self, in_shape, out_size) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_shape[0], 16, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            nn.Flatten()
        )

        # compute conv output size
        with torch.inference_mode():
            output_size = self.conv(torch.zeros(1, *in_shape)).shape[1]
        self.fc = layer_init(nn.Linear(output_size, out_size))

    def forward(self, x):
        x = self.conv(x/255.0)
        x = self.fc(x)
        return x



class CNNValue(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape

        self.obs_encoder = Encoder(obs_shape, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, obs):
        obs_encoding = F.relu(self.obs_encoder(obs))
        x = F.relu(self.fc1(obs_encoding))
        x = self.fc2(x)
        return x
    


class CNNPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape

        self.obs_encoder = Encoder(obs_shape, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, act_shape[0])

        # adaptive std for the stochastic policy
        log_std = -0.5 * np.ones(act_shape[0], dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    
    def forward(self, obs, act=None):
        obs_encoding = F.relu(self.obs_encoder(obs))
        x = F.relu(self.fc1(obs_encoding))
        mean = self.fc2(x)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        if act == None:
            act = dist.sample()
            log_prob_act = dist.log_prob(act).sum(axis=-1)
            return act, log_prob_act, mean
        else:
            return dist.log_prob(act).sum(axis=-1)