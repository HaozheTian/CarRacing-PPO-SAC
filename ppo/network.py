import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNN(nn.Module):
    def __init__(self, in_shape, out_size) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
            nn.ReLU()
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
    def __init__(self, in_shape):
        "in_shape starts with the number of channels, i.e. (C, H, W)"
        super().__init__()
        self.layers = nn.Sequential(
            CNN(in_shape, out_size=256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1))
        )

    def forward(self, x):
        return self.layers(x)
    
class CNNPolicy(nn.Module):
    def __init__(self, in_shape, out_size):
        "in_shape starts with the number of channels, i.e. (C, H, W)"
        super().__init__()
        self.layers = nn.Sequential(
            CNN(in_shape, out_size=256),
            nn.ReLU(),
            layer_init(nn.Linear(256, out_size))
        )
        # adaptive std for the stochastic policy
        log_std = -0.5 * np.ones(out_size, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    
    def forward(self, obs, act=None):
        mean = self.layers(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        if act == None:
            act = dist.sample()
            log_prob_act = dist.log_prob(act).sum(axis=-1)
            return act, log_prob_act, mean
        else:
            return dist.log_prob(act).sum(axis=-1)