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

class CNNQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        if obs_shape[0] == obs_shape[1]:
            h, w, c = obs_shape
            obs_shape = (c, h, w)
        self.encoder = Encoder(obs_shape, 128)
        self.fc1 = nn.Linear(128 + act_shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs, act):
        act = (act - self.action_bias)/self.action_scale
        obs_encoding = F.relu(self.encoder(obs))
        x = torch.cat([obs_encoding, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x