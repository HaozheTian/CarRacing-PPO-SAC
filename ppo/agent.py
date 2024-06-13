import torch
import numpy as np
from typing import Union, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime

from ppo.buffer import PPOBuffer, BatchData
from ppo.network import CNNPolicy, CNNValue



def obs2ten(x: np.ndarray, target_device: torch.device) -> torch.Tensor:
    return torch.Tensor(x).permute(2, 0, 1).unsqueeze(0).to(target_device)



def ten2arr(x: torch.Tensor) -> np.ndarray:
    return x.squeeze(0).detach().cpu().numpy()



class Logger:
    def __init__(self):
        self.eps_returns = []
        self.eps_lengths = []
    
    def add(self, eps_info: Dict):
        self.eps_returns.append(eps_info['eps_return'])
        self.eps_lengths.append(eps_info['eps_length'])
    
    def mean(self, var: str) -> float:
        if hasattr(self, var):
            values = getattr(self, var)
            return sum(values) / len(values) if values else 0
        else:
            raise AttributeError(f"'Logger' object has no attribute '{var}'")
        


class PPO:
    def __init__(self, env, **kwargs):
        self.env = env
        # run-time status
        self.env_name = env.__class__.__name__
        self.time_str = datetime.now().strftime("_%m_%d_%Y_%H_%M")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on {self.device}')

        self._init_hyperparameters(**kwargs)
        self._init_seed()
        self._init_networks()
        self.buffer = PPOBuffer(self.device, self.gam)
        self.logger = Logger()
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/PPO_'+self.env_name+self.time_str)


    def learn(self):
        self.num_eps = 0
        with tqdm(total=self.total_steps) as pbar:      # progress bar
            step = 0
            while step < self.total_steps:
                # collect data (contains multiple episodes)
                data, epoch_steps = self.rollout()
                # update networks
                self.update(data)
                # record the number of steps in the epoch
                step += epoch_steps
                # update progress bar
                pbar.update(epoch_steps)


    def rollout(self) -> Union[BatchData, int]:
        """
        Collect data for one epoch (contains multiple episodes)
        """
        epoch_step = 0
        while epoch_step < self.min_epoch_steps:
            obs, _ = self.env.reset(seed=self.seed)
            eps_ret, eps_len = 0, 0
            while True:
                act, logp, _ = self.get_action(obs)
                obs_next, rew, term, trun, _ = self.env.step(act)
                val = self.evaluate(obs)
                self.buffer.add(obs, act, rew, val, logp)

                obs = obs_next
                eps_ret, eps_len = eps_ret+rew, eps_len+1

                if term or trun:
                    # calculate reward-to-gos and advantages
                    last_val = 0 if term else self.evaluate(obs)
                    self.buffer.path_done(last_val)

                    epoch_step += eps_len
                    self.num_eps += 1
                    eps_info = {'eps_return': eps_ret, 'eps_length': eps_len}
                    self.logger.add(eps_info)
                    if self.use_tb:
                        self._to_tb(eps_info)
                    break
        return self.buffer.get(), epoch_step
    

    def update(self, data: BatchData):
        adv = data.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-10) # advantage normalization
        for _ in range(self.num_updates):
            # update policy
            log_probs = self.policy(data.observations.permute(0, 3, 1, 2), data.actions).squeeze()
            ratio = torch.exp(log_probs - data.log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
            loss_policy = (-torch.min(surr1, surr2)).mean()

            self.policy_optimizer.zero_grad()
            loss_policy.backward()
            self.policy_optimizer.step()

            # update value
            loss_val = ((self.value(data.observations.permute(0, 3, 1, 2)).squeeze() - data.reward_to_gos)**2).mean()

            self.value_optimizer.zero_grad()
            loss_val.backward()
            self.value_optimizer.step()
    

    def get_action(self, obs: np.ndarray) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            act, logp, mean = self.policy(obs2ten(obs, self.device))
        return ten2arr(act), ten2arr(logp), ten2arr(mean)
    

    def evaluate(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            v_obs = self.value(obs2ten(obs, self.device))
        return ten2arr(v_obs)[0]


    def _init_hyperparameters(self, **kwargs):
        self.seed = kwargs.get('seed', 0)
        self.min_epoch_steps = kwargs.get('min_epoch_steps', 4000)
        self.total_steps = kwargs.get('total_steps', 30000000)
        self.clip_ratio = kwargs.get('clip_ratio', 0.2)
        self.num_updates = kwargs.get('num_updates', 80)
        self.gam = kwargs.get('gamma', 0.95)
        self.policy_lr = kwargs.get('policy_lr', 0.005)
        self.value_lr = kwargs.get('v_lr', 0.005)
        self.use_tb = kwargs.get('use_tb', False)
    

    def _init_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


    def _init_networks(self):
        h, w, c = self.env.observation_space.shape
        in_shape = (c, h, w)
        act_dim = self.env.action_space.shape[0]
        self.policy = CNNPolicy(in_shape, act_dim).to(self.device)
        self.value = CNNValue(in_shape).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = Adam(self.value.parameters(), lr=self.value_lr)


    def _to_tb(self, eps_info):
        for name, scalar in eps_info.items():
            self.writer.add_scalar(f'charts/{name}', scalar, self.num_eps)




