import gymnasium
import torch
import numpy as np
from datetime import datetime
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sac.network import CNNActor, CNNQNetwork
from sac.buffer import ReplayBuffer

class SAC():
    def __init__(self, env: gymnasium.Env, **kwargs) ->  None:
        self.env = env
        self._init_hyperparameters(**kwargs)
        self._seed()
        self._init_networks()
        self._init_buffer()
        # Logging
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/SAC_'+self.env_name+self.time_str)
        self.num_eps = 0
        self.eps_rets, self.eps_lens = [], []


    def learn(self):
        eps_ret, eps_len = 0, 0
        obs, info = self.env.reset()
        for step in tqdm(range(self.total_steps)):
            # get action
            act = self.get_act(obs, step)
            assert self.env.action_space.contains(act), f"Invalid action {act} in the action space"

            # interact with the environment
            obs_next, rew, term, trun, info = self.env.step(act)
            done = term or trun

            # record transition in the replay buffer
            self.rb.add(obs, obs_next, act, rew, done)
            eps_ret, eps_len = eps_ret + rew, eps_len + 1

            # IMPORTANT: do not over look
            obs = obs_next

            if done:
                self.num_eps += 1
                self.eps_rets.append(eps_ret)
                self.eps_lens.append(eps_len)
                if self.use_tb:
                    self.writer.add_scalar('charts/episode_return', eps_ret, self.num_eps)
                    self.writer.add_scalar('charts/episode_length', eps_len, self.num_eps)
                
                # reset environment
                eps_ret, eps_len = 0, 0
                obs, info = self.env.reset()
        
            if step >= self.learning_starts:
                self.update()


    def update(self):
        data = self.rb.sample(self.batch_size)

        # Q-NETWORK UPDATE
        # compute target for the Q functions
        with torch.no_grad():
            act_next, act_next_log_prob, _ = self.actor.get_action(data.obs_next)
            qf1_next = self.qf1_target(data.obs_next, act_next)
            qf2_next = self.qf2_target(data.obs_next, act_next)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha*act_next_log_prob
            y = data.reward.flatten() + (1 - data.done.flatten()) * self.gamma * (min_qf_next).view(-1)
        # compute loss for the  Q functions
        qf_1 = self.qf1(data.obs, data.act).view(-1)
        qf_2 = self.qf2(data.obs, data.act).view(-1)
        qf1_loss = F.mse_loss(qf_1, y)
        qf2_loss = F.mse_loss(qf_2, y)
        qf_loss = qf1_loss + qf2_loss
        # update the Q functions
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # POLICY UPDATE
        # update the policy
        if self.global_step % self.policy_freq == 0: # TD3 Delayed update support
            for _ in range(self.policy_freq):
                acts, log_prob, _ = self.actor.get_action(data.obs)
                qf1 = self.qf1(data.obs, acts)
                qf2 = self.qf2(data.obs, acts)
                min_qf = torch.min(qf1, qf2).view(-1)
                # negative sign for maximization
                actor_loss = -(min_qf - self.alpha * log_prob).mean()
                # update parameters
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

        # UPDATE TARGET Q-NETWORKS
        if self.global_step % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) 


    def get_act(self, obs: np.ndarray, step: int) -> np.ndarray:
        if self.ckpt_path == None and step<self.learning_starts:
            act = self.env.action_space.sample()
        else:
            act, _, _ = self.actor.get_action(torch.Tensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device))
            act = act.squeeze(0).detach().cpu().numpy()
        return act


    def _init_hyperparameters(self, **kwargs):
        self.env_name = self.env.__class__.__name__
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running on {self.device}')
        self.time_str = datetime.now().strftime("_%m_%d_%Y_%H_%M")

        self.seed = kwargs.get('seed', 0)
        self.use_tb = kwargs.get('use_tb', True)
        self.ckpt_path = kwargs.get('ckpt_path', None)
        self.q_lr = kwargs.get('q_lr', 1e-3)                # learning rate for Q network
        self.policy_lr = kwargs.get('policy_lr', 3e-4)      # learning rate for policy network
        self.buffer_size = kwargs.get('buffer_size', 100000)   # replay buffer size
        self.batch_size = kwargs.get('batch_size', 1024)    # batch size for updating network
        self.total_steps = kwargs.get('total_steps', 1000000)   # maximum number of iterations
        self.learning_starts = self.batch_size              # start learning
        self.tau = kwargs.get('tau', 0.005)                 # for updating Q target
        self.gamma = kwargs.get('gamma', 0.99)              # forgetting factor
        self.alpha = kwargs.get('alpha', 0.2)               # entropy tuning parameter
        self.policy_freq = kwargs.get('policy_freq', 2)     # frequency for updating policy network
        self.target_q_freq = kwargs.get('target_q_freq', 1) # frequency for updating target network                    # displaying logs


    def _seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)


    def _init_networks(self):
        self.actor = CNNActor(self.env).to(self.device)
        self.qf1 = CNNQNetwork(self.env).to(self.device)
        self.qf2 = CNNQNetwork(self.env).to(self.device)
        self.qf1_target = CNNQNetwork(self.env).to(self.device)
        self.qf2_target = CNNQNetwork(self.env).to(self.device)
        if self.ckpt_path == None:
            print('Training from scratch')
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
        else:
            print('Training from the checkpoint in {self.ckpt_path}')
            self._load_ckpt(torch.load(self.ckpt_path))
        self.q_optimizer = Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)
        self.actor_optimizer = Adam(list(self.actor.parameters()), lr=self.policy_lr)


    def _init_buffer(self):
        self.rb = ReplayBuffer(
            self.env,
            self.buffer_size,
            self.device,
        )