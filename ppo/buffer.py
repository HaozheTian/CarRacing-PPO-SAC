import torch
import scipy
import numpy as np



def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    Compute the discounted cumulative sum

    Input:
        x = [x0, x1, x2]
    Output:
        [x0 + d * x1 + d^2 * x2, x1 + d * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



def gae_lambda(rews: np.ndarray, vals: np.ndarray, gamma: float, lam: float=0.97) -> np.ndarray:
    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    return discount_cumsum(deltas, gamma*lam)



class BatchData:
    def __init__(self, device: torch.device, 
                 observations: np.ndarray, 
                 actions: np.ndarray, 
                 log_probs: np.ndarray, 
                 reward_to_gos: np.ndarray, 
                 advantages: np.ndarray):
        self.observations = torch.tensor(np.array(observations), dtype=torch.float32, device=device)
        self.actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        self.log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32, device=device)
        self.reward_to_gos = torch.tensor(np.array(reward_to_gos), dtype=torch.float32, device=device)
        self.advantages = torch.tensor(np.array(advantages), dtype=torch.float32, device=device)



class PPOBuffer:
    def __init__(self, device: torch.device, gamma: float=0.99) -> None:
        self.device = device
        self.gamma = gamma
        self._init_buffers()
    

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float) -> None:
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)
        self.ptr += 1


    def path_done(self, last_val: float) -> None:
        """
        Call this after each episode to compute reward-to-gos and advantages.

        Input:
            - last-val (float): 0 if the trajectory ended because the agent 
              reached a terminal state (died), and V(s_T) otherwise (cut-off).
        """
        # index from episode start to episode end (now)
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(np.array(self.rew_buf, dtype=np.float32)[path_slice], last_val)
        vals = np.append(np.array(self.val_buf, dtype=np.float32)[path_slice], last_val)

        self.rtg_buf += discount_cumsum(rews, self.gamma)[:-1].tolist()
        self.adv_buf += gae_lambda(rews, vals, self.gamma).tolist()

        self.path_start_idx = self.ptr
        

    def get(self) -> BatchData:
        data = BatchData(self.device, self.obs_buf, self.act_buf, self.logp_buf, 
                         self.rtg_buf, self.adv_buf)
        self._init_buffers()
        return data
        

    def _init_buffers(self)->None:
        self.ptr, self.path_start_idx = 0, 0

        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.logp_buf = []

        self.rtg_buf = []
        self.adv_buf = []