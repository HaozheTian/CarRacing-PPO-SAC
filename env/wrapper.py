import cv2
import numpy as np
import gymnasium
from gymnasium import spaces

def preprocess(img):
    img = img[:84, 6:90] # CarRacing-v2-specific cropping
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

class ImageEnv(gymnasium.Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, initial_no_op=50, **kwargs):
        super().__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.act_shape = env.action_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
    
    def reset(self):
        obs, info = self.env.reset()
        for _ in range(self.initial_no_op):
            obs, _, _, _, info = self.env.step(np.zeros(self.act_shape))

        obs = preprocess(obs)

        self.stacked_state = np.tile(obs, (self.stack_frames, 1, 1))
        return self.stacked_state, info
    
    def step(self, act):
        rew = 0
        for _ in range(self.skip_frames):
            obs, r, term, trun, info = self.env.step(act)
            rew += r
            if term or trun:
                break

        obs = preprocess(obs)
        self.stacked_state = np.concatenate((self.stacked_state[1:], obs[np.newaxis]), axis=0)

        return self.stacked_state, rew, term, trun, info