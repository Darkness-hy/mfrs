### common lib
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch

### personal lib
from myrllib.networks.potential import Potential


class DPBA(object):
    def __init__(self, args, env, device):
        self.args = args
        self.env = env
        self.device = device

        self.p_net = Potential(args, self.env.state_dim, self.env.action_dim, self.env.goal_dim).to(device)
        self.p_net_optimizer = optim.Adam(self.p_net.parameters(), lr=args.lr_p)

    def expected_reward(self, state, goal):
        finger, goal_pos = state[9:12], goal[0:3]
        vec_finger_goal = finger - goal_pos
        l_finger_goal = np.sqrt(vec_finger_goal.dot(vec_finger_goal))

        if self.env.goal_dim == 6:
            obs_pos = goal[3:6]
            vec_finger_obs = finger - obs_pos

            l_finger_obs = np.sqrt(vec_finger_obs.dot(vec_finger_obs))
            expected_r = - l_finger_goal + l_finger_obs

        else:
            obs1_pos, obs2_pos, obs3_pos = goal[3:6], goal[6:9], goal[9:12]
            vec_finger_obs1 = finger - obs1_pos
            vec_finger_obs2 = finger - obs2_pos
            vec_finger_obs3 = finger - obs3_pos

            l_finger_obs1 = np.sqrt(vec_finger_obs1.dot(vec_finger_obs1))
            l_finger_obs2 = np.sqrt(vec_finger_obs2.dot(vec_finger_obs2))
            l_finger_obs3 = np.sqrt(vec_finger_obs3.dot(vec_finger_obs3))
            expected_r = - l_finger_goal + (l_finger_obs1 + l_finger_obs2 + l_finger_obs3) / 3

        return [-expected_r]

    def shaping(self, s, a, s_, a_, g):
        state = torch.FloatTensor(s).to(self.device)
        action = torch.FloatTensor(a).to(self.device)
        next_state = torch.FloatTensor(s_).to(self.device)
        next_action = torch.FloatTensor(a_).to(self.device)
        goal = torch.FloatTensor(g).to(self.device)

        expected_r = torch.FloatTensor(self.expected_reward(s_, g)).to(self.device)
        target_p = self.p_net(next_state, next_action, goal).detach()
        target_p = expected_r + self.args.gamma * target_p
        current_p = self.p_net(state, action, goal)

        p_loss = F.mse_loss(current_p, target_p)
        self.p_net_optimizer.zero_grad()
        p_loss.backward()
        self.p_net_optimizer.step()

        next_p = self.p_net(next_state, next_action, goal)
        shaping_reward = self.args.gamma * next_p - current_p

        return shaping_reward.item()