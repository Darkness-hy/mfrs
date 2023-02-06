### common lib
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch

### personal lib
from myrllib.utils.magnetization import Cuboid, Sphere
from myrllib.buffers.magnet_buffer import MagnetBuffer
from myrllib.networks.potential import Potential


class MFRS(object):
    def __init__(self, args, env, device):
        self.args = args
        self.env = env
        self.device = device

        self.obs_num = int((self.env.goal_dim / 3) - 1)
        self.mean_list, self.std_list = np.zeros(self.obs_num + 1), np.ones(self.obs_num + 1)
        self.magnet_buffer = MagnetBuffer(args.magnet_capacity)

        if self.obs_num == 1:
            self.sphere_goal = Sphere(self.env.goal_r)
            self.cuboid_obs = Cuboid(self.env.obs_a, self.env.obs_b, self.env.obs_h)
        else:
            self.sphere_goal = Sphere(self.env.goal_r)
            self.sphere_obs1 = Sphere(self.env.obs1_r)
            self.sphere_obs2 = Sphere(self.env.obs2_r)
            self.sphere_obs3 = Sphere(self.env.obs3_r)

        self.p_net = Potential(args, self.env.state_dim, self.env.action_dim, self.env.goal_dim).to(device)
        self.p_net_optimizer = optim.Adam(self.p_net.parameters(), lr=args.lr_p)

    def update_norm(self):
        self.mean_list, self.std_list = self.magnet_buffer.normalize()

    def expected_reward(self, state, goal):
        finger, goal_pos = state[9:12], goal[0:3]
        finger_goal_pos = finger - goal_pos

        if self.obs_num == 1:
            finger_obs_pos = finger - self.env.obs_mag_pos

            h_goal = self.sphere_goal.mfi(finger_goal_pos[0], finger_goal_pos[1], finger_goal_pos[2])
            h_obs = self.cuboid_obs.mfi(finger_obs_pos[0], finger_obs_pos[1], finger_obs_pos[2])
            self.magnet_buffer.push([h_goal, h_obs])

            norm_goal = (h_goal - self.mean_list[0]) / (self.std_list[0] + self.args.eps)
            norm_obs = (h_obs - self.mean_list[1]) / (self.std_list[1] + self.args.eps)  # (1-1)
            expected_r = norm_goal - norm_obs

        else:
            obs1_pos, obs2_pos, obs3_pos = goal[3:6], goal[6:9], goal[9:12]
            finger_obs1_pos = finger - obs1_pos
            finger_obs2_pos = finger - obs2_pos
            finger_obs3_pos = finger - obs3_pos

            h_goal = self.sphere_goal.mfi(finger_goal_pos[0], finger_goal_pos[1], finger_goal_pos[2])
            h_obs1 = self.sphere_obs1.mfi(finger_obs1_pos[0], finger_obs1_pos[1], finger_obs1_pos[2])
            h_obs2 = self.sphere_obs2.mfi(finger_obs2_pos[0], finger_obs2_pos[1], finger_obs2_pos[2])
            h_obs3 = self.sphere_obs3.mfi(finger_obs3_pos[0], finger_obs3_pos[1], finger_obs3_pos[2])
            self.magnet_buffer.push([h_goal, h_obs1, h_obs2, h_obs3])

            norm_goal = (h_goal - self.mean_list[0]) / (self.std_list[0] + self.args.eps)
            norm_obs1 = (h_obs1 - self.mean_list[1]) / (self.std_list[1] + self.args.eps)
            norm_obs2 = (h_obs2 - self.mean_list[2]) / (self.std_list[2] + self.args.eps)
            norm_obs3 = (h_obs3 - self.mean_list[3]) / (self.std_list[3] + self.args.eps)
            expected_r = norm_goal - (norm_obs1 + norm_obs2 + norm_obs3) / 3

        expected_r = expected_r / (1 + np.fabs(expected_r))
        return [-expected_r]

    def shaping(self, s_, a_, s, a, g):
        next_state = torch.FloatTensor(s_).to(self.device)
        next_action = torch.FloatTensor(a_).to(self.device)
        state = torch.FloatTensor(s).to(self.device)
        action = torch.FloatTensor(a).to(self.device)
        goal = torch.FloatTensor(g).to(self.device)

        expected_r = torch.FloatTensor(self.expected_reward(s_, g)).to(self.device)
        target_p = self.p_net(next_state, next_action, goal)
        target_p = expected_r + self.args.gamma * target_p.detach()
        current_p = self.p_net(state, action, goal)

        p_loss = F.mse_loss(current_p, target_p)
        self.p_net_optimizer.zero_grad()
        p_loss.backward()
        self.p_net_optimizer.step()

        next_p = self.p_net(next_state, next_action, goal)
        shaping_reward = self.args.gamma * next_p - current_p

        return shaping_reward.item()
