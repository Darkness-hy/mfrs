import torch.nn.functional as F
import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, args, state_dim, action_dim, goal_dim):
        super(Critic, self).__init__()
        self.goal_dim = goal_dim

        self.l1 = nn.Linear(state_dim + goal_dim * 2 + action_dim, args.hidden_dim)
        self.l2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.l3 = nn.Linear(args.hidden_dim, 1)

    def goal_emb(self, state, goal):
        a1_pos, finger, goal_pos = state[:, 6:9], state[:, 9:12], goal[:, 0:3]

        if self.goal_dim == 6:
            obs_pos = goal[:, 3:6]
            vec_1, vec_2 = goal_pos - a1_pos, goal_pos - finger
            vec_3, vec_4 = obs_pos - a1_pos, obs_pos - finger
            vec = torch.cat([state, vec_1, vec_2, vec_3, vec_4], dim=1)
        else:
            obs1_pos, obs2_pos, obs3_pos = goal[:, 3:6], goal[:, 6:9], goal[:, 9:12]
            vec_1, vec_2, vec_3, vec_4 = goal_pos - a1_pos, obs1_pos - a1_pos, obs2_pos - a1_pos, obs3_pos - a1_pos
            vec_5, vec_6, vec_7, vec_8 = goal_pos - finger, obs1_pos - finger, obs2_pos - finger, obs3_pos - finger
            vec = torch.cat([state, vec_1, vec_2, vec_3, vec_4, vec_5, vec_6, vec_7, vec_8], dim=1)

        return vec

    def forward(self, s, a, g):
        s = self.goal_emb(s, g)

        x = F.relu(self.l1(torch.cat([s, a], dim=1)))
        x = F.relu(self.l2(x))
        q = self.l3(x)

        return q