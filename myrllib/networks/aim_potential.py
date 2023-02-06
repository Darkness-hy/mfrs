import torch.nn.functional as F
import torch.nn as nn
import torch


class Potential(nn.Module):
    def __init__(self, args, state_dim, goal_dim):
        super(Potential, self).__init__()

        self.l1 = nn.Linear(state_dim + goal_dim, args.hidden_dim)
        self.l2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.l3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, s, g):
        x = torch.cat([s, g], dim=-1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        p = self.l3(x)

        return p