### common lib
import torch.optim as optim
import torch

### personal lib
from myrllib.networks.aim_potential import Potential
from myrllib.buffers.replay_buffer import Replay_buffer


class AIM(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.p_net = Potential(args, state_dim=3, goal_dim=3).to(device)
        self.p_net_optimizer = optim.Adam(self.p_net.parameters(), lr=args.aim_lr)

        self.replay_buffer = Replay_buffer(args.aim_capacity)

    def update(self):
        for i in range(self.args.aim_iter):
            _, _, _, target, _, _ = self.replay_buffer.sample(self.args.batch_size)
            target = torch.FloatTensor(target)[:,0:3].to(self.device)
            target_noise = target + torch.normal(0, 0.01, size=target.shape).to(self.device)

            state, next_state, _, goal, _, _ = self.replay_buffer.sample(self.args.batch_size)
            state = torch.FloatTensor(state)[:,9:12].to(self.device)
            next_state = torch.FloatTensor(next_state)[:,9:12].to(self.device)
            goal = torch.FloatTensor(goal)[:,0:3].to(self.device)

            ## calculate term 1
            term_1 = - torch.mean(self.p_net(target_noise, target)) + torch.mean(self.p_net(next_state, goal))

            ## calculate term 2
            prev_out = self.p_net(state, goal)
            next_out = self.p_net(next_state, goal)
            term_2 = torch.max(torch.abs(next_out - prev_out) - 0.1, torch.tensor(0.)).pow(2).mean()

            # print('term_1: {} \t term_2: {}'.format(term_1, term_2))
            loss = term_1 + self.args.lam * term_2
            self.p_net_optimizer.zero_grad()
            loss.backward()
            self.p_net_optimizer.step()

    def shaping(self, next_state, state, goal):
        s_ = torch.FloatTensor(next_state[9:12]).to(self.device)
        # s = torch.FloatTensor(state[9:12]).to(self.device)
        g = torch.FloatTensor(goal[0:3]).to(self.device)

        # shaping_reward = self.args.gamma * self.p_net(s_, g) - self.p_net(s, g)
        shaping_reward = self.p_net(s_, g) - self.p_net(g, g)
        return shaping_reward.item()
