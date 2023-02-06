### common lib
import torch.nn.functional as F
import torch.optim as optim
import torch

### person lib
from myrllib.networks.actor import Actor
from myrllib.networks.critic import Critic
from myrllib.buffers.replay_buffer import Replay_buffer


class DDPG(object):
    def __init__(self, args, state_dim, action_dim, max_action, goal_dim, device):
        self.actor = Actor(args, state_dim, action_dim, max_action, goal_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_a)
        self.actor_target = Actor(args, state_dim, action_dim, max_action, goal_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(args, state_dim, action_dim, goal_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_c)
        self.critic_target = Critic(args, state_dim, action_dim, goal_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_buffer = Replay_buffer(args.buffer_capacity)
        self.args, self.device = args, device

    def select_action(self, state, goal):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(self.device)
        return self.actor(state, goal).cpu().data.numpy().flatten()

    def update(self):
        for _ in range(self.args.update_iter):
            # Sample replay buffer
            s, s_, a, g, r, d = self.replay_buffer.sample(self.args.batch_size)

            state = torch.FloatTensor(s).to(self.device)
            next_state = torch.FloatTensor(s_).to(self.device)
            action = torch.FloatTensor(a).to(self.device)
            goal = torch.FloatTensor(g).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            done = torch.FloatTensor(d).to(self.device)

            # update Q network
            target_Q = self.critic_target(next_state, self.actor_target(next_state, goal), goal)
            target_Q = reward + ((1 - done) * self.args.gamma * target_Q).detach()
            current_Q = self.critic(state, action, goal)

            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # update policy network
            actor_loss = - self.critic(state, self.actor(state, goal), goal).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def save(self, directory):
        torch.save(self.actor.state_dict(), directory + '/actor.pth')
        torch.save(self.critic.state_dict(), directory + '/critic.pth')
        print("-----------------------------------------------------")
        print("Saving model at : " + directory)
        print("-----------------------------------------------------")

    def load(self, directory):
        self.actor.load_state_dict(torch.load(directory + '/actor.pth'))
        self.critic.load_state_dict(torch.load(directory + '/critic.pth'))
        print("-----------------------------------------------------")
        print("Loading model at : " + directory)
        print("-----------------------------------------------------")