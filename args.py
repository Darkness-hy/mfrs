import argparse


def get_args():
    parser = argparse.ArgumentParser()

    ## basic setting
    parser.add_argument('--env', type=str, default='Dobot-v1', help='the name of environment')
    parser.add_argument('--shaper', type=str, default='mfrs', help='the shaper to train the agent')
    parser.add_argument('--max_episode', type=int, default=10000, help='total training episodes')
    parser.add_argument('--max_step', type=int, default=1000, help='the number of timesteps of each episode')
    parser.add_argument('--plot_num', type=int, default=500, help='the number of nodes for plotting fig')
    parser.add_argument('--exploration_noise', type=float, default=0.4, help='the exploration noise')

    ## train setting
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.001, help='soft update for target network')
    parser.add_argument('--update_iter', type=int, default=100, help='update iterations of policy')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of policy update')
    parser.add_argument('--lr_a', type=float, default=3e-4, help='learning rate of actor')
    parser.add_argument('--lr_c', type=float, default=1e-3, help='learning rate of critic')

    ## buffer setting
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--magnet_capacity', type=int, default=int(1e6), help='capacity of magnet buffer')

    ## network setting
    parser.add_argument('--hidden_dim', type=int, default=256, help='number of hidden layers')

    ## shaper setting
    parser.add_argument('--lr_p', type=float, default=1e-4, help='learning rate of potential')
    parser.add_argument('--eps', type=float, default=1e-7, help='a small number')

    parser.add_argument('--aim_lr', type=float, default=1e-3, help='learning rate of AIM')
    parser.add_argument('--aim_iter', type=int, default=1, help='update iterations of AIM')
    parser.add_argument('--aim_capacity', type=int, default=int(1e4), help='capacity of smaller replay buffer')
    parser.add_argument('--lam', type=float, default=10., help='trade off hyper-parameter of AIM')

    args = parser.parse_args()
    return args

