### common lib
import numpy as np
import os

### personal lib
from myrllib.utils.generator import generate
from myrllib.utils.plotter import Plotter


class Runner(object):
    def __init__(self, args, device, seed_num):
        print("Seed : {} \t\t Env : {} \t\t Shaper : {}".format(seed_num, args.env, args.shaper))
        self.env, self.learner, self.shaper = generate(args, device)
        self.args, self.seed = args, seed_num

        self.root_dir = './saves/{}/{}/seed_{}'.format(args.env, args.shaper, seed_num)
        self.log_dir = self.root_dir + '/log'
        self.fig_dir = self.root_dir + '/fig'
        self.model_dir = self.root_dir + '/model'

        if args.shaper == 'sr':
            self.log_freq = int(args.max_episode / 2 / args.plot_num)
        else:
            self.log_freq = int(args.max_episode / args.plot_num)

        self.plotter = Plotter(self.log_dir, self.fig_dir, self.log_freq)
        self.running_data = np.zeros(3)  # [step, result, episodes]
        self.avg_array = np.zeros(shape=(2, args.plot_num))  # [step, result]

    @ staticmethod
    def mkdir(path):
        folder = os.path.exists(path)
        if not folder: os.makedirs(path)

    def log_data(self, data_list, episode):  # [step, result, 1]
        self.running_data += np.array(data_list)

        if episode % self.log_freq == 0:
            for i in range(self.avg_array.shape[0]):
                avg_data = round(self.running_data[i] / self.running_data[-1], 4)
                self.avg_array[i][int(episode / self.log_freq) - 1] = avg_data

            self.running_data = np.zeros(3)

    def train(self):
        if self.args.shaper == 'sr':
            for episode in range(int(self.args.max_episode/2)):
                ## reset state and goal
                state = self.env.reset()
                if self.env.goal_dim == 6:
                    goal = np.concatenate([self.env.goal_pos, self.env.obs_pos])
                else:
                    goal = np.concatenate([self.env.goal_pos, self.env.obs1_pos, self.env.obs2_pos, self.env.obs3_pos])

                ep_step, ep_result = 0, 0
                tr_list = []
                for ep in range(2):
                    tr_step, is_crash, tr_result = 0, 0, 0
                    temp_buffer = []
                    while True:
                        action = self.learner.select_action(state, goal)
                        action = action + np.random.normal(0, self.args.exploration_noise)
                        next_state, reward, done, crash = self.env.step(action)

                        temp_buffer.append((state, next_state, action, goal, reward, float(done)))
                        state = next_state
                        tr_step += 1

                        if crash: is_crash = 1
                        if is_crash == 0 and done: tr_result = 1
                        if done or tr_step == self.args.max_step: break

                    tr_list.append(temp_buffer)
                    ep_step += tr_step
                    ep_result += tr_result

                ep_step /= 2
                ep_result /= 2

                ## update sr
                term_fingers, desired_goal = [tr_list[0][-1][1][9:12], tr_list[1][-1][1][9:12]], goal[0:3]
                vec_ab = term_fingers[0] - term_fingers[1]
                vec_ag = term_fingers[0] - desired_goal
                vec_bg = term_fingers[1] - desired_goal

                l_ab, l_ag, l_bg = np.sqrt(vec_ab.dot(vec_ab)), np.sqrt(vec_ag.dot(vec_ag)), np.sqrt(vec_bg.dot(vec_bg))
                add_both, add_idx, l_list = False, 0, [l_ag, l_bg]
                if l_ab < self.env.goal_r or np.min(l_list) < self.env.goal_r:
                    add_both = True
                else:
                    add_idx = l_list.index(np.max(l_list))

                for i in range(len(tr_list)):  # 0-1
                    anti_goal = term_fingers[(i + 1) % 2]

                    for j in range(len(tr_list[i])):  # 0-999
                        s, s_, a, g, r, d = tr_list[i][j]
                        finger = s_[9:12]

                        vec_f_g, vec_f_ag = finger - desired_goal, finger - anti_goal
                        l_f_g, l_f_ag = np.sqrt(vec_f_g.dot(vec_f_g)), np.sqrt(vec_f_ag.dot(vec_f_ag))
                        if l_f_g < self.env.goal_r:
                            r = 100
                        else:
                            r = np.min([0.0, (- l_f_g + l_f_ag) / 100])

                        if add_both or i == add_idx:
                            self.learner.replay_buffer.push((s, s_, a, g, r, d))

                ## update policy
                self.learner.update()

                ## print data
                print("Seed : {} \t\t Episode : {} \t\t Steps : {} \t\t Crash : {} \t\t Done : {}"
                      .format(self.seed, episode, ep_step, str(crash), str(done)))

                ## log data
                data_list = [ep_step, ep_result, 1]
                self.log_data(data_list, episode)

        else:
            for episode in range(self.args.max_episode):
                ## reset state and goal
                state = self.env.reset()
                if self.env.goal_dim == 6:
                    goal = np.concatenate([self.env.goal_pos, self.env.obs_pos])
                else:
                    goal = np.concatenate([self.env.goal_pos, self.env.obs1_pos, self.env.obs2_pos, self.env.obs3_pos])

                ep_step, is_crash, ep_result = 0, 0, 0
                temp_buffer = []
                ## sample episode
                while True:
                    action = self.learner.select_action(state, goal)
                    action = action + np.random.normal(0, self.args.exploration_noise)
                    next_state, reward, done, crash = self.env.step(action)

                    ## shaping reward
                    if self.args.shaper == 'mfrs' or self.args.shaper == 'dpba':
                        next_action = self.learner.select_action(next_state, goal)
                        shaping_reward = self.shaper.shaping(next_state, next_action, state, action, goal)
                    elif self.args.shaper == 'pbrs' or self.args.shaper == 'aim':
                        shaping_reward = self.shaper.shaping(next_state, state, goal)
                    else:
                        shaping_reward = 0

                    if self.args.shaper == 'aim':
                        reward = shaping_reward
                    else:
                        reward += shaping_reward
                    self.learner.replay_buffer.push((state, next_state, action, goal, reward, float(done)))
                    if self.args.shaper == 'aim':
                        self.shaper.replay_buffer.push((state, next_state, action, goal, reward, float(done)))
                    if self.args.shaper == 'her':
                        temp_buffer.append((state, next_state, action, goal, reward, float(done)))

                    state = next_state
                    ep_step += 1

                    if crash: is_crash = 1
                    if is_crash == 0 and done: ep_result = 1
                    if done or ep_step == self.args.max_step: break

                ## update mfrs
                if self.args.shaper == 'mfrs': self.shaper.update_norm()

                ## update her
                if self.args.shaper == 'her':
                    new_goal = np.concatenate([temp_buffer[-1][1][9:12], goal[3:]])
                    for i in range(len(temp_buffer)):
                        s, s_, a, g, r, d = temp_buffer[i]
                        finger, desired_goal = s_[9:12], new_goal[0:3]

                        vec_finger_goal = finger - desired_goal
                        l_finger_goal = np.sqrt(vec_finger_goal.dot(vec_finger_goal))
                        if l_finger_goal < self.env.goal_r:
                            r = 100
                        self.learner.replay_buffer.push((s, s_, a, new_goal, r, d))

                ## update policy
                self.learner.update()

                ## update aim
                if self.args.shaper == 'aim': self.shaper.update()

                ## print data
                print("Seed : {} \t\t Episode : {} \t\t Steps : {} \t\t Crash : {} \t\t Done : {}"
                      .format(self.seed, episode, ep_step, str(crash), str(done)))

                ## log data
                data_list = [ep_step, ep_result, 1]
                self.log_data(data_list, episode)


        print("=== Training end ===")
        ## save log
        self.mkdir(self.log_dir)
        np.save(self.log_dir + '/data.npy', self.avg_array)

        ## save model
        self.mkdir(self.model_dir)
        self.learner.save(self.model_dir)

        ## plot fig
        self.plotter.plot(['step', 'result'])
        print("Saving fig at : {}\n\n".format(self.fig_dir))
