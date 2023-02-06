import numpy as np


class PBRS(object):
    def __init__(self, args, env):
        self.args = args
        self.env = env

    def potential(self, state, goal):
        finger, goal_pos = state[9:12], goal[0:3]
        vec_finger_goal = finger - goal_pos
        l_finger_goal = np.sqrt(vec_finger_goal.dot(vec_finger_goal))

        if self.env.goal_dim == 6:
            obs_pos = goal[3:6]
            vec_finger_obs = finger - obs_pos

            l_finger_obs = np.sqrt(vec_finger_obs.dot(vec_finger_obs))
            p_value = - l_finger_goal + l_finger_obs

        else:
            obs1_pos, obs2_pos, obs3_pos = goal[3:6], goal[6:9], goal[9:12]
            vec_finger_obs1 = finger - obs1_pos
            vec_finger_obs2 = finger - obs2_pos
            vec_finger_obs3 = finger - obs3_pos

            l_finger_obs1 = np.sqrt(vec_finger_obs1.dot(vec_finger_obs1))
            l_finger_obs2 = np.sqrt(vec_finger_obs2.dot(vec_finger_obs2))
            l_finger_obs3 = np.sqrt(vec_finger_obs3.dot(vec_finger_obs3))
            p_value = - l_finger_goal + (l_finger_obs1 + l_finger_obs2 + l_finger_obs3) / 3

        return p_value

    def shaping(self, next_state, state, goal):
        shaping_reward = self.args.gamma * self.potential(next_state, goal) - self.potential(state, goal)

        return shaping_reward
