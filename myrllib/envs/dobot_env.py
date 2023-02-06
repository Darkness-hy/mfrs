import numpy as np


arm_len_1 = [0.13, 0.135081689824231, 0.147034357777703, 0.0597656422222972, 0.0741816898242307]
arm_len_2 = [0.03747, 0.07499, 0.03007, 0.11799]  # [X11, X12, X21, X22]
goal_pos_max = [0.341368, 0.341368, 0.08]  # [x, y, z]
goal_a0_range = [0.127083, 0.351142]


class DobotEnv_V1(object):
    """
        1 Goal: dynamic in different episodes
        1 Obstacle: static in all episodes
    """
    def __init__(self):
        self.state_dim, self.action_dim, self.goal_dim = 12, 3, 6
        self.action_bound = [-1, 1]
        self.RAD2DEG = 180 / np.pi

        ## robot arm
        self.arm_l, self.arm_r = arm_len_1[:3], np.zeros(3)
        self.a0_pos = np.array([0., 0., arm_len_1[0]])

        ## target
        self.goal_pos, self.goal_r = np.zeros(3), 0.02

        ## obstacle
        self.obs_a, self.obs_b, self.obs_h = 0.1, 0.4, 0.05
        self.fixed_pos = np.array([0., -0.5, 0.105])
        self.obs_r, self.obs_pos, self.obs_mag_pos = 0., np.zeros(3), np.zeros(3)
        self.set_obs_pos()

    def new_arm_r(self, arm_r, action):
        arm_r += np.clip(action, *self.action_bound)
        arm_r[0] = np.clip(arm_r[0], -90, 90)
        arm_r[1] = np.clip(arm_r[1], 0, 85)
        arm_r[2] = np.clip(arm_r[2], -10, 75)

        return arm_r

    def get_arm_pos(self, arm_r):
        a0_l, a1_l, a2_l = self.arm_l
        a0_r, a1_r, a2_r = arm_r / self.RAD2DEG
        X11, X12, X21, X22 = arm_len_2

        a0_pos = np.array([0., 0., a0_l])
        a1_pos = np.array([np.sin(a1_r) * np.sin(a0_r), -np.sin(a1_r) * np.cos(a0_r), np.cos(a1_r)]) * a1_l + a0_pos
        a2_pos = np.array([np.cos(a2_r) * np.sin(a0_r), -np.cos(a2_r) * np.cos(a0_r), -np.sin(a2_r)]) * a2_l + a1_pos
        finger = np.array([np.sin(a0_r) * arm_len_1[3], -np.cos(a0_r) * arm_len_1[3], -arm_len_1[4]]) + a2_pos

        front_head = np.array([np.sin(a0_r) * X11, -np.cos(a0_r) * X11, X12]) + finger
        top_head = np.array([np.sin(a0_r) * X21, -np.cos(a0_r) * X21, X22]) + finger
        down_head = np.array([0, 0, 0.0375]) + finger
        point_list = [finger, front_head, top_head, down_head]

        return a1_pos, finger, point_list

    def get_obs_pos(self):
        obs_pos = np.array([-np.sin(self.obs_r), np.cos(self.obs_r), 0]) * \
                  (self.obs_b - self.obs_a) / 2 + self.fixed_pos

        obs_edge = np.array([np.sin(self.obs_r), -np.cos(self.obs_r), 0]) * self.obs_a / 2 + self.fixed_pos
        obs_mag_pos = np.array([-np.cos(self.obs_r) * self.obs_a / 2,
                                -np.sin(self.obs_r) * self.obs_a / 2,
                                -self.obs_h / 2]) + obs_edge

        return obs_pos, obs_mag_pos

    def axes_transfer(self, from_pos, to_pos):
        mat_trans = np.array([[np.cos(-self.obs_r), -np.sin(-self.obs_r), 0],
                              [np.sin(-self.obs_r), np.cos(-self.obs_r), 0],
                              [0, 0, 1]])

        trans_pos = np.dot(mat_trans, from_pos - to_pos)
        return trans_pos

    def step(self, action):
        self.arm_r = self.new_arm_r(self.arm_r, action)
        a1_pos, finger, point_list = self.get_arm_pos(self.arm_r)

        ## native reward
        done, crash = False, False
        r = -1

        vec_finger_goal = finger - self.goal_pos
        if np.sqrt(vec_finger_goal.dot(vec_finger_goal)) < self.goal_r:
            done = True
            r = 100

        if finger[2] < 0:
            crash = True
            r = -10

        for point in point_list:
            trans_pos = self.axes_transfer(point, self.obs_pos)
            if (np.fabs(trans_pos[0]) < self.obs_a / 2 and
                    np.fabs(trans_pos[1]) < self.obs_b / 2 and
                    np.fabs(trans_pos[2]) < self.obs_h / 2):
                crash = True
                r = -10
                break

        theta = self.arm_r / self.RAD2DEG
        s_ = np.concatenate([np.cos(theta), np.sin(theta), a1_pos, finger])
        return s_, r, done, crash

    def set_goal_pos(self):
        while True:
            flag = 1 if np.random.rand() < 0.5 else -1
            self.goal_pos[0] = np.random.uniform(0.1, goal_pos_max[0]) * flag
            self.goal_pos[1] = np.random.uniform(0.1, goal_pos_max[1]) * (-1)
            self.goal_pos[2] = np.random.uniform(0, goal_pos_max[2])

            vec_goal_a0 = self.goal_pos - self.a0_pos
            trans_goal_pos = self.axes_transfer(self.goal_pos, self.obs_pos)

            if goal_a0_range[0] < np.sqrt(vec_goal_a0.dot(vec_goal_a0)) < goal_a0_range[1]:
                if not(np.fabs(trans_goal_pos[0]) < self.obs_a / 2 + self.goal_r and
                       np.fabs(trans_goal_pos[1]) < self.obs_b / 2 + self.goal_r):
                    break

    def set_obs_pos(self):
        self.obs_r = np.random.uniform(-60, 60) / self.RAD2DEG
        self.obs_pos, self.obs_mag_pos = self.get_obs_pos()

    def reset(self):
        self.set_goal_pos()
        self.arm_r = np.zeros(3)
        a1_pos, finger, _ = self.get_arm_pos(self.arm_r)

        theta = self.arm_r / self.RAD2DEG
        s = np.concatenate([np.cos(theta), np.sin(theta), a1_pos, finger])
        return s


class DobotEnv_V2(object):
    """
        1 Goal: dynamic in different episodes
        1 Obstacle: dynamic in different EPISODES
    """
    def __init__(self):
        self.state_dim, self.action_dim, self.goal_dim = 12, 3, 6
        self.action_bound = [-1, 1]
        self.RAD2DEG = 180 / np.pi

        ## robot arm
        self.arm_l, self.arm_r = arm_len_1[:3], np.zeros(3)
        self.a0_pos = np.array([0., 0., arm_len_1[0]])

        ## target
        self.goal_pos, self.goal_r = np.zeros(3), 0.02

        ## obstacle
        self.obs_a, self.obs_b, self.obs_h = 0.1, 0.4, 0.05
        self.fixed_pos = np.array([0., -0.5, 0.105])
        self.obs_r = 0.
        self.obs_pos, self.obs_mag_pos = self.get_obs_pos()

    def new_arm_r(self, arm_r, action):
        arm_r += np.clip(action, *self.action_bound)
        arm_r[0] = np.clip(arm_r[0], -90, 90)
        arm_r[1] = np.clip(arm_r[1], 0, 85)
        arm_r[2] = np.clip(arm_r[2], -10, 75)

        return arm_r

    def get_arm_pos(self, arm_r):
        a0_l, a1_l, a2_l = self.arm_l
        a0_r, a1_r, a2_r = arm_r / self.RAD2DEG
        X11, X12, X21, X22 = arm_len_2

        a0_pos = np.array([0., 0., a0_l])
        a1_pos = np.array(
            [np.sin(a1_r) * np.sin(a0_r), -np.sin(a1_r) * np.cos(a0_r), np.cos(a1_r)]) * a1_l + a0_pos
        a2_pos = np.array(
            [np.cos(a2_r) * np.sin(a0_r), -np.cos(a2_r) * np.cos(a0_r), -np.sin(a2_r)]) * a2_l + a1_pos
        finger = np.array([np.sin(a0_r) * arm_len_1[3], -np.cos(a0_r) * arm_len_1[3], -arm_len_1[4]]) + a2_pos

        front_head = np.array([np.sin(a0_r) * X11, -np.cos(a0_r) * X11, X12]) + finger
        top_head = np.array([np.sin(a0_r) * X21, -np.cos(a0_r) * X21, X22]) + finger
        down_head = np.array([0, 0, 0.0375]) + finger
        point_list = [finger, front_head, top_head, down_head]

        return a1_pos, finger, point_list

    def get_obs_pos(self):
        obs_pos = np.array([-np.sin(self.obs_r),
                            np.cos(self.obs_r),
                            0]) * (self.obs_b - self.obs_a) / 2 + self.fixed_pos

        obs_edge = np.array([np.sin(self.obs_r), -np.cos(self.obs_r), 0]) * self.obs_a / 2 + self.fixed_pos
        obs_mag_pos = np.array([-np.cos(self.obs_r) * self.obs_a / 2,
                                -np.sin(self.obs_r) * self.obs_a / 2,
                                -self.obs_h / 2]) + obs_edge

        return obs_pos, obs_mag_pos

    def axes_transfer(self, from_pos, to_pos):
        mat_trans = np.array([[np.cos(-self.obs_r), -np.sin(-self.obs_r), 0],
                              [np.sin(-self.obs_r), np.cos(-self.obs_r), 0],
                              [0, 0, 1]])

        trans_pos = np.dot(mat_trans, from_pos - to_pos)
        return trans_pos

    def step(self, action):
        self.arm_r = self.new_arm_r(self.arm_r, action)
        a1_pos, finger, point_list = self.get_arm_pos(self.arm_r)

        ## native reward
        done, crash = False, False
        r = -1

        vec_finger_goal = finger - self.goal_pos
        if np.sqrt(vec_finger_goal.dot(vec_finger_goal)) < self.goal_r:
            done = True
            r = 100

        if finger[2] < 0:
            crash = True
            r = -10

        for point in point_list:
            trans_pos = self.axes_transfer(point, self.obs_pos)
            if (np.fabs(trans_pos[0]) < self.obs_a / 2 and
                    np.fabs(trans_pos[1]) < self.obs_b / 2 and
                    np.fabs(trans_pos[2]) < self.obs_h / 2):
                crash = True
                r = -10
                break

        theta = self.arm_r / self.RAD2DEG
        s_ = np.concatenate([np.cos(theta), np.sin(theta), a1_pos, finger])
        return s_, r, done, crash

    def set_goal_pos(self):
        while 1:
            flag = 1 if np.random.rand() < 0.5 else -1
            self.goal_pos[0] = np.random.uniform(0.1, goal_pos_max[0]) * flag
            self.goal_pos[1] = np.random.uniform(0.1, goal_pos_max[1]) * (-1)
            self.goal_pos[2] = np.random.uniform(0, goal_pos_max[2])

            vec_goal_a0 = self.goal_pos - self.a0_pos
            trans_goal_pos = self.axes_transfer(self.goal_pos, self.obs_pos)

            if goal_a0_range[0] < np.sqrt(vec_goal_a0.dot(vec_goal_a0)) < goal_a0_range[1]:
                if not (np.fabs(trans_goal_pos[0]) < self.obs_a / 2 + self.goal_r and
                        np.fabs(trans_goal_pos[1]) < self.obs_b / 2 + self.goal_r):
                    break

    def set_obs_pos(self):
        self.obs_r = np.random.uniform(-60, 60) / self.RAD2DEG
        self.obs_pos, self.obs_mag_pos = self.get_obs_pos()

    def reset(self):
        self.set_obs_pos()
        self.set_goal_pos()
        self.arm_r = np.zeros(3)
        a1_pos, finger, _ = self.get_arm_pos(self.arm_r)

        theta = self.arm_r / self.RAD2DEG
        s = np.concatenate([np.cos(theta), np.sin(theta), a1_pos, finger])
        return s


class DobotEnv_V3(object):
    """
        1 Goal: static in different episodes
        3 Obstacle: dynamic in different episodes
    """
    def __init__(self):
        self.state_dim, self.action_dim, self.goal_dim = 12, 3, 12
        self.action_bound = [-1, 1]
        self.RAD2DEG = 180 / np.pi

        ## robot arm
        self.arm_l, self.arm_r = arm_len_1[:3], np.zeros(3)
        self.a0_pos = np.array([0., 0., arm_len_1[0]])

        ## 1 goal
        self.goal_pos, self.goal_r = np.zeros(3), 0.02
        self.goal_pos = self.generate_point()

        ## 3 obstacles
        self.obs1_pos, self.obs1_r = np.zeros(2), 0.02
        self.obs2_pos, self.obs2_r = np.zeros(2), 0.04
        self.obs3_pos, self.obs3_r = np.zeros(2), 0.06

    def new_arm_r(self, arm_r, action):
        arm_r += np.clip(action, *self.action_bound)
        arm_r[0] = np.clip(arm_r[0], -90, 90)
        arm_r[1] = np.clip(arm_r[1], 0, 85)
        arm_r[2] = np.clip(arm_r[2], -10, 75)

        return arm_r

    def get_arm_pos(self, arm_r):
        a0_l, a1_l, a2_l = self.arm_l
        a0_r, a1_r, a2_r = arm_r / self.RAD2DEG
        X11, X12, X21, X22 = arm_len_2

        a0_pos = np.array([0., 0., a0_l])
        a1_pos = np.array(
            [np.sin(a1_r) * np.sin(a0_r), -np.sin(a1_r) * np.cos(a0_r), np.cos(a1_r)]) * a1_l + a0_pos
        a2_pos = np.array(
            [np.cos(a2_r) * np.sin(a0_r), -np.cos(a2_r) * np.cos(a0_r), -np.sin(a2_r)]) * a2_l + a1_pos
        finger = np.array([np.sin(a0_r) * arm_len_1[3], -np.cos(a0_r) * arm_len_1[3], -arm_len_1[4]]) + a2_pos

        front_head = np.array([np.sin(a0_r) * X11, -np.cos(a0_r) * X11, X12]) + finger
        top_head = np.array([np.sin(a0_r) * X21, -np.cos(a0_r) * X21, X22]) + finger
        down_head = np.array([0, 0, 0.0375]) + finger
        point_list = [finger, front_head, top_head, down_head]

        return a1_pos, finger, point_list

    def step(self, action):
        self.arm_r = self.new_arm_r(self.arm_r, action)
        a1_pos, finger, point_list = self.get_arm_pos(self.arm_r)

        ## native reward
        done, crash = False, False
        r = -1

        vec_finger_goal = finger - self.goal_pos
        if np.sqrt(vec_finger_goal.dot(vec_finger_goal)) < self.goal_r:
            done = True
            r = 100

        if finger[2] < 0:
            crash = True
            r = -10

        for point in point_list:
            vec_obs1 = self.obs1_pos - point
            l_obs1 = np.sqrt(vec_obs1.dot(vec_obs1))
            vec_obs2 = self.obs2_pos - point
            l_obs2 = np.sqrt(vec_obs2.dot(vec_obs2))
            vec_obs3 = self.obs3_pos - point
            l_obs3 = np.sqrt(vec_obs3.dot(vec_obs3))

            if l_obs1 < self.obs1_r or l_obs2 < self.obs2_r or l_obs3 < self.obs3_r:
                crash = True
                r = -10
                break

        theta = self.arm_r / self.RAD2DEG
        s_ = np.concatenate([np.cos(theta), np.sin(theta), a1_pos, finger])
        return s_, r, done, crash

    def generate_point(self):
        while True:
            flag = 1 if np.random.rand() < 0.5 else -1
            x = np.random.uniform(0.1, goal_pos_max[0]) * flag
            y = np.random.uniform(0.1, goal_pos_max[1]) * (-1)
            z = np.random.uniform(0, goal_pos_max[2])
            point = np.array([x, y, z])

            vec_point_a0 = point - self.a0_pos
            if goal_a0_range[0] < np.sqrt(vec_point_a0.dot(vec_point_a0)) < goal_a0_range[1]: break

        return point

    def set_obs_pos(self):
        ## settle obs3
        while True:
            self.obs3_pos = self.generate_point()
            vec_obs3_goal = self.obs3_pos - self.goal_pos
            if np.sqrt(vec_obs3_goal.dot(vec_obs3_goal)) > self.obs3_r + self.goal_r: break

        ## settle obs2
        while True:
            self.obs2_pos = self.generate_point()
            vec_obs2_goal = self.obs2_pos - self.goal_pos
            vec_obs2_obs3 = self.obs2_pos - self.obs3_pos
            if np.sqrt(vec_obs2_goal.dot(vec_obs2_goal)) > self.obs2_r + self.goal_r and \
                    np.sqrt(vec_obs2_obs3.dot(vec_obs2_obs3)) > self.obs2_r + self.obs3_r: break

        ## settle obs1
        while True:
            self.obs1_pos = self.generate_point()
            vec_obs1_goal = self.obs1_pos - self.goal_pos
            vec_obs1_obs3 = self.obs1_pos - self.obs3_pos
            vec_obs1_obs2 = self.obs1_pos - self.obs2_pos
            if np.sqrt(vec_obs1_goal.dot(vec_obs1_goal)) > self.obs1_r + self.goal_r and \
                    np.sqrt(vec_obs1_obs3.dot(vec_obs1_obs3)) > self.obs1_r + self.obs3_r and \
                    np.sqrt(vec_obs1_obs2.dot(vec_obs1_obs2)) > self.obs1_r + self.obs2_r: break

    def reset(self):
        self.set_obs_pos()
        self.arm_r = np.zeros(3)
        a1_pos, finger, _ = self.get_arm_pos(self.arm_r)

        theta = self.arm_r / self.RAD2DEG
        s = np.concatenate([np.cos(theta), np.sin(theta), a1_pos, finger])
        return s


class DobotEnv_V4(object):
    """
        1 Goal: dynamic in different episodes
        3 Obstacle: dynamic in different EPISODES
    """
    def __init__(self):
        self.state_dim, self.action_dim, self.goal_dim = 12, 3, 12
        self.action_bound = [-1, 1]
        self.RAD2DEG = 180 / np.pi

        ## robot arm
        self.arm_l, self.arm_r = arm_len_1[:3], np.zeros(3)
        self.a0_pos = np.array([0., 0., arm_len_1[0]])

        ## 1 goal
        self.goal_pos, self.goal_r = np.zeros(3), 0.02

        ## obstacle
        self.obs1_pos, self.obs1_r = np.zeros(2), 0.02
        self.obs2_pos, self.obs2_r = np.zeros(2), 0.04
        self.obs3_pos, self.obs3_r = np.zeros(2), 0.06

    def new_arm_r(self, arm_r, action):
        arm_r += np.clip(action, *self.action_bound)
        arm_r[0] = np.clip(arm_r[0], -90, 90)
        arm_r[1] = np.clip(arm_r[1], 0, 85)
        arm_r[2] = np.clip(arm_r[2], -10, 75)

        return arm_r

    def get_arm_pos(self, arm_r):
        a0_l, a1_l, a2_l = self.arm_l
        a0_r, a1_r, a2_r = arm_r / self.RAD2DEG
        X11, X12, X21, X22 = arm_len_2

        a0_pos = np.array([0., 0., a0_l])
        a1_pos = np.array(
            [np.sin(a1_r) * np.sin(a0_r), -np.sin(a1_r) * np.cos(a0_r), np.cos(a1_r)]) * a1_l + a0_pos
        a2_pos = np.array(
            [np.cos(a2_r) * np.sin(a0_r), -np.cos(a2_r) * np.cos(a0_r), -np.sin(a2_r)]) * a2_l + a1_pos
        finger = np.array([np.sin(a0_r) * arm_len_1[3], -np.cos(a0_r) * arm_len_1[3], -arm_len_1[4]]) + a2_pos

        front_head = np.array([np.sin(a0_r) * X11, -np.cos(a0_r) * X11, X12]) + finger
        top_head = np.array([np.sin(a0_r) * X21, -np.cos(a0_r) * X21, X22]) + finger
        down_head = np.array([0, 0, 0.0375]) + finger
        point_list = [finger, front_head, top_head, down_head]

        return a1_pos, finger, point_list

    def step(self, action):
        self.arm_r = self.new_arm_r(self.arm_r, action)
        a1_pos, finger, point_list = self.get_arm_pos(self.arm_r)

        ## native reward
        done, crash = False, False
        r = -1

        vec_finger_goal = finger - self.goal_pos
        if np.sqrt(vec_finger_goal.dot(vec_finger_goal)) < self.goal_r:
            done = True
            r = 100

        if finger[2] < 0:
            crash = True
            r = -10

        for point in point_list:
            vec_obs1 = self.obs1_pos - point
            l_obs1 = np.sqrt(vec_obs1.dot(vec_obs1))
            vec_obs2 = self.obs2_pos - point
            l_obs2 = np.sqrt(vec_obs2.dot(vec_obs2))
            vec_obs3 = self.obs3_pos - point
            l_obs3 = np.sqrt(vec_obs3.dot(vec_obs3))

            if l_obs1 < self.obs1_r or l_obs2 < self.obs2_r or l_obs3 < self.obs3_r:
                crash = True
                r = -10
                break

        theta = self.arm_r / self.RAD2DEG
        s_ = np.concatenate([np.cos(theta), np.sin(theta), a1_pos, finger])
        return s_, r, done, crash

    def generate_point(self):
        while True:
            flag = 1 if np.random.rand() < 0.5 else -1
            x = np.random.uniform(0.1, goal_pos_max[0]) * flag
            y = np.random.uniform(0.1, goal_pos_max[1]) * (-1)
            z = np.random.uniform(0, goal_pos_max[2])
            point = np.array([x, y, z])

            vec_point_a0 = point - self.a0_pos
            if goal_a0_range[0] < np.sqrt(vec_point_a0.dot(vec_point_a0)) < goal_a0_range[1]: break

        return point

    def set_obs_pos(self):
        self.obs3_pos = self.generate_point()

        while True:
            self.obs2_pos = self.generate_point()
            vec_23 = self.obs2_pos - self.obs3_pos
            if np.sqrt(vec_23.dot(vec_23)) > self.obs2_r + self.obs3_r: break

        while True:
            self.obs1_pos = self.generate_point()
            vec_12 = self.obs1_pos - self.obs2_pos
            vec_13 = self.obs1_pos - self.obs3_pos
            if np.sqrt(vec_12.dot(vec_12)) > self.obs1_r + self.obs2_r and \
                    np.sqrt(vec_13.dot(vec_13)) > self.obs1_r + self.obs3_r: break

    def set_goal_pos(self):
        while True:
            self.goal_pos = self.generate_point()
            vec_1 = self.goal_pos - self.obs1_pos
            vec_2 = self.goal_pos - self.obs2_pos
            vec_3 = self.goal_pos - self.obs3_pos

            if np.sqrt(vec_1.dot(vec_1)) > self.goal_r + self.obs1_r and \
                    np.sqrt(vec_2.dot(vec_2)) > self.goal_r + self.obs2_r and \
                    np.sqrt(vec_3.dot(vec_3)) > self.goal_r + self.obs3_r: break

    def reset(self):
        self.set_obs_pos()
        self.set_goal_pos()
        self.arm_r = np.zeros(3)
        a1_pos, finger, _ = self.get_arm_pos(self.arm_r)

        theta = self.arm_r / self.RAD2DEG
        s = np.concatenate([np.cos(theta), np.sin(theta), a1_pos, finger])
        return s