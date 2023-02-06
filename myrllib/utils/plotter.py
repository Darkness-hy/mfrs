import matplotlib.pyplot as plt
from more_itertools import chunked
import scipy.stats as st
import seaborn as sns
import numpy as np
import os


legend_font = {'family' : 'Verdana',
'weight' : 'normal',
'size'   : 18}


class Plotter(object):
    def __init__(self, log_dir, fig_dir, log_freq):
        self.log_dir, self.fig_dir = str(log_dir), str(fig_dir)
        self.log_freq = int(log_freq)
        self.y_label = ['Steps to the goal', 'Success rate']

    @ staticmethod
    def mkdir(path):
        folder = os.path.exists(path)
        if not folder: os.makedirs(path)

    def plot(self, item_list):
        y_value = np.load(self.log_dir + '/data.npy')
        x_value = np.array(range(1, y_value.shape[1] + 1)) * self.log_freq
        sns.set_style('whitegrid')
        self.mkdir(self.fig_dir)

        for i in range(y_value.shape[0]):
            plt.figure(figsize=(7, 5))
            sns.lineplot(x=x_value, y=y_value[i])

            plt.xlabel("Learning episodes", size=15)
            plt.ylabel("{}".format(self.y_label[i]), size=15)
            # plt.ylim(-1000, 0)

            plt.savefig(self.fig_dir + '/{}.jpg'.format(item_list[i]))


class MultiPlotter(object):
    max_episode = 10000
    plot_num = 10

    def __init__(self):
        self.env_name = 'DobotReach-v1'
        self.seed_num = 5
        self.root_dir = './../..'
        self.fig_dir = self.root_dir + '/exp/'

        self.shaper_list = ['none', 'pbrs', 'dpba', 'her', 'sr', 'aim', 'mfrs']
        self.label_list = ['NS', 'PBRS', 'DPBA', 'HER', 'SR', 'AIM', 'MFRS']
        self.color_list = ['black', 'cyan', 'green', 'blue', 'orange', 'purple', 'red']
        self.marker_list = ['P', '^', 'o', 'd', 's', 'v', 'X']

        filled_markers = (
            'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

    @ staticmethod
    def mkdir(path):
        folder = os.path.exists(path)
        if not folder:  os.makedirs(path)

    def get_data(self, i):
        shaper_list = []  # (5, 5, 11)
        for shaper in self.shaper_list:
            seed_list = []  # (5, 11)
            for seed in range(self.seed_num):
                data_dir = self.root_dir + '/saves/{}/{}/seed_{}/log/data.npy'.format(self.env_name, shaper, seed)
                full_array = np.load(data_dir)[i]
                interval_list = [sum(x) / len(x) for x in chunked(full_array, int(len(full_array) / self.plot_num))]

                interval_list.insert(0, 0)
                seed_list.append(interval_list)


            shaper_list.append(seed_list)
        return np.array(shaper_list)

    def plot(self):
        # sns.set_style('white')
        plt.figure(figsize=(8, 6))

        plot_interval = int(self.max_episode / self.plot_num)
        x_value = list(range(0, (self.max_episode + plot_interval), plot_interval))  # 11
        x_value = np.array(x_value) / 1e4
        data = self.get_data(1)  # (5,5,11)

        for i in range(len(self.shaper_list)):  # 0-4
            y_value = np.mean(data[i], 0)
            # print(y_value)
            low_ci, high_ci = st.t.interval(0.90, len(x_value) - 1, loc=np.mean(data[i], 0), scale=st.sem(data[i]))
            low_ci[0], high_ci[0] = 0, 0

            # sns.lineplot(x=x_value, y=y_value, label=self.label_list[i], color=self.color_list[i],
            #              marker=self.marker_list[i], markersize=10)
            sns.lineplot(x=x_value, y=y_value, color=self.color_list[i],
                         marker=self.marker_list[i], markersize=10)
            plt.fill_between(x_value, low_ci, high_ci, alpha=0.15, color=self.color_list[i])

        plt.xlabel("Learning episodes (x1e4)", size=20)
        plt.ylabel("Success rate", size=20)
        plt.ylim((0, 1))

        bwith = 1.5
        plt.tick_params(labelsize=15, width=bwith, length=7)
        # plt.legend(prop=legend_font)
        plt.grid(which='major', axis='both')

        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)

        self.mkdir(self.fig_dir)
        plt.savefig(self.fig_dir + '{}_{}.png'.format(self.env_name, self.seed_num))
        # plt.show()

    def steps(self):
        data = self.get_data(0)  # (5,4,11)
        for i in range(data.shape[0]):  # 0-4
            list_seed = []
            for j in range(data.shape[1]):  # 0-3
                avg_steps = np.sum(data[i][j]) / (data.shape[2] - 1)
                # avg_steps = np.mean(data[i][j])
                list_seed.append(avg_steps)

            avg_method = np.mean(list_seed)
            std_err = np.std(list_seed) / np.sqrt(len(list_seed))
            # std_err = np.std(list_seed)
            print('{:.2f} \t\t {:.2f}'.format(avg_method, std_err))

    def employ(self):
        seed_list = []  # (5, 11)
        shaper_name = 'sr'

        for seed in range(self.seed_num):
            data_dir = self.root_dir + '/saves/{}/{}/seeds/seed_{}/log/data.npy'.format(self.env_name, shaper_name, seed)
            full_array = np.load(data_dir)[1]
            interval_list = [sum(x) / len(x) for x in chunked(full_array, int(len(full_array) / self.plot_num))]

            interval_list.insert(0, 0)
            seed_list.append(interval_list)

        plt.figure(figsize=(8, 6))
        plot_interval = int(self.max_episode / self.plot_num)
        x_value = list(range(0, (self.max_episode + plot_interval), plot_interval))  # 11
        data = np.array(seed_list)

        for i in range(len(data)):  # 0-4
            sns.lineplot(x=x_value, y=data[i], label=i)

        plt.xlabel("Learning episodes", size=20)
        plt.ylabel("Success rate", size=20)
        # plt.ylim((0, 1))

        bwith = 1.5
        plt.tick_params(labelsize=15, width=bwith, length=7)
        plt.legend(prop=legend_font)
        plt.grid(which='major', axis='both')

        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)

        self.mkdir(self.fig_dir)
        plt.savefig(self.fig_dir + '{}_{}_{}.png'.format(shaper_name, self.env_name, self.seed_num))
        # plt.show()



if __name__ == '__main__':
    MultiPlotter().plot()