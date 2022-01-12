from multiprocessing import cpu_count, Process, Queue
import matplotlib.pyplot as plt
import numpy as np


def man_dist_pbc(m, vector, shape=(10, 10)):
    dims = np.array(shape)
    delta = np.abs(m - vector)
    delta = np.where(delta > 0.5 * dims, np.abs(delta - dims), delta)
    return np.sum(delta, axis=len(m.shape) - 1)


class kohonen_nn(object):
    def __init__(self, x, y, alpha_start=0.6, seed=42, wta=False, count_side=True):
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.sigma = x / 2.
        self.alpha_start = alpha_start
        self.alphas = None
        self.sigmas = None
        self.epoch = 0
        self.interval = int()
        self.map = np.array([])
        self.indxmap = np.stack(np.unravel_index(np.arange(x * y, dtype=int).reshape(x, y), (x, y)), 2)
        self.winner_indices = np.array([])
        self.inizialized = False
        self.wta = wta
        self.count_side = count_side
        self.error = 0.
        self.history = list()

    def initialize(self, data):
        self.map = np.random.normal(np.mean(data), np.std(data), size=(self.x, self.y, len(data[0])))
        self.inizialized = True

    def winner(self, vector):
        indx = np.argmin(np.sum((self.map - vector) ** 2, axis=2))
        return np.array([int(indx / self.x), indx % self.y])

    def winner_rect(self, vector):
        distns = np.sum((self.map - vector) ** 2, axis=2)
        indx = np.argmin(distns)
        side_winners = []
        distns_updated =  np.delete(distns, indx)
        for i in range(4):
            temp_winner_indx = np.argmin(distns_updated)
            side_winners.append(np.array([int(temp_winner_indx / self.x), temp_winner_indx % self.y]))
            distns_updated =  np.delete(distns_updated, temp_winner_indx)
        return np.array([int(indx / self.x), indx % self.y]), side_winners

    def process_map(self, vector):
        w, side_w = self.winner_rect(vector)
        # print('Winner', side_w)
        # print('INDEX FOR WINNER:', indx)
        # print('Vector shape:', vector.shape)
        if self.wta:
            self.map[w[0]][w[1]] -= self.alphas[self.epoch] * (self.map[w[0]][w[1]] - vector)
            if self.count_side:
                for w_i in side_w:
                    self.map[w_i[0]][w_i[1]] -= (self.alphas[self.epoch]*0.5) * (self.map[w_i[0]][w_i[1]] - vector)
            print("Epoch %i;    Neuron [%i, %i];  alpha: %.4f" %
              (self.epoch, w[0], w[1], self.alphas[self.epoch]))
        else:
            dists = man_dist_pbc(self.indxmap, w, self.shape)
            h = np.exp(-(dists / self.sigmas[self.epoch]) ** 2).reshape(self.x, self.y, 1)
            self.map -= h * self.alphas[self.epoch] * (self.map - vector)
            print("Epoch %i;    Neuron [%i, %i];    \tSigma: %.4f;    alpha: %.4f" %
              (self.epoch, w[0], w[1], self.sigmas[self.epoch], self.alphas[self.epoch]))
            
        self.epoch = self.epoch + 1

    def fit(self, data, labels, epochs=0, save_e=False, interval=1000, decay='non_lin'):
        self.interval = interval
        if not self.inizialized:
            self.initialize(data)
        if not epochs:
            epochs = len(data)
            indx = np.random.choice(np.arange(len(data)), epochs, replace=False)
        else:
            indx = np.random.choice(np.arange(len(data)), epochs)

        if decay == 'non_lin':
            epoch_list = np.linspace(0, 1, epochs)
            self.alphas = self.alpha_start / (1 + (epoch_list / 0.5) ** 4)
            self.sigmas = self.sigma / (1 + (epoch_list / 0.5) ** 4)
        else:
            self.alphas = np.linspace(self.alpha_start, 0.05, epochs)
            self.sigmas = np.linspace(self.sigma, 1, epochs)

        if save_e:
            for i in range(epochs):
                self.process_map(data[indx[i]])
                if i % interval == 0:
                    self.history.append(self.kohonen_error(data))
        else:
            for i in range(epochs):
                self.process_map(data[indx[i]])
        self.error = self.kohonen_error(data)


    def winner_map(self, data):
        wm = np.zeros(self.shape, dtype=int)
        for d in data:
            [x, y] = self.winner(d)
            wm[x, y] += 1
        return wm

    def _one_error(self, data, q):
        errs = list()
        for d in data:
            w = self.winner(d)
            dist = self.map[w[0], w[1]] - d
            errs.append(np.sqrt(np.dot(dist, dist.T)))
        q.put(errs)

    def kohonen_error(self, data):
        queue = Queue()
        for d in np.array_split(np.array(data), cpu_count()):
            p = Process(target=self._one_error, args=(d, queue,))
            p.start()
        rslt = []
        for _ in range(cpu_count()):
            rslt.extend(queue.get(50))
        return float(sum(rslt) / float(len(data)))

    def plot_class_density(self, data, targets, t=1, name='actives', colormap='Oranges', example_dict=None,
                           filename=None):
        targets = np.array(targets)
        t_data = data[np.where(targets == t)[0]]
        wm = self.winner_map(t_data)
        fig, ax = plt.subplots(figsize=self.shape)
        plt.pcolormesh(wm, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        plt.title(name, fontweight='bold', fontsize=28)
        ax.set_aspect('equal')
        plt.text(0.1, -1., "%i Datapoints" % len(t_data), fontsize=20, fontweight='bold')

        if example_dict:
            for k, v in example_dict.items():
                w = self.winner(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


    def plot_error_history(self, color='orange', filename=None):
        if not len(self.history):
            raise LookupError("No error history was found! Is the kohonen_nn already trained?")
        fig, ax = plt.subplots()
        ax.plot(range(0, self.epoch, self.interval), self.history, '-o', c=color)
        ax.set_title('kohonen_nn Error History', fontweight='bold')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Error', fontweight='bold')
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
