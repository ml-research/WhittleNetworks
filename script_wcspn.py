"""
@author: Xiaoting Shao
Modified on July 23, 2021
@author: Zhongjie Yu

Please note that the (W)CSPN code is still under development
Details of the CSPN implementation please refer to:
Shao, X., Molina, A., Vergari, A., Stelzner, K., Peharz, R., Liebig, T., & Kersting, K. 
(2020, February). Conditional sum-product networks: Imposing structure on deep 
probabilistic architectures. In International Conference on PGM (pp. 401-412). PMLR.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation
from sklearn.metrics import mean_squared_error
import time
import os
import csv
import matplotlib.pyplot as plt

# for neural CSPN
import sys
sys.path.append('./WCSPN/')
import RAT_SPN
from relu_mlp import ReluMLP
import region_graph
import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False
np.set_printoptions(precision=3)
np.random.seed(2021)
tf.set_random_seed(2021)


def dump_attributes(obj, filename):
    w = csv.writer(open(filename, 'w'))
    for key, val in vars(obj).items():
        w.writerow([str(key), str(val)])


class Config:
    def __init__(self):
        self.num_epochs = 10
        self.batch_size = 64
        self.ckpt_dir_cspn = './WCSPN_results/checkpoints_cspn/cspn'
        self.ckpt_dir_wcspn = './WCSPN_results/checkpoints_wcspn/wcspn'


class CspnTrainer:
    def __init__(self, spn, data, x_ph, train_ph, conf, sess=tf.Session()):
        self.spn, self.data, self.x_ph = spn, data, x_ph
        self.conf, self.sess = conf, sess

        self.y_ph = tf.placeholder(tf.float32,
                                   [conf.batch_size] + list(data.train_y.shape[1:]),
                                   name="y_ph")
        self.train_ph = train_ph
        spn_input = tf.reshape(self.y_ph, [conf.batch_size, -1])
        self.marginalized = tf.placeholder(tf.float32, spn_input.shape, name="marg_ph")
        self.spn_output = spn.forward(spn_input, self.marginalized)
        self.loss = -1 * tf.reduce_mean(tf.reduce_logsumexp(self.spn_output, axis=1))
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()
        if os.path.exists(conf.ckpt_dir_cspn):
            self.saver.restore(self.sess, conf.ckpt_dir_cspn)
            print('Loaded parameters')
        else:
            sess.run(tf.global_variables_initializer())
            print('Initialized parameters')

        i = 0
        log_path = 'WCSPN_results/run0'
        while os.path.exists(log_path):
            log_path = 'WCSPN_results/run{}'.format(i)
            i += 1
        os.makedirs(log_path)
        dump_attributes(conf, log_path + '/conf.csv')
        self.log_path = log_path
        self.log_file = open(log_path + '/results.csv', 'a')

    def run_training(self):
        batch_size = self.conf.batch_size
        batches_per_epoch = self.data.train_y.shape[0] // batch_size
        for i in range(self.conf.num_epochs):
            for j in range(batches_per_epoch):
                x_batch = self.data.train_x[j * batch_size : (j + 1) * batch_size, :]
                y_batch = self.data.train_y[j * batch_size : (j + 1) * batch_size, :]
                feed_dict = {self.x_ph: x_batch,
                             self.y_ph: y_batch,
                             self.marginalized: np.zeros(self.marginalized.shape),
                             self.train_ph: True}

                _, cur_output, cur_loss = self.sess.run(
                    [self.train_op, self.spn_output, self.loss], feed_dict=feed_dict)

                if j % 1000 == 0:
                    print('ep. {}, batch {}, train ll {:.2f}'.format(i, j, -cur_loss))
            if i % 2 == 1:
                self.saver.save(self.sess, self.conf.ckpt_dir_cspn)
                print('Parameters saved')


class WspnTrainer:
    def __init__(self, spn, data, x_ph, train_ph, conf, sess=tf.Session()):
        self.spn, self.data, self.x_ph = spn, data, x_ph
        self.conf, self.sess = conf, sess

        self.y_ph = tf.placeholder(tf.float32,
                                   [conf.batch_size] + list(data.train_y.shape[1:]),
                                   name="y_ph")
        self.train_ph = train_ph
        # rfft to get real valued coefficients
        y_ph_rfft = tf.signal.rfft(tf.reshape(self.y_ph, [64, 2, 32]))
        y_ph_rfft = tf.reshape(y_ph_rfft, [64, 2, -1, 1])
        y_ph_rfft_real = tf.real(y_ph_rfft)
        y_ph_rfft_imag = tf.imag(y_ph_rfft)
        self.y_ph_rfft = tf.concat([y_ph_rfft_real, y_ph_rfft_imag], 2)
        spn_input = tf.reshape(self.y_ph_rfft, [conf.batch_size, -1])
        self.marginalized = tf.placeholder(tf.float32, spn_input.shape, name="marg_ph")
        self.spn_output = spn.forward(spn_input, self.marginalized)
        self.loss = -1 * tf.reduce_mean(tf.reduce_logsumexp(self.spn_output, axis=1))
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()
        if os.path.exists(conf.ckpt_dir_wcspn):
            self.saver.restore(self.sess, conf.ckpt_dir_wcspn)
            print('Loaded parameters')
        else:
            sess.run(tf.global_variables_initializer())
            print('Initialized parameters')

        i = 0
        log_path = 'WCSPN_results/run0'
        while os.path.exists(log_path):
            log_path = 'WCSPN_results/run{}'.format(i)
            i += 1
        os.makedirs(log_path)
        dump_attributes(conf, log_path + '/conf.csv')
        self.log_path = log_path
        self.log_file = open(log_path + '/results.csv', 'a')

    def run_training(self):
        batch_size = self.conf.batch_size
        batches_per_epoch = self.data.train_y.shape[0] // batch_size
        for i in range(self.conf.num_epochs):
            for j in range(batches_per_epoch):
                x_batch = self.data.train_x[j * batch_size : (j + 1) * batch_size, :]
                y_batch = self.data.train_y[j * batch_size : (j + 1) * batch_size, :]
                feed_dict = {self.x_ph: x_batch,
                             self.y_ph: y_batch,
                             self.marginalized: np.zeros(self.marginalized.shape),
                             self.train_ph: True}

                _, cur_output, cur_loss = self.sess.run(
                    [self.train_op, self.spn_output, self.loss], feed_dict=feed_dict)

                if j % 1000 == 0:
                    print('ep. {}, batch {}, train ll {:.2f}'.format(i, j, -cur_loss))
            if i % 2 == 1:
                self.saver.save(self.sess, self.conf.ckpt_dir_wcspn)
                print('Parameters saved')


def generate_mackey(batch_size=100, tmax=200, delta_t=1, rnd=True):
    """
    Function to generate a 1d Mackey-Glass series
    details see:
    https://github.com/v0lta/Spectral-RNN/blob/master/src/mackey_glass_generator.py
    """
    steps = int(tmax/delta_t) + 100

    def mackey(x, tau, gamma=0.1, beta=0.2, n=10):
        return beta*x[:, -tau]/(1 + np.power(x[:, -tau], n)) - gamma*x[:, -1]

    tau = int(17*(1/delta_t))
    x0 = np.ones([tau])
    x0 = np.stack(batch_size*[x0], axis=0)
    if rnd:
        print('Mackey initial state is random.')
        x0 += np.random.uniform(-0.1, 0.1, x0.shape)
    else:
        np.random.seed(0)
        x0 += np.random.uniform(-0.1, 0.1, x0.shape)

    x = x0
    for _ in range(steps):
        res = np.expand_dims(x[:, -1] + delta_t*mackey(x, tau), -1)
        x = np.concatenate([x, res], -1)

    discard = 100 + tau

    return x[:, discard:]


class MackeyGlassDataset:
    # MG data for continuous prediction
    def __init__(self, noise=False):
        L = 1024 # length
        l_window = 32 # window size for FFT
        N = 3000 # number of samples
        n_train=512 # number of training data
        mackey1 = generate_mackey(batch_size=N, tmax=L, delta_t=1, rnd=False)
        mackey2 = generate_mackey(batch_size=N, tmax=L*3, delta_t=3, rnd=False)
        x1 = mackey1.reshape(-1, 1, L, 1)
        x2 = mackey2.reshape(-1, 1, L, 1)
        train = np.concatenate([x1, x2], axis=1)
        # with overlap, training forms 32+32 with overlaps
        # while test is a complete TS
        self.train_x = train[:, :, :l_window] # shape of (N, 2, 32)
        self.train_y = train[:, :, l_window:l_window*2] # shape of (N, 2, 32)
        self.test = train[:, :, n_train:] # shape of (N, 2, L-n_train)
        for i in range(1, n_train//l_window):
            train_x_next = train[:, :, l_window*i:l_window*(i+1)]
            train_y_next = train[:, :, l_window*(i+1):l_window*(i+2)]
            self.train_x = np.concatenate([self.train_x, train_x_next], axis=0)
            self.train_y = np.concatenate([self.train_y, train_y_next], axis=0)


def cspn_ts(conf):
    """
    TS prediction in time domain with cspn
    """
    
    # load data and parameters
    batch_size = conf.batch_size
    x_shape = (batch_size, 2, 32, 1)
    y_shape = (batch_size, 2 * 32)
    x_dims = 2 * 32
    y_dims = 2 * 32
    dataset = MackeyGlassDataset()
    x_ph = tf.placeholder(tf.float32, x_shape)
    train_ph = tf.placeholder(tf.bool)
    
    # initialize CSPN
    sum_weights, leaf_weights = model.build_nn_mnist_half(x_ph, y_shape, train_ph, 2600, 320)
    param_provider = RAT_SPN.ScopeBasedParamProvider(sum_weights, leaf_weights)

    rg = region_graph.RegionGraph(range(y_dims))
    for _ in range(0, 8):
        rg.random_split(2, 2)

    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.param_provider = param_provider
    args.num_sums = 8
    args.num_gauss = 4
    args.dist = 'Gauss'
    spn = RAT_SPN.RatSpn(1, region_graph=rg, name="spn", args=args)
    
    # train CSPN
    t_start = time.time()
    print("Training CSPN")
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    trainer = CspnTrainer(spn, dataset, x_ph, train_ph, conf, sess=sess)
    trainer.run_training()
    t_end = time.time()
    print("Train: took {} min".format(str((t_end-t_start)/60)))
    
    # test phase, use MPE
    t_start = time.time()
    batches_per_epoch = dataset.test.shape[0] // conf.batch_size
    mse=0
    # for all batches
    for j in range(batches_per_epoch):
        # there are 15 windows for test
        for i in range(15):
            x_batch = dataset.test[j*conf.batch_size:(j+1)*conf.batch_size, :,  i   *32:(i+1)*32]
            y_batch = dataset.test[j*conf.batch_size:(j+1)*conf.batch_size, :, (i+1)*32:(i+2)*32]
            feed_dict = {trainer.x_ph: x_batch,
                         trainer.y_ph: np.zeros_like(y_batch),
                         trainer.marginalized: np.ones(trainer.marginalized.shape),
                         trainer.train_ph: False}
            mpe_i = trainer.spn.reconstruct_batch(feed_dict, trainer.sess)
            y_gt = np.concatenate([y_batch[:,0,:], y_batch[:,1,:]], axis=1).reshape(conf.batch_size, -1)
            mse += mean_squared_error(y_gt, mpe_i)
    mse /= (i*j)
    print("MSE from CSPN:", mse)
    t_end = time.time()
    print("Test: took {} min".format(str((t_end-t_start)/60)))
    
    # for plot    
    mpe1 = []
    mpe2 = []
    data_x = []
    # plot 3 windows of forecasting
    for i in range(3):
        x_batch = dataset.test[:conf.batch_size, :,  i   *32:(i+1)*32]
        y_batch = dataset.test[:conf.batch_size, :, (i+1)*32:(i+2)*32]
        feed_dict = {trainer.x_ph: x_batch,
                     trainer.y_ph: np.zeros_like(y_batch),
                     trainer.marginalized: np.ones(trainer.marginalized.shape),
                     trainer.train_ph: False}
        mpe_i = trainer.spn.reconstruct_batch(feed_dict, trainer.sess)
        mpe1.append(mpe_i[:,:32])
        mpe2.append(mpe_i[:,32:])
        data_x.append(x_batch)
    
    return data_x, y_batch, mpe1, mpe2, mse


def wcspn_ts(conf):
    """
    TS prediction in fourier domain with cspn-wspn
    """

    # load data and parameters
    batch_size = conf.batch_size
    x_shape = (batch_size, 2, 32, 1)
    y_shape = (batch_size, 2*34)
    x_dims = 2 * 32
    y_dims = 2 * 34
    dataset = MackeyGlassDataset()
    x_ph = tf.placeholder(tf.float32, x_shape)
    train_ph = tf.placeholder(tf.bool)

    # apply FFT
    x_ph_rfft = tf.signal.rfft(tf.reshape(x_ph, [64, 2, 32]))
    x_ph_rfft = tf.reshape(x_ph_rfft, [64, 2, -1, 1])
    x_ph_rfft_real = tf.real(x_ph_rfft)
    x_ph_rfft_imag = tf.imag(x_ph_rfft)
    x_ph_rfft = tf.concat([x_ph_rfft_real, x_ph_rfft_imag], 2)
    sum_weights, leaf_weights = model.build_nn_mnist_half(x_ph_rfft, y_shape, train_ph, 2600, 320)
    # initialize CSPN
    param_provider = RAT_SPN.ScopeBasedParamProvider(sum_weights, leaf_weights)

    rg = region_graph.RegionGraph(range(y_dims))
    for _ in range(0, 8):
        rg.random_split(2, 2)

    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.param_provider = param_provider
    args.num_sums = 8
    args.num_gauss = 4
    args.dist = 'Gauss'
    spn = RAT_SPN.RatSpn(1, region_graph=rg, name="spn", args=args)

    # train CSPN
    t_start = time.time()
    print("Training WSPN")
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    trainer = WspnTrainer(spn, dataset, x_ph, train_ph, conf, sess=sess)
    trainer.run_training()
    t_end = time.time()
    print("Train: took {} min".format(str((t_end-t_start)/60)))


    def wspn_irfft(mpe):
        # do irfft on mpe/sample results
        # 1. extract the real and imag coefficients
        T_W = 32//2 + 1
        y1_r = mpe[:, 0   :T_W]
        y1_i = mpe[:,T_W  :T_W*2]
        y2_r = mpe[:,T_W*2:T_W*3]
        y2_i = mpe[:,T_W*3:T_W*4]
        # 2. construct the complex number
        y1_rfft = y1_r + y1_i * 1j
        y2_rfft = y2_r + y2_i * 1j
        # 3. irfft
        y1 = np.fft.irfft(y1_rfft)
        y2 = np.fft.irfft(y2_rfft)

        return np.concatenate([y1, y2], axis=1)

        
    # test phase, use MPE
    t_start = time.time()
    batches_per_epoch = dataset.test.shape[0] // conf.batch_size
    mse=0
    # for all batches
    for j in range(batches_per_epoch):
        # there are 15 windows for test
        for i in range(15):
            x_batch = dataset.test[j*conf.batch_size:(j+1)*conf.batch_size, :,  i   *32:(i+1)*32]
            y_batch = dataset.test[j*conf.batch_size:(j+1)*conf.batch_size, :, (i+1)*32:(i+2)*32]
            feed_dict = {trainer.x_ph: x_batch,
                         trainer.y_ph: np.zeros_like(y_batch),
                         trainer.marginalized: np.ones(trainer.marginalized.shape),
                         trainer.train_ph: False}
            mpe_i = wspn_irfft(trainer.spn.reconstruct_batch(feed_dict, trainer.sess))
            y_gt = np.concatenate([y_batch[:,0,:], y_batch[:,1,:]], axis=1).reshape(conf.batch_size, -1)
            mse += mean_squared_error(y_gt, mpe_i)
    mse /= (i*j)
    print("MSE from WSPN:", mse)
    t_end = time.time()
    print("Test: took {} min".format(str((t_end-t_start)/60)))
    
    # for plot
    mpe1 = []
    mpe2 = []
    data_x = []
    # plot 3 windows of forecasting
    for i in range(3):
        x_batch = dataset.test[:conf.batch_size, :,  i   *32:(i+1)*32]
        y_batch = dataset.test[:conf.batch_size, :, (i+1)*32:(i+2)*32]
        feed_dict = {trainer.x_ph: x_batch,
                     trainer.y_ph: np.zeros_like(y_batch),
                     trainer.marginalized: np.ones(trainer.marginalized.shape),
                     trainer.train_ph: False}
        # need to irfft to transfer back to time domain
        mpe_i = wspn_irfft(trainer.spn.reconstruct_batch(feed_dict, trainer.sess))
        mpe1.append(mpe_i[:,:32])
        mpe2.append(mpe_i[:,32:])
        data_x.append(x_batch)
    
    return data_x, y_batch, mpe1, mpe2, mse


def plot_prediction_3(x_batch1, y_batch1, mpe11, mpe12, 
                      x_batch2, y_batch2, mpe21, mpe22):
    """
    Plot the time series forecasting results,
    from both CSPN and Whittle CSPN
    """
    n_window = len(mpe11)
    t = np.arange(32*(n_window+1))
    t_mpe = np.arange(32, 32*(n_window+1))
    fig, a = plt.subplots(2,2,figsize=(8,4))
    k=0# 0th sample
    # plot CSPN results
    data = np.concatenate(x_batch1, axis=2)
    data = np.concatenate([data, y_batch1], axis=2)
    mpe1 = np.concatenate(mpe11, axis=1)
    mpe2 = np.concatenate(mpe12, axis=1)
    i=0
    gt = "GT"
    a[i][0].plot(t, data[k, 0, :, 0], 'b-', label=gt)
    a[i][0].plot(t_mpe, mpe1[k, :], 'r--', label="CSPN")
    a[i][0].legend(prop={'size':10}, loc='lower left')
    a[i][0].set_ylabel('$y_1=f_{1}(t)$')
    a[i][0].set_title('CSPN in time domain - $1^{st}$ channel')
    a[i][0].set_xticks([])
    a[i][1].plot(t, data[k, 1, :, 0], 'c-', label=gt)
    a[i][1].plot(t_mpe, mpe2[k, :], 'm--', label="CSPN")
    a[i][1].legend(prop={'size':10}, loc='lower left')
    a[i][1].set_ylabel('$y_2=f_{2}(t)$')
    a[i][1].set_title('CSPN in time domain - $2^{nd}$ channel')
    a[i][1].set_xticks([])

    # plot WCSPN results
    data = np.concatenate(x_batch2, axis=2)
    data = np.concatenate([data, y_batch2], axis=2)
    mpe1 = np.concatenate(mpe21, axis=1)
    mpe2 = np.concatenate(mpe22, axis=1)
    i=1
    a[i][0].plot(t, data[k, 0, :, 0], 'b-', label=gt)
    a[i][0].plot(t_mpe, mpe1[k, :], 'r--', label="WSPN")
    a[i][0].legend(prop={'size':10}, loc='lower left')
    a[i][0].set_ylabel('$y_1=f_{1}(t)$')
    a[i][0].set_title('WSPN in Fourier domain - $1^{st}$ channel')
    a[i][1].plot(t, data[k, 1, :, 0], 'c-', label=gt)
    a[i][1].plot(t_mpe, mpe2[k, :], 'm--', label="WSPN")
    a[i][1].legend(prop={'size':10}, loc='lower left')
    a[i][1].set_ylabel('$y_2=f_{2}(t)$')
    a[i][1].set_title('WSPN in Fourier domain - $2^{nd}$ channel')
    
    plt.tight_layout()
    fig.savefig("wcspn.pdf")


if __name__ == "__main__":
    np.random.seed(23)
    with tf.device('/GPU:0'):
        conf = Config()

        x_batch1, y_batch1, mpe11, mpe12, mse1 = cspn_ts(conf)
        x_batch2, y_batch2, mpe21, mpe22, mse2 = wcspn_ts(conf)
        plot_prediction_3(x_batch1, y_batch1, mpe11, mpe12,
                x_batch2, y_batch2, mpe21, mpe22)
        print('CSPN MSE:', mse1)
        print('WSPN MSE:', mse2)


