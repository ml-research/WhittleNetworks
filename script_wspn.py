"""
script modeling Whittle Sun-Product Network with structure learning
Created on May 17, 2021
@author: Zhongjie Yu
"""
import numpy as np
import logging
import time
import argparse
import pickle
import scipy
import matplotlib.pyplot as plt
import sys

from scipy import stats
sys.path.append('./SPFlow/src/')
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats
from spn.structure.Base import Context
from spn.algorithms.LearningWrappers import learn_parametric

# set path base for saving all models and results
path_base = './results/'


def get_save_path(ARGS):
    # returns a path for saving models and results
    # define the WSPN leaf types
    if ARGS.wspn_type == 1:
        key = 'wspn1d'
    elif ARGS.wspn_type == 2:
        key = 'wspn_pair'
    elif ARGS.wspn_type == 3:
        key = 'wspn2d'
    else:
        print('input wspn type error')
        sys.exit()

    save_path = path_base + ARGS.data_type + '/' + key + '_' + str(
        ARGS.n_min_slice) + '_' + str(ARGS.threshold) + '/'

    return save_path


def get_l_rfft(ARGS):
    # get T_W based on datasets
    # hard coded here, but can be adapted to other window sizes
    if ARGS.data_type == 'sine':
        # T_W = floor(32/2) + 1
        l_rfft = 17
    elif ARGS.data_type == 'mnist':
        # T_W = floor(14/2) + 1 
        l_rfft = 8
    elif ARGS.data_type == 'SP':
        # T_W = floor(32/2) + 1
        l_rfft = 17
    elif ARGS.data_type == 'stock':
        # T_W = floor(32/2) + 1
        l_rfft = 17
    elif ARGS.data_type == 'billiards':
        # T_W = floor(100/2) + 1
        l_rfft = 51
    elif ARGS.data_type == 'VAR':
        # T_W = floor(32/2) + 1
        l_rfft = 17
    else:
        print('input l_rfft error')
        sys.exit()

    return l_rfft


def learn_whittle_spn_1d(train_data, n_RV, n_min_slice=2000, init_scope=None):
    # train a WSPN with univariate Gaussian leaf nodes.
    # no pairwise constraints hold for Real and Imaginary parts of the Fourier coefficients
    from spn.structure.leaves.parametric.Parametric import Gaussian
    # pre-define for training
    ds_context = Context(parametric_types=[Gaussian] * n_RV).add_domains(train_data)
    # learn a WSPN
    print('learning WSPN 1d')
    # l_rfft=None --> 1d gaussian node, 
    # ==> is_2d does not work
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, threshold=ARGS.threshold,
                            initial_scope=init_scope, cpus=1, l_rfft=None, is_2d=False)
    # save the WSPN
    save_path = get_save_path(ARGS)
    check_path(save_path)
    f = open(save_path + 'wspn_1d.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_1d(ARGS):
    # load the trained WSPN for test use
    save_path = get_save_path(ARGS)
    print('Load model from:', save_path+'wspn_1d.pkl') 
    f = open(save_path + 'wspn_1d.pkl', 'rb')
    spn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(spn)
    print(log_msg)
    logger.info(log_msg)
    return spn


def learn_whittle_spn_pair(train_data, n_RV, n_min_slice, init_scope=None):
    # train a WSPN with pairwise Gaussian leaf nodes.
    # pairwise constraints hold for Real and Imaginary parts of the Fourier coefficients
    # Diagonal covariance matrix for each leaf node
    from spn.structure.leaves.parametric.Parametric import Gaussian
    # pre-define for training
    ds_context = Context(parametric_types=[Gaussian] * n_RV).add_domains(train_data)
    print('learning WSPN Pair')
    l_rfft = get_l_rfft(ARGS)
    # l_rfft!=None --> 2d/pair gaussian node, 
    # ==> is_2d=False --> pair gaussian, i.e., diagonal covariance matrix
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, threshold=ARGS.threshold,
                            initial_scope=init_scope, cpus=1, l_rfft=l_rfft, is_2d=False)
    # save the WSPN
    save_path = get_save_path(ARGS)
    check_path(save_path)
    f = open(save_path + 'wspn_pair.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_pair(ARGS, log=False):
    # load the trained WSPN for test use
    save_path = get_save_path(ARGS)
    print('Load model from:', save_path+'wspn_pair.pkl') 
    f = open(save_path + 'wspn_pair.pkl', 'rb')
    spn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(spn)
    print(log_msg)
    logger.info(log_msg)
    return spn


def learn_whittle_spn_2d(train_data, n_RV, n_min_slice, init_scope=None):
    # train a WSPN with 2d Gaussian leaf nodes.
    # pairwise constraints hold for Real and Imaginary parts of the Fourier coefficients
    # Full covariance matrix for each leaf node
    from spn.structure.leaves.parametric.Parametric import MultivariateGaussian
    # pre-define for training
    ds_context = Context(parametric_types=[MultivariateGaussian] * n_RV).add_domains(train_data)
    print('learning WSPN 2d')
    l_rfft = get_l_rfft(ARGS)
    # l_rfft!=None --> 2d/pair gaussian node, 
    # ==> is_2d=True --> pairwise gaussian, i.e., full covariance matrix
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, threshold=ARGS.threshold,
                            initial_scope=init_scope, cpus=1, l_rfft=l_rfft, is_2d=True)
    # save the WSPN
    save_path = get_save_path(ARGS)
    check_path(save_path)
    f = open(save_path + 'wspn_2d.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_2d(ARGS, log=True):
    # load the trained WSPN for test use
    save_path = get_save_path(ARGS)
    print('Load model from:', save_path+'wspn_2d.pkl') 
    f = open(save_path + 'wspn_2d.pkl', 'rb')
    spn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(spn)
    print(log_msg)
    logger.info(log_msg)

    return spn


def data_to_2d(data, p, L):
    # transfer data from 1d to 2d
    h, w = data.shape
    l = L // 2 + 1
    data1 = data.reshape(h * p, -1)
    data1_r = data1[:, 0:l].reshape(h * p, l, 1)
    data1_i = data1[:, l:].reshape(h * p, l, 1)
    data2 = np.concatenate([data1_r, data1_i], 2)
    data2 = data2.reshape(h, -1, 2)

    return data2


def load_data_for_wspn(ARGS):
    if ARGS.data_type == 'sine':
        log_msg = 'loading sine data'
        print(log_msg)
        data_train = np.fromfile('./data/sine/train_sine.dat',
                                 dtype=np.float64).reshape(-1, 204)
        data_pos = np.fromfile('./data/sine/test_sine_positive.dat',
                               dtype=np.float64).reshape(-1, 204)
        data_neg = np.fromfile('./data/sine/test_sine_negative.dat',
                               dtype=np.float64).reshape(-1, 204)
        n_RV = 204  # number of RVs
        p = 6  # dim
        L = 32  # TS length
        # Two RVs are not modeled as they are all 0, they are the imaginary parts of frequency 0 and \pi
        scope_list = np.arange(n_RV)
        scope_temp = np.delete(scope_list, np.where(scope_list % 34 == 17))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 34 == 33)))
    elif ARGS.data_type == 'mnist':
        log_msg = 'loading mnist data'
        print(log_msg)
        data_train = np.fromfile('./data/train_mnist.dat',
                                 dtype=np.float64).reshape(-1, 224)
        data_pos = np.fromfile('./data/test_mnist_positive.dat',
                               dtype=np.float64).reshape(-1, 224)
        data_neg = np.fromfile('./data/test_mnist_negative.dat',
                               dtype=np.float64).reshape(-1, 224)
        n_RV = 224  # number of RVs
        p = 14  # dim
        L = 14  # TS length
        # Two RVs are not modeled as they are all 0, they are the imaginary parts of frequency 0 and \pi
        scope_list = np.arange(n_RV)
        scope_temp = np.delete(scope_list, np.where(scope_list % 16 == 8))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 16 == 15)))
    elif ARGS.data_type == 'SP':
        log_msg = 'loading S&P data'
        print(log_msg)
        data_train = np.fromfile('./data/train_SP.dat',
                                 dtype=np.float64).reshape(-1, 374)
        # fill pos and neg with training data, not used
        data_pos = data_train.copy()
        data_neg = data_train.copy()
        n_RV = 374  # number of RVs
        p = 11  # dim
        L = 32  # TS length
        # Two RVs are not modeled as they are all 0, they are the imaginary parts of frequency 0 and \pi
        scope_list = np.arange(n_RV)
        scope_temp = np.delete(scope_list, np.where(scope_list % 34 == 17))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 34 == 33)))
    elif ARGS.data_type == 'stock':
        log_msg = 'loading Stock data'
        print(log_msg)
        data_train = np.fromfile('./data/train_stock.dat',
                                 dtype=np.float64).reshape(-1, 578)
        data_pos = data_train.copy()
        data_neg = data_train.copy()
        n_RV = 578  # number of RVs
        p = 17  # dim
        L = 32  # TS length
        # Two RVs are not modeled as they are all 0, they are the imaginary parts of frequency 0 and \pi
        scope_list = np.arange(n_RV)
        scope_temp = np.delete(scope_list, np.where(scope_list % 34 == 17))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 34 == 33)))
    elif ARGS.data_type == 'billiards':
        log_msg = 'loading Billiards data'
        print(log_msg)
        # Load data
        data_path = './data/'
        data = pickle.load(open(data_path + 'billiards_train_10000.pkl', 'rb'))
        # Extract training data
        positions = data['y']
        positions = positions[..., :2]
        positions = positions[0:9700, ...]
        # normalize to [-1, 1]
        data_max = np.max(positions)
        data_min = np.min(positions)
        positions = 2/(data_max-data_min)*(positions-data_min)-1
        # Apply FFT
        data_rfft = np.fft.rfft(positions, axis=1)
        d_r = data_rfft.real
        d_i = data_rfft.imag
        data_x1 = np.concatenate([d_r[:, :, 0, 0], d_i[:, :, 0, 0]], axis=1)
        data_y1 = np.concatenate([d_r[:, :, 0, 1], d_i[:, :, 0, 1]], axis=1)
        data_x2 = np.concatenate([d_r[:, :, 1, 0], d_i[:, :, 1, 0]], axis=1)
        data_y2 = np.concatenate([d_r[:, :, 1, 1], d_i[:, :, 1, 1]], axis=1)
        data_x3 = np.concatenate([d_r[:, :, 2, 0], d_i[:, :, 2, 0]], axis=1)
        data_y3 = np.concatenate([d_r[:, :, 2, 1], d_i[:, :, 2, 1]], axis=1)
        # data_train in the form of [x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, x3r, x3i, y3r, y3i]
        data_train = np.concatenate((data_x1, data_y1, data_x2, data_y2, data_x3, data_y3), axis=1)

        # Extract test data
        positions = data['y']
        positions = positions[..., :2]
        positions = positions[9700:, ...]
        # normalize to [-1, 1]
        positions = 2/(data_max-data_min)*(positions-data_min)-1
        data_rfft = np.fft.rfft(positions, axis=1)
        d_r = data_rfft.real
        d_i = data_rfft.imag
        data_x1 = np.concatenate([d_r[:, :, 0, 0], d_i[:, :, 0, 0]], axis=1)
        data_y1 = np.concatenate([d_r[:, :, 0, 1], d_i[:, :, 0, 1]], axis=1)
        data_x2 = np.concatenate([d_r[:, :, 1, 0], d_i[:, :, 1, 0]], axis=1)
        data_y2 = np.concatenate([d_r[:, :, 1, 1], d_i[:, :, 1, 1]], axis=1)
        data_x3 = np.concatenate([d_r[:, :, 2, 0], d_i[:, :, 2, 0]], axis=1)
        data_y3 = np.concatenate([d_r[:, :, 2, 1], d_i[:, :, 2, 1]], axis=1)
        # data_train in the form of [x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, x3r, x3i, y3r, y3i]
        data_pos = np.concatenate((data_x1, data_y1, data_x2, data_y2, data_x3, data_y3), axis=1)

        # Load outlier data
        # first load drift, that balls having no collision
        print('billiards_test_drift.pkl file not uploaded....\nUse billiards_train_10000.pkl file instead if you want.')
        sys.exit()
        data = pickle.load(open(data_path + 'billiards_test_drift.pkl', 'rb'))
        # extract data and do DTFT
        positions = data['y']
        positions = positions[..., :2]
        # normalize to [-1, 1]
        positions = 2 / (data_max - data_min) * (positions - data_min) - 1
        ##### create simple outlier by adding noise to the movement
        states00 = positions.copy()
        for i in range(positions.shape[0]):
            for j in range(3):
                # choose x or y
                xy = np.random.randint(0, 2)
                # set it constant
                rand_pos = np.random.rand(1)*2-1 + np.random.rand(100)*0.2
                rand_pos[rand_pos > 1] = 1
                rand_pos[rand_pos < -1] = -1
                states00[i, :, j, xy] = rand_pos
        positions = states00.copy()
        #####
        # Apply FFT
        data_rfft = np.fft.rfft(positions, axis=1)
        d_r = data_rfft.real
        d_i = data_rfft.imag
        data_x1 = np.concatenate([d_r[:, :, 0, 0], d_i[:, :, 0, 0]], axis=1)
        data_y1 = np.concatenate([d_r[:, :, 0, 1], d_i[:, :, 0, 1]], axis=1)
        data_x2 = np.concatenate([d_r[:, :, 1, 0], d_i[:, :, 1, 0]], axis=1)
        data_y2 = np.concatenate([d_r[:, :, 1, 1], d_i[:, :, 1, 1]], axis=1)
        data_x3 = np.concatenate([d_r[:, :, 2, 0], d_i[:, :, 2, 0]], axis=1)
        data_y3 = np.concatenate([d_r[:, :, 2, 1], d_i[:, :, 2, 1]], axis=1)
        # data_train in the form of [x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, x3r, x3i, y3r, y3i]
        data_neg = np.concatenate((data_x1, data_y1, data_x2, data_y2, data_x3, data_y3), axis=1)

        n_RV = 612  # number of RVs
        p = 6  # dim
        L = 100  # TS length
        scope_list = np.arange(n_RV)
        # Two RVs are not modeled as they are all 0, they are the imaginary parts of frequency 0 and \pi
        scope_temp = np.delete(scope_list, np.where(scope_list % 102 == 51))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 102 == 101)))
    elif ARGS.data_type == 'VAR':
        log_msg = 'loading VAR Simulation data'
        print(log_msg)
        var_data = np.genfromtxt('./data/VAR.csv', delimiter=',')
        n_RV = 34*7  # number of RVs
        p = var_data.shape[0]  # dim
        L = 32  # TS length
        N = var_data.shape[1]//L
        # create windows
        X = np.zeros((N*p,L), dtype=np.float64) 
        for i in range(N):
            X[i*p:(i+1)*p, :] = var_data[:,i*L:(i+1)*L]
        # Apply FFT
        data_rfft = np.fft.rfft(X, axis=1) 
        data_rfft = np.concatenate([data_rfft.real, data_rfft.imag], axis=1).reshape(-1, n_RV)
        # split training/test sets
        L = 16384
        data_train = data_rfft[0:L, :]
        data_pos = data_rfft[L:, :]  # similar to Sine data
        # exchange channels of data_pos for data_neg
        data_neg = data_pos.copy()
        data_neg[:, 0:34] = data_pos[:, 136:170]
        data_neg[:, 136:170] = data_pos[:, 0:34] 
        data_neg[:, 34:68] = data_pos[:, 102:136]
        data_neg[:, 102:136] = data_pos[:, 34:68] 
        scope_list = np.arange(n_RV)
        # Two RVs are not modeled as they are all 0, they are the imaginary parts of frequency 0 and \pi
        scope_temp = np.delete(scope_list, np.where(scope_list % 34 == 17))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 34 == 33)))
    else:
        raise Exception("Incorrect dataset, can only be the following:\n sine\n mnist\n SP\n stock\n billiards\n VAR\n")
    print('data done')
    return data_train, data_pos, data_neg, n_RV, p, L, init_scope


def check_path(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def calc_ll(wspn, data_train, data_pos, data_neg):
    # calculate LL
    log_msg = 'Log-likelihood calculating...'
    print(log_msg)
    logger.info(log_msg)

    ll_train = log_likelihood(wspn, data_train)
    ll_pos = log_likelihood(wspn, data_pos)
    ll_neg = log_likelihood(wspn, data_neg)
    log_msg = '---------median-----------'
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_train=' + str(np.median(ll_train))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_test=' + str(np.median(ll_pos))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_ood=' + str(np.median(ll_neg))
    print(log_msg)
    logger.info(log_msg)
    log_msg = '--------- mean -----------'
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_train=' + str(np.mean(ll_train))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_test=' + str(np.mean(ll_pos))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_ood=' + str(np.mean(ll_neg))
    print(log_msg)
    logger.info(log_msg)

    return ll_train, ll_pos, ll_neg


def save_ll(ll1, ll2, ll3):
    save_path = get_save_path(ARGS)
    check_path(save_path)

    np.savetxt(save_path + 'll_train.csv', ll1, delimiter=',')
    np.savetxt(save_path + 'll_pos.csv', ll2, delimiter=',')
    np.savetxt(save_path + 'll_neg.csv', ll3, delimiter=',')


def init_log(ARGS):
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # Creating log file
    save_path = get_save_path(ARGS)
    check_path(save_path)
    # path_base = '/media/yu/data/yu/code/gp_whittle/WhittleNetwork/dev/'
    if ARGS.train_type == 1:
        file_base = 'train_wspn_' + str(ARGS.wspn_type) + '_on_data_' + ARGS.data_type + '_'
    elif ARGS.train_type == 2:
        file_base = 'test_wspn_' + str(ARGS.wspn_type) + '_on_data_' + ARGS.data_type + '_'
    else:
        file_base = 'error'
    logging.basicConfig(
        filename=save_path + file_base + current_time + '.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    return logger


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    # Args go here
    parser.add_argument('--wspn_type', type=int, default=3,
                        help='Type of wspn, 1-1d, 2-pair, 3-2d')
    parser.add_argument('--train_type', type=int, default=2,
                        help='Type of train, 1-train, 2-test')
    parser.add_argument('--n_min_slice', type=int, default=100,
                        help='minimum size of slice.')
    parser.add_argument('--data_type', type=str, default='sine',
                        help='Type of data, can be: sine, mnist, SP, stock, billiards, VAR')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Threshold of splitting features')

    ARGS, unparsed = parser.parse_known_args()

    # init logger
    logger = init_log(ARGS)
    log_msg = '\n--wspn_type=' + str(ARGS.wspn_type) + \
              '\n--train_type=' + str(ARGS.train_type) + \
              '\n--n_min_slice=' + str(ARGS.n_min_slice) + \
              '\n--data_type=' + ARGS.data_type + \
              '\n--threshold=' + str(ARGS.threshold)
    print(log_msg)
    logger.info(log_msg)
    start_time = time.time()
    np.random.seed(123)

    # load data and data_info
    data_train, data_pos, data_neg, n_RV, p, L, init_scope = load_data_for_wspn(ARGS)

    if ARGS.wspn_type == 1:
        # train/load wspn 1d
        n_min_slice = ARGS.n_min_slice
        if ARGS.train_type == 1:
            log_msg = 'Train WSPN 1d'
            logger.info(log_msg)
            wspn = learn_whittle_spn_1d(data_train, n_RV, n_min_slice, init_scope)
        elif ARGS.train_type == 2:
            log_msg = 'Test WSPN 1d'
            logger.info(log_msg)
            wspn = load_whittle_spn_1d(ARGS)
            # calculate LL and save for significance test
            [ll_train, ll_pos, ll_neg] = calc_ll(wspn, data_train, data_pos, data_neg)
            save_ll(ll_train, ll_pos, ll_neg)

    elif ARGS.wspn_type == 2:
        # train/load wspn pair
        n_min_slice = ARGS.n_min_slice
        if ARGS.train_type == 1:
            log_msg = 'Train WSPN pair'
            logger.info(log_msg)
            wspn = learn_whittle_spn_pair(data_train, n_RV, n_min_slice, init_scope)
        elif ARGS.train_type == 2:
            log_msg = 'Test WSPN pair'
            logger.info(log_msg)
            wspn = load_whittle_spn_pair(ARGS, log=True)
            # calculate LL and save for significance test
            [ll_train, ll_pos, ll_neg] = calc_ll(wspn, data_train, data_pos, data_neg)
            save_ll(ll_train, ll_pos, ll_neg)

    elif ARGS.wspn_type == 3:
        # train/load wspn 2d
        n_min_slice = ARGS.n_min_slice
        if ARGS.train_type == 1:
            log_msg = 'Train WSPN 2d'
            logger.info(log_msg)
            wspn = learn_whittle_spn_2d(data_train, n_RV, n_min_slice, init_scope)
        elif ARGS.train_type == 2:
            log_msg = 'Test WSPN 2d'
            logger.info(log_msg)
            wspn = load_whittle_spn_2d(ARGS)
            # calculate LL and save for significance test
            [ll_train, ll_pos, ll_neg] = calc_ll(wspn, data_train, data_pos, data_neg)
            save_ll(ll_train, ll_pos, ll_neg)

    log_msg = 'Running time: ' + str((time.time() - start_time) / 60.0) + 'minutes'
    logger.info(log_msg)
