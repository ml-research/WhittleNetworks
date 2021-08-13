"""
script extracting (conditional) independencies from WSPNs (Bayesian network or Markov network)
Created on May 17, 2021
@author: Zhongjie Yu
for VAR dataset
"""
import numpy as np
import networkx as nx
import logging
import time
import argparse
import pickle
import matplotlib.pyplot as plt
from graph_functions import spn2bn_hill_climb, spn2mn_hill_climb 
import sys
sys.path.append('./SPFlow/src/')

path_base = './results/'


def init_log_var(args):
    # the log file to store the graph information
    #dataset = get_dataset_name(args)
    dataset = args.data_type
    current_time=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    logging.basicConfig(
        filename=path_base + '/log_graph_' + dataset + '_'+current_time+'.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    return logger


def load_data_and_model(args):
    # load params, data, and model, given data type
    if args.data_type == 'Sine':
        # get the parameters based on data
        L = 32 # window size
        n_RV = 6 # TS dimensions
        label = ['Sine11', 'Sine12', 'Sine13', 'Sine21', 'Sine22', 'Gauss1']
        # load data, pre-processed with FFT already
        data = np.fromfile('./data/sine/train_sine.dat', dtype=np.float64).reshape(-1, 204) 
        N = data.shape[0]
        # load WSPN model
        save_path = path_base + 'sine/wspn_pair_1000_0.3/'
        f = open(save_path+'wspn_pair.pkl', 'rb') 
        wspn = pickle.load(f)
        f.close()
    elif args.data_type == 'SP':
        L = 32 # window size
        n_RV = 11 # TS dimensions
        label = ['Industrials', 'Consumer Staples', 'Utilities', 'Consumer Discretionary', 
                'Real Estate', 'Energy', 'Information Technology', 'Financials', 
                'Telecom Services', 'Health Care', 'Materials']
        # load data, pre-processed with FFT already 
        data = np.fromfile('./data/train_SP.dat', dtype=np.float64).reshape(-1, 374) 
        N = data.shape[0]
        # load WSPN model 
        save_path = path_base + 'SP/wspn_pair_5_0.8/'
        f = open(save_path+'wspn_pair.pkl', 'rb') 
        wspn = pickle.load(f) 
        f.close()
    elif args.data_type == 'Stock':
        L = 32 # window size
        n_RV = 17 # TS dimensions
        label = ['NE', 'AU', 'AT', 'BE', 'FR', 'IT', 'UK', 'GE', 'CA', 
                'HK', 'SP', 'IR', 'JP', 'FN', 'PO', 'US', 'CH']
        # load data, pre-processed with FFT already 
        data = np.fromfile('./data/train_stock.dat', dtype=np.float64).reshape(-1, 578) 
        N = data.shape[0]
        # load WSPN model 
        save_path = path_base + 'stock/wspn_pair_5_0.8/'
        f = open(save_path+'wspn_pair.pkl', 'rb') 
        wspn = pickle.load(f) 
        f.close()
    elif args.data_type == 'VAR':
        # get the parameters based on data
        L = 32 # window size
        n_RV = 7 # number of RVs
        label = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']
        # load data and do FFT
        var_data = np.genfromtxt('./data/VAR.csv', delimiter=',')
        N_window = var_data.shape[1]//L
        T_W = 32//2 + 1
        X = np.zeros((N_window*n_RV,L), dtype=np.float64)
        for i in range(N_window):
            X[i*n_RV:(i+1)*n_RV, :] = var_data[:,i*L:(i+1)*L]
        data_rfft = np.fft.rfft(X, axis=1)
        data_rfft = np.concatenate([data_rfft.real, data_rfft.imag], axis=1).reshape(-1, n_RV*T_W*2)
        N = 16384 # samples to train
        data = data_rfft[0:N, :]
        # load WSPN model
        save_path = path_base + 'VAR/wspn_pair_1000_0.3/'
        f = open(save_path+'wspn_pair.pkl', 'rb')
        wspn = pickle.load(f)
        f.close()
    else:
        raise Exception("Incorrect dataset, can only be the following:\n Sine\n SP\n Stock\n VAR\n")
    
    log_msg = '\nLoad model from: \n' + save_path + 'wspn_pair.pkl'
    print(log_msg)
    logger.info(log_msg)
    print("WSPN loaded")
    
    return L, N, n_RV, label, data, wspn


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    # Args go here
    parser.add_argument('--data_type', type=str, default='Sine',
                        help='Type of data, can be: Sine, SP, Stock, VAR')
    parser.add_argument('--graph_type', type=str, default='bn',
                        help='Type of graph, can be: bn, mn')
    parser.add_argument('--BIC', default=False, action="store_true",
                        help='flag to use BIC or not')

    args, unparsed = parser.parse_known_args()

    # init logger
    logger = init_log_var(args)
    if args.graph_type=='bn':
        log_msg = '\n--graph_type=BN'
    elif args.graph_type=='mn':
        log_msg = '\n--graph_type=MN'
    else:
        raise AssertionError("Wrong graph type, can be either bn or mn")
    print(log_msg)
    logger.info(log_msg)

    start_time = time.time()
    np.random.seed(123)
    # load all params, data, and model
    L, N, n_RV, label, data, wspn = load_data_and_model(args)
    
    # graph structure
    if args.graph_type == 'bn':
        # Directed Graph case
        print("Creating Directed Graph")
        G = spn2bn_hill_climb(wspn, data, n_RV, label, 2, 2, logger, bic=args.BIC)
        logger.info('Directed Graph created')
    else:
        # Undirected Graph case
        print("Creating Undirected Graph")
        G = spn2mn_hill_climb(wspn, data, n_RV, label, 2, 2, logger, bic=args.BIC)
        logger.info('Undirected Graph created')
    print("Edges in the graph:")
    print(G.edges)
    logger.info(G.edges)
    log_msg = 'Running time: ' + str((time.time() - start_time)/60.0) + 'minutes'
    logger.info(log_msg)
    print("--- %.2f minutes ---" % ((time.time() - start_time)/60.0))

    print('Finished')



