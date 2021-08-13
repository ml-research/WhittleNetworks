"""
Functions for Extracting Conditional independencies from  Whittle SPNs
created on May 17, 2021
@author Zhongjie Yu
"""
import numpy as np
import networkx as nx
from tqdm import tqdm
import sys

sys.path.append('./SPFlow/src/')
from spn.algorithms.Marginalization import marginalize
from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.Base import Sum, Product, assign_ids, rebuild_scopes_bottom_up


def learn_whittle_spn(train_data, n_RV, L, n_min_slice=50):
    """
    Function to train Whittle SPN given data
    1) All frequencies are being trained.
    2) Assuming each frequancy to be independent --> A product node over all frequencies
    Note: this function is not use in the demo, it shows an example of WSPN with independent frequencies.
    """

    ds_context = Context(parametric_types=[Gaussian] * n_RV * L).add_domains(train_data)
    # prepare scopes of all RVs
    init_scope_init = np.arange(n_RV)
    # WSPNs for freq=0, and \pi, whose imaginary part is all 0s
    # the scopes to be trained for freq=0
    init_scope = list(init_scope_init * L)
    print('learning SPN0')
    spn_real_0 = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, initial_scope=init_scope)
    # the scopes to be trained for freq=\pi
    init_scope = list(init_scope_init * L + int(L / 2))
    print('learning SPN1')
    spn_real_1 = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, initial_scope=init_scope)
    # combine the two WSPNs with a product node
    whittle_spn = Product([spn_real_0, spn_real_1])
    assign_ids(whittle_spn)
    rebuild_scopes_bottom_up(whittle_spn)
    # for the other freqs #1 - #L/2
    init_scope_init = np.arange(n_RV)
    init_scope1 = init_scope_init * L
    init_scope2 = init_scope1 + int(L / 2)
    init_scope_init = np.concatenate([init_scope1, init_scope2])
    for k in range(1, int(L / 2)):
        # train spn for the frequency
        init_scope = list(init_scope_init + k)
        print('learning SPN', k, 'of', int(L / 2))
        spn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, initial_scope=init_scope)
        # combine it with other freqs with product node
        whittle_spn = Product([whittle_spn, spn])
        assign_ids(whittle_spn)
        rebuild_scopes_bottom_up(whittle_spn)

    return whittle_spn


def spn2bn_hill_climb(whittle_spn, train_data, n_RV, label, max_num_parents, max_num_children, logger, bic=False):
    """
    Extract directed graph from WSPN, by maximizing the Whittle likelihood
    :param whittle_spn:      the WSPN learned from data
    :param train_data:       the Fourier coefficients of data
    :param n_RV:             number of random variables 
    :param label:            label of each dimension of multivariate time series
    :param max_num_parents:  max number of parents of a node
    :param max_num_children: max number of children of a node
    :return: G               directed graph   
    """
    # number of data instances
    N = train_data.shape[0]
    # init the nodes of G (enpty graph)
    G = nx.DiGraph()
    for idx in range(n_RV):
        G.add_node(idx, label=label[idx])
    v_out, v_in = node_list_bn(n_RV)
    # list of valid edges for DAG
    v_out_valid, v_in_valid = node_valid_bn(G, v_out, v_in)
    # Whittle likelihood of empty graph
    whittle_llh_cur = calc_whittle_llh(G, whittle_spn, train_data)
    if bic:
        # adjust the Whittle likelihood if BIC is enabled
        whittle_llh_cur = -calc_BIC_score(N, 0, whittle_llh_cur)
    flag = True
    n_iter = 1
    log_msg = 'Initial Whittle llh: ' + str(whittle_llh_cur)
    print(log_msg)
    logger.info(log_msg)
    while flag:
        print('searching iter:', n_iter)
        # list Whittle likelihood after adding each edge
        # init likelihood list as empty, same shape as v_out_valid list
        whittle_llh_list = np.empty(v_out_valid.shape)
        # calculate Whittle likelihood for all valid edges
        for index in tqdm(range(len(v_out_valid))):
            G_temp = G.copy()
            G_temp.add_edge(v_out_valid[index], v_in_valid[index])
            whittle_llh_list[index] = calc_whittle_llh(G_temp, whittle_spn, train_data)
        # if enable BIC 
        if bic:
            whittle_llh_list = -calc_BIC_score(n_RV, n_iter, whittle_llh_list)

        # find max Whittle likelihood
        max_index = np.argmax(whittle_llh_list)
        # if it is larger than the current Whittle likelihood, add the corresponding edge
        if whittle_llh_list[max_index] > whittle_llh_cur:
            G.add_edge(v_out_valid[max_index], v_in_valid[max_index])
            print('Current edges:', G.edges)
            whittle_llh_cur = whittle_llh_list[max_index]
            log_msg = 'In iter ' + str(n_iter) + ', add edge: ' + str(v_out_valid[max_index]) + '-->' + str(
                v_in_valid[max_index])
            print(log_msg)
            logger.info(log_msg)
            log_msg = 'Current Whittle llh: ' + str(whittle_llh_list[max_index])
            print(log_msg)
            logger.info(log_msg)
            # delete the pair of nodes of the index
            v_out, v_in = node_delete(G, v_out, v_in, max_index, max_num_parents, max_num_children)
            # list of valid edges for DAG
            v_out_valid, v_in_valid = node_valid_bn(G, v_out, v_in)
            # check if all edges are added
            if len(v_in_valid) == 0:
                flag = False
                log_msg = 'Edge list empty.'
                print(log_msg)
                logger.info(log_msg)
        # check if reaches the top
        else:
            flag = False
            log_msg = 'Reach top of hill.'
            print(log_msg)
            logger.info(log_msg)
        # next iteration of adding an edge
        n_iter += 1
    return G


def spn2mn_hill_climb(whittle_spn, train_data, n_RV, label, max_num_parents, max_num_children, logger, bic=False):
    """
    Extract undirected graph from WSPN, by maximizing the Whittle likelihood
    :param whittle_spn:      the WSPN learned from data
    :param train_data:       the Fourier coefficients of data
    :param n_RV:             number of random variables 
    :param label:            label of each dimension of multivariate time series
    :param max_num_parents:  max number of parents of a node, not activated in MN
    :param max_num_children: max number of children of a node, not activated in MN
    :return: G               undirected graph   
    """
    # number of data instances
    N = train_data.shape[0]
    # init the nodes of G (empty graph)
    G = nx.Graph()
    for idx in range(n_RV):
        G.add_node(idx, label=label[idx])
    # list of valid edges for undirected graph
    v_1, v_2 = node_list_mn(n_RV)
    # Whittle likelihood of empty graph
    whittle_llh_cur = calc_whittle_llh(G, whittle_spn, train_data)
    if bic:
        # adjust the Whittle likelihood if BIC is enabled
        whittle_llh_cur = -calc_BIC_score(N, 0, whittle_llh_cur)

    flag = True
    n_iter = 1
    log_msg = 'Initial Whittle llh: ' + str(whittle_llh_cur)
    print(log_msg)
    logger.info(log_msg)
    while flag:
        print('searching iter:', n_iter)
        # list Whittle likelihood after adding each edge
        # first get the valid list of node pairs that make the graph chordal
        v_1_valid, v_2_valid = node_valid_mn(G, v_1, v_2)
        # init likelihood list as empty, same shape as v_out_valid list
        whittle_llh_list = np.empty(v_1_valid.shape)
        # calculate Whittle likelihood for all valid edges
        for index in tqdm(range(len(v_1_valid))):
            G_temp = G.copy()
            G_temp.add_edge(v_1_valid[index], v_2_valid[index])
            whittle_llh_list[index] = calc_whittle_llh(G_temp, whittle_spn, train_data)
        # if use BIC 
        if bic:
            whittle_llh_list = -calc_BIC_score(n_RV, n_iter, whittle_llh_list)
        # find max Whittle likelihood
        max_index = np.argmax(whittle_llh_list)
        # if it is larger than the current Whittle likelihood, add the corresponding edge
        if whittle_llh_list[max_index] > whittle_llh_cur:
            G.add_edge(v_1_valid[max_index], v_2_valid[max_index])
            print('Current edges:', G.edges)
            whittle_llh_cur = whittle_llh_list[max_index]
            log_msg = 'In iter ' + str(n_iter) + ', add edge: ' + str(v_1_valid[max_index]) + '-->' + str(
                v_2_valid[max_index])
            print(log_msg)
            logger.info(log_msg)
            log_msg = 'Current Whittle llh: ' + str(whittle_llh_list[max_index])
            print(log_msg)
            logger.info(log_msg)
            # delete the pair of nodes of the index
            ind_1 = v_1 == v_1_valid[max_index]
            ind_2 = v_2 == v_2_valid[max_index]
            ind_v = ind_1 * ind_2
            index_to_delete = int(np.where(ind_v == 1)[0])
            v_1 = np.delete(v_1, index_to_delete)
            v_2 = np.delete(v_2, index_to_delete)
            # check if all edges are added
            if len(v_1) == 0:
                flag = False
                log_msg = 'Edge list empty.'
                print(log_msg)
                logger.info(log_msg)
        # check if reaches the top
        else:
            flag = False
            log_msg = 'Reach top of hill.'
            print(log_msg)
            logger.info(log_msg)
        # next iteration of adding an edge
        n_iter += 1
    return G


def calc_BIC_score(n, k, llh):
    # calc BIC = ln(n)*k - 2ln(lh)
    bic = np.log(n) * k - 2 * llh

    return bic


def node_list_bn(n):
    # pairwise nodes list for directed graph
    v_list = np.arange(0, n).reshape(1, -1)
    v_out = np.repeat(v_list, n - 1, axis=1)
    v_in = np.repeat(v_list, n, axis=0)
    v_in = v_in[~np.eye(v_in.shape[0], dtype=bool)].reshape(1, -1)

    return v_out[0], v_in[0]


def node_list_mn(n):
    # pairwise nodes list for undirected graph
    from itertools import combinations
    v_list = list(combinations(range(n), 2))
    v_array = np.array(v_list)
    v_1 = v_array[:, 0]
    v_2 = v_array[:, 1]

    return v_1, v_2


def node_valid_bn(G, v_out, v_in):
    # filter out node pairs that conflicts the BN setting
    # the BN needs to be acyclic
    keep_list = np.ones(v_out.shape, dtype=bool)
    for index in range(len(v_out)):
        G_temp = G.copy()
        G_temp.add_edge(v_out[index], v_in[index])
        n_cycles = len(list(nx.simple_cycles(G_temp)))
        if n_cycles > 0:
            keep_list[index] = False
    v_out_valid = v_out[keep_list]
    v_in_valid = v_in[keep_list]

    return v_out_valid, v_in_valid


def node_valid_mn(G, v_1, v_2):
    # filter out node pairs that make the MN non-chordal
    # the MN needs to be chordal
    keep_list = np.ones(v_1.shape, dtype=bool)
    for index in range(len(v_1)):
        G_temp = G.copy()
        G_temp.add_edge(v_1[index], v_2[index])
        if not nx.is_chordal(G_temp):
            keep_list[index] = False
    v_1_valid = v_1[keep_list]
    v_2_valid = v_2[keep_list]

    return v_1_valid, v_2_valid


def node_delete(G, v_out, v_in, max_index, max_num_parents, max_num_children):
    # after adding an edge to BN, the node list has to be modified
    # 1. delete pair of nodes, e.g., (0,1) and (1,0)
    i1 = v_in == v_out[max_index]
    i2 = v_out == v_in[max_index]
    pair_index = [i for i, val in enumerate(i1 * i2) if val]
    v_out_temp = v_out.copy()
    v_in_temp = v_in.copy()
    if len(pair_index) > 0:
        v_out_temp = np.delete(v_out, np.array([max_index, pair_index[0]]))
        v_in_temp = np.delete(v_in, np.array([max_index, pair_index[0]]))
    else:
        v_out_temp = np.delete(v_out, max_index)
        v_in_temp = np.delete(v_in, max_index)
    # 2. delete pairs which will have too many parents
    for index in range(len(G)):
        if len(list(G.predecessors(index))) >= max_num_parents:
            # control max of parents
            i3 = v_in_temp == index
            pair_index2 = [i for i, val in enumerate(i3) if val]
            if len(pair_index2) > 0:
                v_out_temp = np.delete(v_out_temp, np.asarray(pair_index2))
                v_in_temp = np.delete(v_in_temp, np.asarray(pair_index2))
        if len(list(G.successors(index))) >= max_num_children:
            # control max of children
            i4 = v_out_temp == index
            pair_index3 = [i for i, val in enumerate(i4) if val]
            if len(pair_index3) > 0:
                v_out_temp = np.delete(v_out_temp, np.asarray(pair_index3))
                v_in_temp = np.delete(v_in_temp, np.asarray(pair_index3))

    return v_out_temp, v_in_temp


def calc_whittle_llh(G, whittle_spn, train_data):
    # calculate the Whittle likelihood given Graph, WSPN and data
    # choose the calculation with BN or MN
    if nx.is_directed(G):
        llh = calc_whittle_llh_bn(G, whittle_spn, train_data)
    else:
        llh = calc_whittle_llh_mn(G, whittle_spn, train_data)

    return llh


def calc_whittle_llh_bn(G, whittle_spn, train_data):
    # Calculate Whittle likelihood given BN, 
    # cf. eq(8) in Yu et al. Whittle Networks: A Deep Likelihood Model for Time Series. ICML 2020
    llh = 0
    # for each node
    for p in range(len(G.nodes)):
        # get Pa(V_p)
        Parents_Vp = list(G.predecessors(p))
        # check if Vp has parents or not
        if len(Parents_Vp) == 0:
            # in this case no Denominator of P(Pa)
            # scopes of V_p and parents of V_p
            scopes_Vp = get_scope_from_frequency(G, whittle_spn, [p])
            # marginal SPNs
            spn_marg_Vp = marginalize(whittle_spn, scopes_Vp)
            # accumulate llh
            llh += log_likelihood(spn_marg_Vp, train_data)
        else:
            # in this case both P(Vp, Pa) and P(Pa)
            # scopes of V_p and parents of V_p
            scopes_Vp = get_scope_from_frequency(G, whittle_spn, [p])
            scopes_Pa = get_scope_from_frequency(G, whittle_spn, Parents_Vp)
            # marginal SPNs
            spn_marg_Vp_Pa = marginalize(whittle_spn, scopes_Pa + scopes_Vp)
            spn_marg_Pa = marginalize(whittle_spn, scopes_Pa)
            # accumulate llh
            llh += log_likelihood(spn_marg_Vp_Pa, train_data)
            llh -= log_likelihood(spn_marg_Pa, train_data)
    # return the mean of llh
    return np.mean(llh)


def calc_whittle_llh_mn(G, whittle_spn, train_data):
    # Calculate Whittle likelihood given BN, cf. eq(8) in paper
    # cf. eq(1) in Tank et al. Bayesian Structure Learning for Stationary Time Series. UAI 2015
    from networkx.algorithms.clique import find_cliques
    llh = 0
    # for each sub-graph
    # for sub_graph in nx.connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        sub_graph = G.subgraph(c)
        # get cliques and separators
        clique_list = list(find_cliques(sub_graph))
        # if there is only one node in the sub-graph
        if len(clique_list) == 1:
            # if the sub-graph contains only one clique
            llh += calc_marginal_from_scope(G, whittle_spn, train_data, list(sub_graph.nodes))
        else:
            # if the sub-graph contains more than one clique
            # find the separator
            separator_list = get_separator_list(clique_list)
            # LL for cliques
            for clique in clique_list:
                llh += calc_marginal_from_scope(G, whittle_spn, train_data, clique)
            for separator in separator_list:
                llh -= calc_marginal_from_scope(G, whittle_spn, train_data, list(separator))
    # return the mean of llh
    return np.mean(llh)


def get_separator_list(clique_list):
    # get separator list from the set of cliques
    separator_list = []
    # source http://conferences.inf.ed.ac.uk/bayes250/slides/green.pdf
    c_j = clique_list[0] # prepare for the union of C_j
    i = 1
    while i < len(clique_list):
        c_i = clique_list[i] # current clique
        s_i = list(set(c_j) & set(c_i)) # find current separator
        # it is possible that s_i=[] because of the order of cliques
        # thus if s_i=[], muve the current clique to the end of the list
        if len(s_i)==0:
            #print('temporal failure, re-order cliques...')
            clique_list.append(clique_list.pop(clique_list.index(c_i)))
        else:
            c_j = list(set(c_j).union(c_i)) # update the union of C_j
            separator_list.append(s_i)
            i+=1
    
    return separator_list


def calc_marginal_from_scope(G, whittle_spn, train_data, node_list):
    # 1 get corresponding scopes from nodes
    # 2 calculate marginal likelihood
    scopes = get_scope_from_frequency(G, whittle_spn, node_list)
    # marginal SPNs
    spn_marg = marginalize(whittle_spn, scopes)
    # accumulate llh
    llh = log_likelihood(spn_marg, train_data)

    return llh


def get_scope_from_frequency(G, whittle_spn, p_list):
    """
    :param G:
    :param whittle_spn:   WSPN model,
    :param p_list:    list of frequencies
    :return:
    """
    n_scope = len(whittle_spn.scope)
    n_freq = int(n_scope / len(G))
    base = np.arange(n_freq)
    scope_list = []
    for index in range(len(p_list)):
        cur_list = base + p_list[index] * n_freq
        scope_list += list(cur_list)

    return scope_list


def log_return(data):
    # series column-wise
    data_t = np.log(data) * 100
    data_p1 = np.delete(data_t, 0, 0)
    data_p0 = np.delete(data_t, -1, 0)
    log_r = data_p1 - data_p0
    return log_r


def training_data_prepare(data, L, N):
    # apply rfft to input data
    data = np.transpose(data)  # --> (p,T)
    T = data.shape[1]
    assert (T > L)
    # deal with overlaps
    assert ((T - L) > (N-1))
    k = int((T - L) / (N-1))

    data0 = data[:, 0:L]
    for i in range(1, N):
        data0 = np.concatenate([data0, data[:, k * i:L + k * i]], axis=0)
    # FFT
    fft_L = np.fft.rfft(data0)
    fft_L_real = fft_L.real
    fft_L_imag = fft_L.imag
    # remove 2 columns of 0s in fft_L_imag
    fft_L_imag_remove_0s = fft_L_imag
    train_data = np.concatenate([fft_L_real, fft_L_imag_remove_0s], axis=1).reshape(N, -1)

    return train_data
