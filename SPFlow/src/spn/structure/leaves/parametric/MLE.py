"""
Created on April 15, 2018

@author: Alejandro Molina

Modified on July 23, 2021

@author: Zhongjie Yu
"""

import numpy as np
from scipy.stats import gamma, lognorm, bernoulli

from spn.structure.leaves.parametric.Parametric import (
    Gaussian,
    LogNormal,
    Gamma,
    Poisson,
    Exponential,
    Geometric,
    Categorical,
    Bernoulli,
    CategoricalDictionary,
    MultivariateGaussian,
)
import logging

logger = logging.getLogger(__name__)


def update_parametric_parameters_mle(node, data):
    if data.shape[0] == 0:
        return

    if isinstance(node, MultivariateGaussian):
        node.mean = np.mean(data, axis=0).tolist()
        node.sigma = np.cov(data, rowvar=False).tolist()

        # add by zhongjie on 06.10.2019, the square of stdev
        # check if 0 in cov
        if len(node.mean) == 1:
            # two cases in 1d: sigma=0 (all samples are the same) or sigma=nan (only one sample in data slice)
            if node.sigma < 0.00000001 or np.isnan(node.sigma):
            # if np.isclose(node.sigma, 0) or np.isnan(node.sigma):
                # 1d
                node.sigma = 0.00000001
                # node.sigma = 0.159154943
        elif len(node.mean) == 2:
            # 2d, make it PSD
            arr_cov = np.array(node.sigma)
            if arr_cov[0][0] < 0.00000001:
            # if np.isclose(arr_cov[0][0], 0, atol=1e-16):
                node.sigma[0][0] = 0.00000001
                # node.sigma[0][0] = 0.159154943
                node.sigma[0][1] = 0.0
                node.sigma[1][0] = 0.0
                print('changing variance!')
                # sys.exit()
            if arr_cov[1][1] < 0.00000001:
            # if np.isclose(arr_cov[1][1], 0, atol=1e-16):
                node.sigma[1][1] = 0.00000001
                # node.sigma[1][1] = 0.159154943
                node.sigma[0][1] = 0.0
                node.sigma[1][0] = 0.0
                print('changing variance!')
                # sys.exit()
        if isinstance(node.sigma, float):
            if np.isnan(node.sigma):
                print('nan!')
        return

    assert data.shape[1] == 1

    data = data[~np.isnan(data)]

    if isinstance(node, Gaussian):
        node.mean = np.mean(data).item()
        node.stdev = np.std(data).item()

        if node.stdev < 0.0001:
        # if np.isclose(node.stdev, 0):
            node.stdev = 0.0001
            print('changing stdev!')
            # sys.exit()
            # node.stdev = 0.39894228

    elif isinstance(node, Gamma):
        # default
        node.alpha = 1.1
        node.beta = 1.0
        if np.any(data <= 0):
            # negative data? impossible gamma
            return

        # zero variance? adding noise
        if np.isclose(np.std(data), 0):
            node.alpha = np.mean(data).item()
            return

        alpha, loc, theta = gamma.fit(data, floc=0)
        beta = 1.0 / theta
        if np.isfinite(alpha):
            node.alpha = alpha
            node.beta = beta

    elif isinstance(node, LogNormal):
        lognorm_params = lognorm.fit(data, floc=0)
        node.mean = np.log(lognorm_params[2]).item()
        node.stdev = lognorm_params[0]

    elif isinstance(node, Bernoulli):
        node.p = data.sum().item() / len(data)

    elif isinstance(node, Poisson):
        node.mean = np.mean(data).item()

    elif isinstance(node, Exponential):
        node.l = np.mean(data).item()

    elif isinstance(node, Geometric):
        node.p = len(data) / data.sum().item()

    elif isinstance(node, Categorical):
        psum = 0
        for i in range(len(node.p)):
            node.p[i] = 0
        for i in range(node.k):
            node.p[i] = np.sum(data == i)
            psum += node.p[i]
        node.p = node.p / psum
        node.p = node.p.tolist()

    elif isinstance(node, CategoricalDictionary):
        if node.p is not None:
            node.p.clear()
        v, c = np.unique(data, return_counts=True)
        p = c / c.sum()
        node.p = dict(zip(v.tolist(), p.tolist()))

    else:
        raise Exception("Unknown parametric " + str(type(node)))


if __name__ == "__main__":
    node = MultivariateGaussian(np.inf, np.inf)
    data = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 3, 3, 6, 2]).reshape(-1, 2)
    update_parametric_parameters_mle(node, data)
    assert np.all(np.isclose(node.mean, np.mean(data, axis=0)))
    assert np.all(np.isclose(node.sigma, np.cov(data, rowvar=0)))

    node = Gaussian(np.inf, np.inf)
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    update_parametric_parameters_mle(node, data)
    assert np.isclose(node.mean, np.mean(data))
    assert np.isclose(node.stdev, np.std(data))

    node = Gamma(np.inf, np.inf)
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    update_parametric_parameters_mle(node, data)
    assert np.isclose(node.alpha / node.beta, np.mean(data)), node.alpha

    node = LogNormal(np.inf, np.inf)
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    update_parametric_parameters_mle(node, data)
    assert np.isclose(node.mean, np.log(data).mean(), atol=0.00001)
    assert np.isclose(node.stdev, np.log(data).std(), atol=0.00001)

    node = Poisson(np.inf)
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    update_parametric_parameters_mle(node, data)
    assert np.isclose(node.mean, np.mean(data))

    node = Categorical(np.array([1, 1, 1, 1, 1, 1]) / 6)
    data = np.array([0, 0, 1, 3, 5]).reshape(-1, 1)
    update_parametric_parameters_mle(node, data)
    assert np.isclose(node.p[0], 2 / 5)
    assert np.isclose(node.p[1], 1 / 5)
    assert np.isclose(node.p[2], 0)
    assert np.isclose(node.p[3], 1 / 5)
    assert np.isclose(node.p[4], 0)
