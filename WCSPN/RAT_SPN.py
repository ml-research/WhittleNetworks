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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import tensorflow.contrib.distributions as dists

import time


def add_to_map(given_map, key, item):
    existing_items = given_map.get(key, [])
    given_map[key] = existing_items + [item]


def variable_with_weight_decay(name, shape, stddev, wd, mean=0.0, values=None):
    if values is None:
        initializer = tf.truncated_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32)
    else:
        initializer = tf.constant_initializer(values)
    """Get a TF variable with optional l2-loss attached."""
    var = tf.get_variable(
        name,
        shape,
        initializer=initializer,
        dtype=tf.float32)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        tf.add_to_collection('weight_losses', weight_decay)

    return var


def print_if_nan(tensor, msg):
    is_nan = tf.reduce_any(tf.is_nan(tensor))
    return tf.cond(is_nan, lambda: tf.Print(tensor, [is_nan], message=msg), lambda: tf.identity(tensor))


class NodeVector(object):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class SpnArgs(object):
    def __init__(self):
        self.gauss_min_var = 0.1
        self.gauss_max_var = 1.0
        self.gauss_min_mean = None
        self.gauss_max_mean = None
        self.num_gauss = 20
        self.gauss_param_l2 = None
        self.gauss_isotropic = False
        self.dist = 'Gauss'

        self.linear_sum_weights = False
        self.normalized_sums = True
        self.sum_weight_l2 = None
        self.num_sums = 20
        self.drop_connect = False
        self.param_provider = BasicParamProvider(0.0)


class BasicParamProvider:
    def __init__(self, wd_l2=0.0):
        self.wd_l2 = wd_l2

    def grab_sum_parameters(self, num_inputs, num_sums, name=None):
        return variable_with_weight_decay(
            name,
            [1, num_inputs, num_sums],
            stddev=1e-1,
            wd=self.wd_l2,
            values=None)

    def grab_leaf_parameters(self, scope, number, name=None):
        num_inputs = len(scope)
        return self.grab_sum_parameters(num_inputs, number, name)


class UnorderedParamProvider(object):
    def __init__(self, parameters):
        self.params = parameters
        self.used = 0

    def grab_sum_parameters(self, num_inputs, num_sums, name=None):
        num_vars = num_inputs * num_sums
        batch_size = int(self.params.shape[0])
        allocated_params = self.params[:, self.used : self.used + num_vars]
        self.used += num_vars
        result = tf.reshape(allocated_params, [batch_size, num_inputs, num_sums], name=name)
        return result

    def grab_leaf_parameters(self, scope, number, name=None):
        num_inputs = len(scope)
        return self.grab_sum_parameters(num_inputs, number, name)


class ScopeBasedParamProvider:
    def __init__(self, sum_params, leaf_params):
        self.sum_params = sum_params
        self.leaf_params = leaf_params
        self.sum_params_used = 0
        self.leaf_params_used = [0] * int(leaf_params.shape[1])
        self.batch_size = int(self.sum_params.shape[0])

    def grab_sum_parameters(self, num_inputs, num_sums, name=None):
        num_vars = num_inputs * num_sums
        assert(self.sum_params_used + num_vars < int(self.sum_params.shape[1]))
        allocated_params = self.sum_params[:, self.sum_params_used : self.sum_params_used + num_vars]
        self.sum_params_used += num_vars
        result = tf.reshape(allocated_params, [self.batch_size, num_inputs, num_sums], name=name)
        return result

    def grab_leaf_parameters(self, scope, number, name=None):
        result = []
        idxs = np.zeros((self.batch_size, len(scope), number, 3), dtype=np.int32)
        idxs[..., 0] = np.reshape(np.arange(self.batch_size), (self.batch_size, 1, 1))
        for i, dim in enumerate(scope):
            idxs[:, i, :, 1] = dim
            cur_used = self.leaf_params_used[dim]
            idxs[:, i, :, 2] = np.expand_dims(np.arange(cur_used, cur_used + number), 0)
            self.leaf_params_used[dim] += number
        result = tf.gather_nd(self.leaf_params, idxs)
        return result


class BernoulliVector(NodeVector):
    def __init__(self, region, args, name,
                 given_params=None, num_dims=0):
        super().__init__(name)
        self.local_size = len(region)
        self.args = args
        self.scope = sorted(list(region))
        self.size = args.num_gauss
        self.num_dims = num_dims
        self.np_params = None
        self.params = self.args.param_provider.grab_leaf_parameters(
            self.scope,
            args.num_gauss,
            name=name + "_p")

        self.dist = dists.Bernoulli(logits=self.params)

    def forward(self, inputs, marginalized=None):
        local_inputs = tf.gather(inputs, self.scope, axis=1)
        local_inputs = tf.expand_dims(local_inputs, axis=-1)
        log_pdf_single = self.dist.log_prob(local_inputs)

        if marginalized is not None:
            marginalized = tf.clip_by_value(marginalized, 0.0, 1.0)
            local_marginalized = tf.expand_dims(tf.gather(marginalized, self.scope, axis=1), axis=-1)
            log_pdf_single *= (1 - local_marginalized)

        log_pdf = tf.reduce_sum(log_pdf_single, 1)
        return log_pdf

    def modes(self, case_num=0):
        probs = 1 / (1 + np.exp(-self.np_params[case_num]))
        return np.round(probs)

    def reconstruct(self, max_idxs, node_num, case_num, sample):
        if sample:
            my_sample = sess.run(self.dist.sample())[case_num]
        else:
            my_sample = self.modes(case_num)

        my_sample = my_sample[:, node_num]
        full_sample = np.zeros((self.num_dims,))
        full_sample[self.scope] = my_sample
        return full_sample

    def sample(self, num_samples, num_dims, seed=None):
        sample_values = self.dist.sample(num_samples, seed=seed)[:, 0]
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        sample_shape = [num_samples, num_dims, self.size]
        indices = tf.meshgrid(tf.range(num_samples), self.scope, tf.range(self.size))
        indices = tf.stack(indices, axis=-1)
        indices = tf.transpose(indices, [1, 0, 2, 3])
        samples = tf.scatter_nd(indices, sample_values, sample_shape)
        # indices is for filling the other values with 0s so we can just add in the same dimenison simply
        return samples

    def num_params(self):
        return self.params.shape[1:].num_elements()


class GaussVector(NodeVector):
    def __init__(self, region, args, name,
                 given_means=None, given_stddevs=None, mean=0.0, num_dims=0):
        super().__init__(name)
        self.local_size = len(region)
        self.args = args
        self.scope = sorted(list(region))
        self.size = args.num_gauss
        self.num_dims = num_dims
        self.np_means = None
        self.means = self.args.param_provider.grab_leaf_parameters(
            self.scope,
            args.num_gauss,
            name=name + "_means")

        if args.gauss_min_var < args.gauss_max_var:
            sigma_params = self.args.param_provider.grab_leaf_parameters(
                self.scope,
                args.num_gauss,
                name=name + "_sigma_params")

            self.sigma = args.gauss_min_var + (args.gauss_max_var - args.gauss_min_var) * tf.sigmoid(sigma_params)
        else:
            self.sigma = 1.0

        self.dist = dists.Normal(self.means, tf.sqrt(self.sigma))

    def forward(self, inputs, marginalized=None):
        local_inputs = tf.gather(inputs, self.scope, axis=1)
        local_inputs = tf.expand_dims(local_inputs, axis=-1)
        gauss_log_pdf_single = self.dist.log_prob(local_inputs)
        if marginalized is not None:
            marginalized = tf.clip_by_value(marginalized, 0.0, 1.0)
            local_marginalized = tf.expand_dims(tf.gather(marginalized, self.scope, axis=1), axis=-1)
            gauss_log_pdf_single = gauss_log_pdf_single * (1 - local_marginalized)

        gauss_log_pdf = tf.reduce_sum(gauss_log_pdf_single, 1)
        return gauss_log_pdf

    def reconstruct(self, max_idxs, node_num, case_num, sample):
        if sample:
            my_sample = sess.run(self.dist.sample())[case_num]
        else:
            my_sample = self.np_means[case_num]
        my_sample = my_sample[:, node_num]
        full_sample = np.zeros((self.num_dims,))
        full_sample[self.scope] = my_sample
        return full_sample

    def sample(self, num_samples, num_dims, seed=None):
        sample_values = self.dist.sample([num_samples], seed=seed)[:, 0]
        sample_shape = [num_samples, num_dims, self.size]
        indices = tf.meshgrid(self.scope, tf.range(num_samples), tf.range(self.size))
        indices = tf.stack(indices, axis=-1)
        indices = tf.transpose(indices, [1, 0, 2, 3])
        samples = tf.scatter_nd(indices, sample_values, sample_shape)
        return samples

    def num_params(self):
        result = self.means.shape[1:].num_elements()
        if isinstance(self.sigma, tf.Tensor):
            result += self.sigma.shape.num_elements()
        return result


class ProductVector(NodeVector):
    def __init__(self, vector1, vector2, name):
        """Initialize a product vector, which takes the cross-product of two distribution vectors."""
        super().__init__(name)
        self.vector1 = vector1
        self.vector2 = vector2
        self.inputs = [vector1, vector2]

        self.scope = list(set(vector1.scope) | set(vector2.scope))

        assert len(set(vector1.scope) & set(vector2.scope)) == 0

        self.size = vector1.size * vector2.size

    def forward(self, inputs):
        dists1 = inputs[0]
        dists2 = inputs[1]
        # HARD CODED Forward!!

        with tf.variable_scope('products') as scope:
            num_dist1 = int(dists1.shape[1])
            num_dist2 = int(dists2.shape[1])

            # we take outer products, thus expand in different dims
            dists1_expand = tf.expand_dims(dists1, 1)
            dists2_expand = tf.expand_dims(dists2, 2)

            # product == sum in log-domain
            prod = dists1_expand + dists2_expand
            # flatten out the outer product
            prod = tf.reshape(prod, [dists1.shape[0], num_dist1 * num_dist2])
        return prod

    def num_params(self):
        return 0

    def sample(self, inputs, seed=None):
        output_shape = [inputs[0].shape[0], inputs[0].shape[1], (inputs[0].shape[2] * inputs[1].shape[2])]
        in1_expand = tf.expand_dims(inputs[0], -1)
        in2_expand = tf.expand_dims(inputs[1], -2)

        result = in1_expand + in2_expand

        # We need to sample from each distribution with regards to one other one
        # so the last dimension has to be squared to go over all possible combinations
        # of the two different parts

        result_shape = list(result.shape[:2]) + [result.shape[2] * result.shape[3]]
        result = tf.reshape(result, result_shape)
        return result

    def reconstruct(self, max_idxs, node_num, case_num, sample):
        row_num = node_num // self.vector1.size
        col_num = node_num % self.vector1.size
        result1 = self.vector1.reconstruct(max_idxs, col_num, case_num, sample)
        result2 = self.vector2.reconstruct(max_idxs, row_num, case_num, sample)
        return result1 + result2


class SumVector(NodeVector):
    def __init__(self, prod_vectors, num_sums, args, dropout_op=None, name="", given_weights=None):
        super().__init__(name)
        self.inputs = prod_vectors
        self.size = num_sums

        self.scope = self.inputs[0].scope

        for inp in self.inputs:
            assert set(inp.scope) == set(self.scope)

        self.dropout_op = dropout_op
        self.args = args

        num_inputs = sum([v.size for v in prod_vectors])
        self.params = self.args.param_provider.grab_sum_parameters(
            num_inputs,
            num_sums,
            name=name + "_weights"
        )

        if args.linear_sum_weights:
            if args.normalized_sums:
                self.weights = tf.nn.softmax(self.params, 1)
            else:
                self.weights = self.params ** 2
        else:
            if args.normalized_sums:
                self.weights = tf.nn.log_softmax(self.params, 1)
                if args.sum_weight_l2:
                    exp_weights = tf.exp(self.weights)
                    weight_decay = tf.multiply(tf.nn.l2_loss(exp_weights), args.sum_weight_l2)
                    tf.add_to_collection('losses', weight_decay)
                    tf.add_to_collection('weight_losses', weight_decay)
            else:
                self.weights = self.params

        self.max_child_idx = None

    def forward(self, inputs):
        prods = tf.concat(inputs, 1)
        weights = self.weights

        if self.args.linear_sum_weights:
            sums = tf.log(tf.matmul(tf.exp(prods), tf.squeeze(self.weights)))
        else:
            prods = tf.expand_dims(prods, axis=-1)
            if self.dropout_op is not None:
                if self.args.drop_connect:
                    batch_size = prods.shape[0]
                    prod_num = prods.shape[1]
                    dropout_shape = [batch_size, prod_num, self.size]

                    random_tensor = random_ops.random_uniform(dropout_shape, dtype=self.weights.dtype)
                    dropout_mask = tf.log(math_ops.floor(self.dropout_op + random_tensor))
                    weights = weights + dropout_mask

                else:
                    random_tensor = random_ops.random_uniform(prods.shape, dtype=prods.dtype)
                    dropout_mask = tf.log(math_ops.floor(self.dropout_op + random_tensor))
                    prods = prods + dropout_mask

            child_values = prods + weights
            self.max_child_idx = tf.argmax(child_values, axis=1)
            sums = tf.reduce_logsumexp(child_values, axis=1)

        return sums

    def reconstruct(self, max_idxs, node_num, case_num, sample):
        my_max_idx = max_idxs[self.name][case_num, node_num]
        for inp_vector in self.inputs:
            if my_max_idx < inp_vector.size:
                return inp_vector.reconstruct(max_idxs, my_max_idx, case_num, sample)
            my_max_idx -= inp_vector.size

    def sample(self, inputs, seed=None, differentiable=False):
        inputs = tf.concat(inputs, 2)
        inputs = tf.transpose(inputs, [0, 2, 1])
        logits = tf.transpose(self.weights[0])
        dist = dists.Categorical(logits=logits)
        indices = dist.sample([inputs.shape[0]], seed=seed)
        case_idx = tf.tile(tf.expand_dims(tf.range(inputs.shape[0]), -1), [1, self.size])

        full_idx = tf.stack((case_idx, indices), axis=-1)
        result = tf.gather_nd(inputs, full_idx)
        result = tf.transpose(result, [0, 2, 1])
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        return result

    def num_params(self):
        return self.weights.shape[1:].num_elements()


class RatSpn(object):
    def __init__(self, num_classes, region_graph=None, vector_list=None, args=SpnArgs(), name=None, mean=0.0):
        if name is None:
            name = str(id(self))
        self.name = name
        self._region_graph = region_graph
        self.args = args
        self.default_mean = mean

        self.num_classes = num_classes

        # dictionary mapping regions to tensor of sums/input distributions
        self._region_distributions = dict()
        # dictionary mapping regions to tensor of products
        self._region_products = dict()

        self.vector_list = []
        self.output_vector = None

        # make the SPN...
        with tf.variable_scope(self.name) as scope:
            if region_graph is not None:
                self.num_dims = region_graph.get_num_items()
                self._make_spn_from_region_graph()
            elif vector_list is not None:
                self._make_spn_from_vector_list(vector_list)
            else:
                raise ValueError('Either vector_list or region_graph must not be None')

        self.num_dims = len(self.output_vector.scope)

    def _make_spn_from_vector_list(self, vector_list):
        self.vector_list = [[]]

        node_to_vec = {}

        for i, leaf_vector in enumerate(vector_list[0]):
            name = 'gauss_{}'.format(i)
            scope = leaf_vector[0].scope
            num_gauss = len(leaf_vector)
            means = np.zeros((len(scope), num_gauss))
            stdevs = np.zeros((len(scope), num_gauss))
            for j, prod_node in enumerate(leaf_vector):
                for k, gauss_node in enumerate(prod_node.children):
                    means[k, j] = gauss_node.mean
                    stdevs[k, j] = gauss_node.stdev

            gauss_vector = GaussVector(scope, self.args, name, given_means=means, given_stddevs=stdevs)
            self.vector_list[0].append(gauss_vector)

            for j, prod_node in enumerate(leaf_vector):
                node_to_vec[id(prod_node)] = gauss_vector

        for layer_num, layer in enumerate(vector_list[1:]):
            self.vector_list.append([])
            for vector_num, vector in enumerate(layer):
                if type(vector[0]) == base.Product:
                    child_vec1 = node_to_vec[id(vector[0].children[0])]
                    child_vec2 = node_to_vec[id(vector[0].children[1])]
                    name = 'prod_{}_{}'.format(layer_num, vector_num)
                    new_vector = ProductVector(child_vec1, child_vec2, name)
                elif type(vector[0]) == base.Sum:
                    child_vecs = list(set([node_to_vec[id(child_node)] for child_node in vector[0].children]))
                    assert len(child_vecs) <= 2
                    name = 'sum_{}_{}'.format(layer_num, vector_num)
                    num_inputs = sum([v.size for v in child_vecs])
                    weights = np.zeros((num_inputs, len(vector)))
                    for node_num, node in enumerate(vector):
                        weights[:, node_num] = node.weights
                    new_vector = SumVector(child_vecs, len(vector), self.args, name=name, given_weights=weights)
                else:
                    assert False

                self.vector_list[-1].append(new_vector)

                for node in vector:
                    node_to_vec[id(node)] = new_vector

        self.output_vector = self.vector_list[-1][-1]

    def _make_spn_from_region_graph(self):
        """Build a RAT-SPN."""

        rg_layers = self._region_graph.make_layers()
        self.rg_layers = rg_layers

        # make leaf layer (always Gauss currently)
        self.vector_list.append([])
        for i, leaf_region in enumerate(rg_layers[0]):
            if self.args.dist == 'Gauss':
                name = 'gauss_{}'.format(i)
                leaf_vector = GaussVector(leaf_region, self.args, name, mean=self.default_mean, num_dims=self.num_dims)
            elif self.args.dist == 'Bernoulli':
                name = 'bernoulli_{}'.format(i)
                leaf_vector = BernoulliVector(leaf_region, self.args, name, num_dims=self.num_dims)
            self.vector_list[-1].append(leaf_vector)
            self._region_distributions[leaf_region] = leaf_vector

        # make sum-product layers
        ps_count = 0
        for layer_idx in range(1, len(rg_layers)):
            self.vector_list.append([])
            if layer_idx % 2 == 1:
                partitions = rg_layers[layer_idx]
                for i, partition in enumerate(partitions):
                    input_regions = list(partition)
                    input1 = self._region_distributions[input_regions[0]]
                    input2 = self._region_distributions[input_regions[1]]
                    vector_name = 'prod_{}_{}'.format(layer_idx, i)
                    prod_vector = ProductVector(input1, input2, vector_name)
                    self.vector_list[-1].append(prod_vector)

                    resulting_region = tuple(sorted(input_regions[0] + input_regions[1]))
                    add_to_map(self._region_products, resulting_region, prod_vector)
            else:
                cur_num_sums = self.num_classes if layer_idx == len(rg_layers) - 1 else self.args.num_sums

                regions = rg_layers[layer_idx]
                for i, region in enumerate(regions):
                    product_vectors = self._region_products[region]
                    vector_name = 'sum_{}_{}'.format(layer_idx, i)
                    sum_vector = SumVector(product_vectors, cur_num_sums, self.args, name=vector_name)
                    self.vector_list[-1].append(sum_vector)

                    self._region_distributions[region] = sum_vector

                ps_count = ps_count + 1

        self.output_vector = self._region_distributions[self._region_graph.get_root_region()]

    def forward(self, inputs, marginalized=None):
        obj_to_tensor = {}
        for leaf_vector in self.vector_list[0]:
            obj_to_tensor[leaf_vector] = leaf_vector.forward(inputs, marginalized)

        for layer_idx in range(1, len(self.vector_list)):
            for vector in self.vector_list[layer_idx]:
                input_tensors = [obj_to_tensor[obj] for obj in vector.inputs]
                result = vector.forward(input_tensors)
                obj_to_tensor[vector] = result

        return obj_to_tensor[self.output_vector]

    def sample(self, num_samples=10, seed=None):
        vec_to_samples = {}
        for leaf_vector in self.vector_list[0]:
            # only leaf vector has number of samples, the other ones work on batch
            # size, can this even work?, yES since the input shape will bu used
            # not the actual batch size
            vec_to_samples[leaf_vector] = leaf_vector.sample(num_samples, self.num_dims, seed=seed)
        # vec_to_samples is now filled with keys of Bernoulli Vectors with
        # values of tensor shaped [1, 16, 3] (1, full scope size, num gauss)
        # itarate through layers(depth), ignore the leaf vectors
        for layer_idx in range(1, len(self.vector_list)):
            # iterate through each vector
            for vector in self.vector_list[layer_idx]:
                input_samples = [vec_to_samples[vec] for vec in vector.inputs]
                result = vector.sample(input_samples, seed=seed)
                vec_to_samples[vector] = result

        return vec_to_samples[self.output_vector]

    def reconstruct_batch(self, feed_dict, sess, sample=False):
        max_idx_tensors = {}
        for layer in self.vector_list:
            for vector in layer:
                if isinstance(vector, SumVector):
                    max_idx_tensors[vector.name] = vector.max_child_idx

        batch_size = list(feed_dict.values())[0].shape[0]
        self.eval_params(sess, feed_dict)
        max_idxs = sess.run(max_idx_tensors, feed_dict=feed_dict)
        recons = []
        for i in range(batch_size):
            recons.append(self.reconstruct(max_idxs, i, sample))
        recons = np.stack(recons, axis=0)
        return recons


    def reconstruct(self, max_idxs, case_num, sample):
        return self.output_vector.reconstruct(max_idxs, 0, case_num, sample)

    def eval_params(self, sess, feed_dict):
        param_tensors = {}
        for leaf_vector in self.vector_list[0]:
            if isinstance(leaf_vector, GaussVector):
                param_tensors[leaf_vector] = leaf_vector.means
            elif isinstance(leaf_vector, BernoulliVector):
                param_tensors[leaf_vector] = leaf_vector.params
            else:
                raise ValueError('unknown leaf vector')
        params_np = sess.run(param_tensors, feed_dict=feed_dict)
        for leaf_vector in self.vector_list[0]:
            if isinstance(leaf_vector, GaussVector):
                leaf_vector.np_means = params_np[leaf_vector]
            elif isinstance(leaf_vector, BernoulliVector):
                leaf_vector.np_params = params_np[leaf_vector]

    def num_params(self):
        result = 0
        params_per_dim = [0] * self.num_dims
        print('--- SPN parameters ---')
        for i, layer in enumerate(self.vector_list):
            layer_result = 0
            for vector in layer:
                layer_result += vector.num_params()
                if i == 0:
                    for dim in vector.scope:
                        params_per_dim[dim] += vector.size

            print("Layer {} has {} parameters.".format(i, layer_result))
            result += layer_result
        print('leaf parameters per RV', params_per_dim)

        return result

    def get_simple_spn(self, sess, single_root=False):
        start_time = time.time()
        vec_to_params = {}
        for leaf_vector in self.vector_list[0]:
            vec_to_params[leaf_vector] = (leaf_vector.means[0],
                                          leaf_vector.sigma[0])
        for layer_idx in range(1, len(self.vector_list)):
            if layer_idx % 2 == 0:
                for sum_vec in self.vector_list[layer_idx]:
                    vec_to_params[sum_vec] = sum_vec.weights[0]

        st = time.time()
        vec_to_params = sess.run(vec_to_params)
        time_tf = time.time() - st

        vec_to_nodes = {}
        node_id = -1

        for leaf_vector in self.vector_list[0]:
            vec_to_nodes[leaf_vector] = []
            means, sigmas = vec_to_params[leaf_vector]
            stdevs = np.sqrt(sigmas) + np.zeros_like(means) # Use broadcasting to expand stdev is necessary
            for i in range(leaf_vector.size):
                prod = base.Product()
                prod.id = node_id = node_id + 1
                prod.scope.extend(leaf_vector.scope)
                for j, r in enumerate(leaf_vector.scope):
                    gaussian = para.Gaussian(mean=means[j, i],
                                             stdev=stdevs[j, i],
                                             scope=[r])
                    gaussian.id = node_id = node_id + 1
                    prod.children.append(gaussian)

                vec_to_nodes[leaf_vector].append(prod)

        for layer_idx in range(1, len(self.vector_list)):
            # vector_list.append([])
            if layer_idx % 2 == 1:
                prod_vectors = self.vector_list[layer_idx]
                for i, prod_vector in enumerate(prod_vectors):
                    input1 = prod_vector.vector1
                    input2 = prod_vector.vector2

                    vec_to_nodes[prod_vector] = []

                    # The order of these loops is very important, otherwise weights will be mismatched
                    # input1 is the inner loop because it is the inner dimension
                    # of the outer product in ProductVector::forward
                    for c2 in range(input2.size):
                        for c1 in range(input1.size):
                            prod = base.Product()
                            prod.id = node_id = node_id + 1
                            prod.children.append(vec_to_nodes[input1][c1])
                            prod.children.append(vec_to_nodes[input2][c2])
                            prod.scope.extend(input1.scope)
                            prod.scope.extend(input2.scope)
                            vec_to_nodes[prod_vector].append(prod)

            else:
                sum_vectors = self.vector_list[layer_idx]
                for i, sum_vector in enumerate(sum_vectors):
                    vec_to_nodes[sum_vector] = []
                    weights = vec_to_params[sum_vector]

                    for j in range(sum_vector.size):
                        sum_node = base.Sum()
                        if layer_idx < len(self.vector_list) - 1:
                            sum_node.id = node_id = node_id + 1
                        else:
                            sum_node.id = node_id + 1
                        sum_node.scope.extend(sum_vector.scope)
                        input_vecs = [vec_to_nodes[prod_vec] for prod_vec in sum_vector.inputs]
                        input_nodes = [node for vec in input_vecs for node in vec]
                        sum_node.children.extend(input_nodes)

                        vec_to_nodes[sum_vector].append(sum_node)

                        log_weights = weights[:, j]
                        scaled_weights = np.exp(log_weights - np.max(log_weights))
                        normalized_weights = scaled_weights / np.sum(scaled_weights)
                        sum_node.weights.extend(normalized_weights)

        output_nodes = vec_to_nodes[self.output_vector]

        if single_root:
            for i, node in enumerate(output_nodes):
                node.id = node.id + i
                node_id += 1
            root = base.Sum()
            root.id = node_id = node_id + 1
            root.children.extend(output_nodes)
            root.scope.extend(output_nodes[0].scope)
            root.weights.extend([1.0 / float(len(output_nodes))] * len(output_nodes))
            return root

        print('conversion finished in {:3f}s'.format(time.time() - start_time))
        print('time spent evaluating by Tensorflow: {:3f}s'.format(time_tf))
        return output_nodes


def compute_performance(sess, data_x, data_labels, batch_size, spn):
    """Compute classification accuracy"""

    num_batches = int(np.ceil(float(data_x.shape[0]) / float(batch_size)))
    test_idx = 0
    num_correct = 0

    for test_k in range(0, num_batches):
        if test_k + 1 < num_batches:
            batch_data = data_x[test_idx:test_idx + batch_size, :]
            batch_labels = data_labels[test_idx:test_idx + batch_size]

        feed_dict = {spn.inputs: batch_data, spn.labels: batch_labels}
        if spn.dropout_input_placeholder is not None:
            feed_dict[spn.dropout_input_placeholder] = 1.0
        for dropout_op in spn.dropout_layer_placeholders:
            if dropout_op is not None:
                feed_dict[dropout_op] = 1.0

        spn_outputs = sess.run(spn.outputs, feed_dict=feed_dict)
        max_output = np.argmax(spn_outputs, axis=1)

        num_correct_batch = np.sum(max_output == batch_labels)

        num_correct += num_correct_batch

        test_idx += batch_size

    accuracy = num_correct / (num_batches * batch_size)

    return accuracy

