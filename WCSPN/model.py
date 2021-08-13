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
import tensorflow as tf
import numpy as np
import RAT_SPN

def build_mnist_conv_layers(inp, training):
    with tf.variable_scope('nn'):
        print(inp.shape)
        conv1 = tf.layers.conv2d(inputs=inp,
                                 filters=64,
                                 kernel_size=5,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, 3, 3, padding='same')
        pool1 = tf.layers.dropout(pool1, training=training)
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=5,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, 3, 3, padding='same')
        pool2 = tf.layers.dropout(pool2, training=training)
        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=64,
                                 kernel_size=5,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3, 3, 3, padding='same')
    linearized = tf.reshape(pool3, (tf.shape(pool3)[0], 64*2*2))
    linearized = tf.layers.dropout(linearized, training=training)
    return linearized


def build_nn_mnist_baseline(inp, output_shape, training):
    output_dims = 1
    for dim in output_shape[1:]:
        output_dims *= int(dim)
    print("nn output shape:", output_shape)
    print("output dims:", output_dims)

    with tf.variable_scope('nn'):
        if int(inp.shape[-2]) == 28:
            features = build_mnist_conv_layers(inp, training)
        else:
            features = build_feature_extraction_layers(inp, training)
        fc1 = tf.layers.dense(inputs=features,
                              units=output_dims*2,
                              activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1,
                              units=output_dims,
                              activation=None)
    print_num_vars()
    return fc2

def build_nn_mnist(inp, output_shape, training, num_sum_weights, num_leaf_weights):
    batch_size = int(inp.shape[0])
    output_shape = list(output_shape)
    output_dims = 1
    for dim in output_shape[1:]:
        output_dims *= int(dim)

    if int(inp.shape[-2]) == 28:
        features = build_mnist_conv_layers(inp, training)
    else:
        features = build_feature_extraction_layers(inp, training)

    sums_roundup = ((num_sum_weights + 63) // 64) * 64
    sum_weights = tf.layers.dense(inputs=features,
                                  units=sums_roundup // 8,
                                  activation=tf.nn.relu)
    sum_weights = tf.reshape(sum_weights, [batch_size, sums_roundup // 64, 8])
    sum_weights = tf.layers.dense(inputs=sum_weights,
                                  units=64,
                                  activation=None)
    sum_weights = tf.reshape(sum_weights, [batch_size, sums_roundup])
    print(sum_weights.shape)
    output_proposal = tf.layers.dense(inputs=features,
                                      units=(num_leaf_weights * output_dims) // 4,
                                      activation=tf.nn.relu)
    output_proposal = tf.reshape(output_proposal, list(output_shape) + [-1])
    leaf_weights = tf.layers.dense(inputs=output_proposal,
                                   units=num_leaf_weights)
    print_num_vars()
    return sum_weights, leaf_weights


def build_nn_mnist_half(inp, output_shape, training, num_sum_weights, num_leaf_weights):
    batch_size = int(inp.shape[0])
    output_shape = list(output_shape)
    output_dims = 1
    for dim in output_shape[1:]:
        output_dims *= int(dim)

    sums_roundup = ((num_sum_weights + 63) // 64) * 64
    features = tf.reshape(inp, [batch_size, -1])
    dense_x1 = tf.layers.dense(inputs=features,
                               units=sums_roundup//8,
                               activation=tf.nn.relu)
    dense_x1 = tf.reshape(dense_x1, [batch_size, sums_roundup // 64, 8])
    sum_weights = tf.layers.dense(inputs=dense_x1,
                                  units=64,
                                  activation=None)
    sum_weights = tf.reshape(sum_weights, [batch_size, sums_roundup])
    # print(sum_weights.shape)
    dense_y1 = tf.layers.dense(inputs=features,
                               units=(num_leaf_weights * output_dims) // 4,
                               activation=tf.nn.relu)
    dense_y1 = tf.reshape(dense_y1, list(output_shape) + [-1])
    leaf_weights = tf.layers.dense(inputs=dense_y1,
                                   units=num_leaf_weights)
    leaf_weights = tf.reshape(leaf_weights, [batch_size, -1, num_leaf_weights])
    ##### print_num_vars()
    return sum_weights, leaf_weights


def build_feature_extraction_layers(inp, training):
    with tf.variable_scope('nn'):
        conv1 = tf.layers.conv2d(inputs=inp,
                                 filters=64,
                                 kernel_size=7,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, 3, 3, padding='same')
        pool1 = tf.layers.dropout(pool1, training=training)
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=5,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, 3, 3, padding='same')
        pool2 = tf.layers.dropout(pool2, training=training)
        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=64,
                                 kernel_size=5,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3, 3, 3, padding='same')
        conv4 = tf.layers.conv2d(inputs=pool3,
                                 filters=64,
                                 kernel_size=3,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv4, 3, 3, padding='same')
        pool4 = tf.layers.dropout(pool4, training=training)
        print(pool4.shape)
        linearized = tf.reshape(pool4, (tf.shape(pool4)[0], 2*2*64))
        return linearized


class MeanField:
    def __init__(self, params):
        self.params = params
        self.dist = tf.distributions.Bernoulli(logits=params)
        self.ll_output = None

    def forward(self, inp, _):
        self.ll_output = self.dist.log_prob(inp)
        return tf.reduce_sum(self.ll_output, axis=1, keep_dims=True)

    def reconstruct_batch(self, feed_dict, sess):
        p = sess.run(self.params, feed_dict=feed_dict)
        probs = 1/ (1+np.exp(-p))
        mpe = np.round(probs)
        return mpe


class MixtureDensityNetwork:
    def __init__(self, in_params, k, n_labels, alpha_weights=None):
        """
        Expects alphas to be already through softmax layer
        :param in_params:
        :param k:
        :param n_labels:
        :alpha_weigths: If parameter is set, will take it else will
        use k values from an_params
        """
        print("Building MDN")
        self.k = k
        self.n_labels = n_labels

        if alpha_weights is not None:
            self.params = in_params
            self.alphas = alpha_weights
        else:
            self.params = tf.reshape(in_params[:, :k * n_labels], (-1, n_labels, k))
            self.alphas = tf.nn.softmax(in_params[:, k * n_labels:k * n_labels + k])
            # alphas should be the rest of the output, trasform it so it so it sums to one
            # traniert nach erste batch net, da alphas zu groß oder klein werden => nan
            # self.alphas = tf.exp(in_params[:,k*n_labels:])
            # trainiert, aber net wirklich gut
            # self.alphas = tf.constant(1.0, shape=(self.alphas.shape))

        self.dists = tf.distributions.Bernoulli(logits=self.params)

        print(f"parameters: k: {k}, n_labels: {n_labels},  params shape: {self.params.shape}")
        print("alphas shape: ", self.alphas.shape)
        print("distr: ", self.dists)
        self.ll_output = None

    def forward(self, inp, _=None):
        inp = tf.expand_dims(inp, -1)
        print("input shape: ", inp.shape)
        self.ll_output = self.dists.log_prob(inp)
        print("log probs shape: ", self.ll_output.shape)

        # Log Likelihood of one mixture (shape: (num_batches, k)
        single_ll = tf.reduce_sum(self.ll_output, axis=1)
        print("LL per k ", single_ll.shape)

        # Add likelihoods of each mixture together after multiplying with α's
        print("α: ", self.alphas)
        eps = 1e-10
        full_ll = tf.reduce_logsumexp(single_ll + tf.log(self.alphas + eps), axis=-1, keepdims=True)
        print("log_likelihood full shape ", full_ll.shape)

        assert full_ll.shape == (inp.shape[0], 1), f"ll shape: {full_ll.shape} does not match (batch_size: {inp.shape[0]}, 1)"
        return full_ll

    def reconstruct_batch(self, feed_dict, sess):
        # Get all probabilities for returned from the nn and alpha values
        p, alpha_vals = sess.run([self.params, self.alphas], feed_dict=feed_dict)

        # get probabilities in normal domain
        probs = (1 / (1+np.exp(-p)))

        # Get most likely configurations
        configurations = np.round(probs)

        # Calculate probability of most likely event : max(p, 1-p)
        # np.max will not work, np.maximum takes two arrays and compares
        prob_mle = np.log(np.maximum(probs, 1.0 - probs))

        # Caculate ll for each of the k mixture densitys
        ll_single = np.sum(prob_mle, axis=1)
        ll_single = ll_single + np.log(alpha_vals)
        # print("ll shape ", ll_single.shape)
        # print("alphas: ", alpha_vals[0])
        # print(np.sum(alpha_vals[0]))
        # print(np.log(alpha_vals[0]))
        # print((ll_single * alpha_vals)[0])

        # get the highest ll in k options
        max_k = np.argmax(ll_single, axis=-1)
        # print(f"max k, {max_k}, shape: {max_k.shape}")

        # get the bast configuration for each batch
        max_configs = np.array([configurations[batch,:,k] for batch, k in enumerate(max_k)])
        # print("max den shape: ", np.array(max_densitys).shape)

        # Come back from log domain and round val
        # print("max configs", max_configs)
        # print("probbs: ", probs.shape)
        return max_configs


def build_nn_celeb_baseline(inp, output_shape):
    output_dims = 1
    for dim in output_shape[1:]:
        output_dims *= int(dim)
    print("nn output shape: ", output_shape)
    print("output_dims ", output_dims)
    with tf.variable_scope('nn'):
        linearized = build_feature_extraction_layers(inp)
        fc1 = tf.layers.dense(inputs=linearized,
                              units=output_dims*2,
                              activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1,
                              units=output_dims,
                              activation=None)
    print_num_vars()
    return fc2


def build_nn_celeb(inp, output_shape, num_sum_weights, num_leaf_weights):
    batch_size = int(inp.shape[0])
    output_shape = list(output_shape)
    output_dims = 1
    for dim in output_shape[1:]:
        output_dims *= int(dim)

    with tf.variable_scope('nn'):
        linearized = build_feature_extraction_layers(inp)
        linearized = tf.layers.dropout(linearized)
        sum_weights = tf.layers.dense(inputs=linearized,
                                      units=num_sum_weights,
                                      activation=None)
        leaf1 = tf.layers.dense(inputs=linearized,
                                units=output_dims * 4,
                                activation=tf.nn.relu)
        leaf_linear = tf.reshape(leaf1, [batch_size, output_dims, 4])
        leaf_weights = tf.layers.dense(inputs=leaf_linear,
                                units=num_leaf_weights,
                                activation=None)
    print_num_vars()
    print(leaf_weights, 'LEAF')
    return sum_weights, leaf_weights


def print_num_vars():
    all_vars = tf.trainable_variables()
    num_params = 0
    print('--- parameters ---')
    for var in all_vars:
        num_params += var.shape.num_elements()
        print(var, var.shape.num_elements())
    print('The neural network has {} parameters in total'.format(num_params))


