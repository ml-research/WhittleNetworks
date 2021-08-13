import tensorflow as tf
from observations import mnist
from skimage.transform import resize
import sys
sys.path.append('./SPFlow/src/')
import spn.experiments.RandomSPNs.RAT_SPN as RAT_SPN
import spn.experiments.RandomSPNs.region_graph as region_graph
import os


def create_rat_spn(input_size, output_size, leaf_type, spn_name=None, spn_depth=4):
    # Create SPN
    # region graph for RAT-SPN
    # details see: 
    # https://github.com/SPFlow/SPFlow/blob/master/src/spn/experiments/RandomSPNs/train_mnist.py 
    # C=inputsize
    rg = region_graph.RegionGraph(range(input_size))
    # range(0,2) ==> R=2
    for _ in range(0, 2):
        # partition=2, Depth=4
        rg.random_split(2, spn_depth)

    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.leaf = leaf_type  # gaussian, 2d gaussian, or Bernoulli
    print("leaf {}".format(args.leaf))
    # S=2?
    args.num_sums = 2
    args.gauss_min_var = 0.001
    args.gauss_max_var = 1
    # args.num_gauss = 2
    # C = output_size
    spn = RAT_SPN.RatSpn(output_size, region_graph=rg, name=spn_name, args=args)
    print("num_params=", spn.num_params())

    return spn


def load_mnist(f_path, p):
    # Function to load (Fashion-) MNIST dataset
    # f_path: path of the dataset
    # p:      resize image from 28x28 to pxp
    (train_im, train_lab), (test_im, test_lab) = mnist(f_path)
    # re-size the images from 28*28 to smaller size
    train_im = resize(train_im.reshape(-1, 28, 28), (60000, p, p)).reshape(60000, -1)
    test_im = resize(test_im.reshape(-1, 28, 28), (10000, p, p)).reshape(10000, -1)
    # normalize to [0,1]
    train_im /= 255.0
    test_im /= 255.0
    return (train_im, train_lab), (test_im, test_lab)


def encoder(x, dim, d_code=8):
    # hidden layer settings
    n_input = int(dim * dim)
    n_hidden_1 = 128
    n_hidden_2 = 64
    n_hidden_3 = 16
    n_hidden_4 = d_code
    # weights and bias
    weights = {
        'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
        'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
        'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
        'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),
        }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
        }
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                     biases['encoder_b4'])
    return layer_4


def decoder(x, dim, d_code=8):
    # hidden layer settings
    n_input = int(dim * dim)
    n_hidden_1 = 128
    n_hidden_2 = 64
    n_hidden_3 = 16
    n_hidden_4 = d_code
    # weights and bias
    weights = {
        'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
        'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
        'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
        'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
        }
    biases = {
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b4': tf.Variable(tf.random_normal([n_input])),
        }
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    return layer_4


def calc_loss(loss, inputs, data, batch_size, sess):
    batches_per_epoch = int(data.shape[0] / batch_size)
    loss_out = 0.0
    # normal batch
    for j in range(batches_per_epoch):
        im_batch = data[j * batch_size: (j + 1) * batch_size, :]
        loss_out += sess.run(loss, feed_dict={inputs: im_batch})
    # last batch
    if data.shape[0] % batch_size != 0:
        im_batch = data[batches_per_epoch * batch_size:, :]
        loss_out += sess.run(loss, feed_dict={inputs: im_batch})

    return loss_out/batches_per_epoch


def save_model(sess, lr, wspn, i):
    save_path = "./WhittleAE_results/model_lr_%.8f_w_%.8f/iter_%d/" % (lr, wspn, i)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_name = save_path + "model.ckpt"
    saver = tf.train.Saver()
    saver.save(sess, model_name)


def load_model(sess, lr, wspn, i):
    model_name = "./WhittleAE_results/model_lr_%.8f_w_%.8f/iter_%d/model.ckpt" % (lr, wspn, i)
    saver = tf.train.Saver()
    saver.restore(sess, model_name)




