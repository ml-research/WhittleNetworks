"""
Train Whittle AE
Created on July 23, 2021

@author: Zhongjie Yu
"""
import numpy as np
import argparse
import sys
import csv
import time
import setproctitle
import tensorflow as tf
from tensorflow.python.util import deprecation
from utils import create_rat_spn, load_mnist, encoder, decoder, calc_loss, save_model 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False

# parameter settings
path_base = "./data/WhittleAE/"


def save_error(loss_list_all, loss_list_wspn, loss_list_mse, test_list_mse04, test_list_wspn04, test_list_mse59, test_list_wspn59):
    save_path = "./WhittleAE_results/model_lr_%.8f_w_%.8f/" % (FLAGS.lr, FLAGS.wspn)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save all loss
    file_ = save_path + 'loss_list_all.csv'
    with open(file_, 'w', newline='') as my_file:
        wr = csv.writer(my_file)
        wr.writerow(loss_list_all)
    # save WSPN_in loss
    file_ = save_path + 'loss_list_wspn.csv'
    with open(file_, 'w', newline='') as my_file:
        wr = csv.writer(my_file)
        wr.writerow(loss_list_wspn)
    # save mse loss
    file_ = save_path + 'loss_list_mse.csv'
    with open(file_, 'w', newline='') as my_file:
        wr = csv.writer(my_file)
        wr.writerow(loss_list_mse)
    # save test mse 04
    file_ = save_path + 'loss_test_mse04.csv'
    with open(file_, 'w', newline='') as my_file:
        wr = csv.writer(my_file)
        wr.writerow(test_list_mse04)
    # save test Whittle loss 04
    file_ = save_path + 'loss_test_wspn04.csv'
    with open(file_, 'w', newline='') as my_file:
        wr = csv.writer(my_file)
        wr.writerow(test_list_wspn04)
    # save test mse 59
    file_ = save_path + 'loss_test_mse59.csv'
    with open(file_, 'w', newline='') as my_file:
        wr = csv.writer(my_file)
        wr.writerow(test_list_mse59)
    # save test Whittle loss 59
    file_ = save_path + 'loss_test_wspn59.csv'
    with open(file_, 'w', newline='') as my_file:
        wr = csv.writer(my_file)
        wr.writerow(test_list_wspn59)


def train_Whittle_AE(dim, x_train, x_test_04, x_test_59, batch_size, sess):
    """Function to train Whittle Autoencoder (Whittle AE)

    Parameters
    ----------
    dim
        width and hight of the resized images
    x_train
        training data from MNIST with labels 0-4
    x_test_04
        test data from MNIST with labels 0-4
    x_test_59
        test data from MNIST with labels 5-9
    batch_size
        batch size in training

    Returns
    -------
    None

    """
    # 0. Set dimensions and input
    image_size = dim * dim
    x_number = x_train.shape[0]
    fft_real_length = int(dim/2) + 1 # T_W
    inputs = tf.placeholder(tf.float32, shape=[None, image_size])

    # 1. Set AutoEncoder layer
    with tf.variable_scope('AE'):
        encoder_op = encoder(inputs, dim, d_code=2)
        decoder_op = decoder(encoder_op, dim, d_code=2)

    # 2. Create WSPN
    # # 2.0. Initialise WSPN structure
    INPUT_SIZE = dim * fft_real_length  # t(real) + t(imag)
    OUTPUT_SIZE = 1  # likelihood
    print('creating spn')
    wspn_in = create_rat_spn(INPUT_SIZE, OUTPUT_SIZE, 'gaussian2d', spn_name='spn_2d_in', spn_depth=7)
    wspn_ae = create_rat_spn(INPUT_SIZE, OUTPUT_SIZE, 'gaussian2d', spn_name='spn_2d_ae', spn_depth=7)

    # # 2.1. DFT
    # # DFT from input, for WSPN_input | from symmetry, use rfft
    in_fft_cut = tf.signal.rfft(tf.reshape(inputs, (-1, dim, dim)))
    in_fft_real = tf.reshape(tf.real(in_fft_cut), [-1, dim * fft_real_length, 1])
    in_fft_imag = tf.reshape(tf.imag(in_fft_cut), [-1, dim * fft_real_length, 1])
    in_fft = tf.concat([in_fft_real, in_fft_imag], 2)
    # # DFT from auto encoder, for WSPN_output | from symmetry, use rfft
    ae_fft_cut = tf.signal.rfft(tf.reshape(decoder_op, (-1, dim, dim)))
    ae_fft_real = tf.reshape(tf.real(ae_fft_cut), [-1, dim * fft_real_length, 1])
    ae_fft_imag = tf.reshape(tf.imag(ae_fft_cut), [-1, dim * fft_real_length, 1])
    ae_fft = tf.concat([ae_fft_real, ae_fft_imag], 2)

    # 3. Losses
    # # 3.1. WSPN forward, likelihood, no marginalization
    wspn_output_in = wspn_in.forward(in_fft)
    wspn_output_ae = wspn_ae.forward(ae_fft)
    wspn_output_ae2 = wspn_ae.forward(in_fft)
    very_gen_loss_in = -1 * tf.reduce_mean(wspn_output_in)
    very_gen_loss_ae = -1 * tf.reduce_mean(wspn_output_ae)
    whittle_loss = very_gen_loss_in * 0.5 + very_gen_loss_ae * 0.5

    # # 3.2. MSE loss
    mse_loss = tf.losses.mean_squared_error(inputs, decoder_op)

    # # 3.3. KL divergence
    # # KL(P || Q) = \sum( P(x) * (log_P(x) - log_Q(x)))
    # # Here we use MCMC approximation to estimate the KL divergence,
    # # by assuming batch data are samples from the data distribution.
    # # More advanced and more efficient estimations will be followed in future work.
    kl_wspn = tf.reduce_mean(wspn_output_in - wspn_output_ae2)
    
    # # 3.4. Total loss
    loss = mse_loss + FLAGS.wspn * (whittle_loss + 0.001*kl_wspn)

    # 4. Optimizer
    optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    train_op = optimizer.minimize(loss)
    sess.run(tf.global_variables_initializer())

    # 5. Train Whittle AE
    loss_list_all = []
    loss_list_wspn = []
    loss_list_wspn2= []
    loss_list_kl = []
    loss_list_mse = []
    test_list_mse04 = []
    test_list_wspn04 = []
    test_list_mse59 = []
    test_list_wspn59 = []

    batches_per_epoch = int(x_number / batch_size)
    print('start training')
    input_train = x_train.copy()
    for i in range(FLAGS.epochs + 1):
        proc_name = "WhittleAE:%d:%d-mlp2d-lr%.8fw%.8f" % (i, FLAGS.epochs, FLAGS.lr, FLAGS.wspn)
        setproctitle.setproctitle(proc_name)
        start_time = time.time()
        # train one epoch, first shuffle training data
        np.random.shuffle(x_train)
        # per batch
        for j in range(batches_per_epoch):
            im_batch = x_train[j * batch_size: (j + 1) * batch_size, :]
            sess.run(train_op, feed_dict={inputs: im_batch})
        # last batch
        if x_number % batch_size != 0:
            im_batch = x_train[batches_per_epoch * batch_size:, :]
            sess.run(train_op, feed_dict={inputs: im_batch})
        # print some losses
        if i % 2 == 0:
            # measure epoch time
            end_time = time.time()
            epoch_time = int(end_time - start_time)
            loss_mse = calc_loss(mse_loss, inputs, x_train, batch_size, sess)
            loss_whittle = calc_loss(whittle_loss, inputs, x_train, batch_size, sess)
            loss_spn_in = calc_loss(very_gen_loss_in, inputs, x_train, batch_size, sess)
            loss_spn_ae = calc_loss(very_gen_loss_ae, inputs, x_train, batch_size, sess)
            loss_spn_kl = calc_loss(kl_wspn, inputs, x_train, batch_size, sess)
            cur_loss = loss_mse + FLAGS.wspn * (loss_whittle + 0.001*loss_spn_kl)

            print('Epoch:', i, 'Loss:', cur_loss, '=mse:', loss_mse, '+wspn*(', loss_whittle, '+wkl*', loss_spn_kl, '))')
            print('Epoch:', i, ' Time:', epoch_time, 's/epoch')
            print('Epoch:', i, 'spn_llh_in:', -loss_spn_in, '; spn_llh_ae', -loss_spn_ae, '; spn_KL', loss_spn_kl)
            # store loss for saving
            loss_list_all.append(cur_loss)
            loss_list_wspn.append(-loss_spn_in)
            loss_list_wspn2.append(-loss_spn_ae)
            loss_list_mse.append(loss_mse)
            loss_list_kl.append(loss_spn_kl)

            # evaluate loss for test data and outlier1
            test_mse04 = calc_loss(mse_loss, inputs, x_test_04, batch_size, sess)
            test_mse59 = calc_loss(mse_loss, inputs, x_test_59, batch_size, sess)
            test_whittle04 = calc_loss(whittle_loss, inputs, x_test_04, batch_size, sess)
            test_whittle59 = calc_loss(whittle_loss, inputs, x_test_59, batch_size, sess)

            print('Test mse04:', test_mse04, ', Test whittle:', -test_whittle04)
            print('Test mse59:', test_mse59, ', Test whittle:', -test_whittle59)
            # store loss for saving
            test_list_mse04.append(test_mse04)
            test_list_wspn04.append(test_whittle04)
            test_list_mse59.append(test_mse59)
            test_list_wspn59.append(test_whittle59)

            # save temporal results and model
            print("Saving SPN model after %d epochs" % i)
            save_model(sess, FLAGS.lr, FLAGS.wspn, i)
            # save test error with #epoch
            save_error(loss_list_all, loss_list_wspn, loss_list_mse, test_list_mse04, test_list_wspn04, test_list_mse59, test_list_wspn59)

    print('-' * 20, ' Standard exit ', '-' * 20)


def main(_):
    # resize images from 28x28 to 14x14, and load data
    dim = 14
    f_path = path_base + "mnist"
    (train_im, train_label), (test_im, test_label) = load_mnist(f_path, dim)
    # training 
    train_04 = train_im[train_label<5]
    # test
    test_04 = test_im[test_label<5]
    # outlier 1
    test_59 = test_im[test_label>4]

    train_Whittle_AE(dim, train_04, test_04, test_59, batch_size=FLAGS.batch, sess=tf.Session())


if __name__ == '__main__':
    np.random.seed(123)
    parser = argparse.ArgumentParser()
    # Args go here
    parser.add_argument('--lr', type=float, default=0.00422,
                        help='Learning Rate in optimization')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of minibatch steps to do')
    parser.add_argument('--wspn', type=float, default=0.000005,
                        help='Learning Rate in optimization')
    parser.add_argument('--batch', type=int, default=256,
                        help='Number of minibatch steps to do')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
