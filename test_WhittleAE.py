import numpy as np
import argparse
import sys
import time
import tensorflow as tf
from tensorflow.python.util import deprecation
from utils import create_rat_spn, load_mnist, encoder, decoder, load_model
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False

# parameter settings
path_base = "./data/WhittleAE"


def plot_Whittle_AE(test_decode, test_batch, k, test_llh1, test_llh2, flag):
    """Function to plot Whittle AE input and output, with Whittle likelihood

    Parameters
    ----------
    test_decode
        Whittle AE output
    test_batch
        input data
    k
        data index
    test_llh1
        Whittle likelihood from WSPN out 
    test_llh2
        Whittle likelihood from WSPN in 
    flag
        label of pos/neg/out
    Returns
    -------
    None

    """

    plt.figure(figsize=(5.8, 3))
    plt.subplots_adjust(left=0.0, bottom=0, top=0.88, right=1, wspace=0.01)
    # plot input and its Whittle likelihood
    plt.subplot(1, 2, 1) 
    im2 = test_batch[k, :].reshape(14, 14)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=1) 
    plt.axis('off') 
    plt.title('In LL: {:.1f}'.format(test_llh2[k][0]), fontsize=22)
    # plot output and its Whittle likelihood
    plt.subplot(1, 2, 2)
    im1 = test_decode[k, :].reshape(14, 14)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Out LL: {:.1f}'.format(test_llh1[k][0]), fontsize=22)
    # create path and save the plots
    save_path = "./WhittleAE_plots/" + flag + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(save_path + "/LL_" + flag + "_" + str(k) + ".pdf") 




def test_Whittle_AE(dim, x_train, x_test04, x_test59, x_out, x_test_label, sess):
    """Function to test Whittle Autoencoder (Whittle AE)

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
    x_out
        test data from Fashion-MNIST with labels 5-9
    x_test_label
        ?

    Returns
    -------
    None

    """
    # 0. Set dimensions and input
    image_size = dim * dim
    x_number = x_train.shape[0]
    fft_real_length = int(dim / 2) + 1 # T_W
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

    # # 2.1 DFT
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
    # 3.1 WSPN forward, likelihood, no marginalization
    wspn_output_in = wspn_in.forward(in_fft)
    wspn_output_ae = wspn_ae.forward(ae_fft)
    wspn_output_ae2 = wspn_ae.forward(in_fft)
    very_gen_loss_in = -1 * tf.reduce_mean(wspn_output_in)
    very_gen_loss_ae = -1 * tf.reduce_mean(wspn_output_ae)
    whittle_loss = very_gen_loss_in * 0.5 + very_gen_loss_ae * 0.5

    # 3.2 MSE loss
    mse_loss = tf.losses.mean_squared_error(inputs, decoder_op)

    # 3.3 KL divergence
    # # KL(P || Q) = \sum( P(x) * (log_P(x) - log_Q(x)))
    # # Here we use MCMC approximation to estimate the KL divergence,
    # # by assuming batch data are samples from the data distribution.
    # # More advanced and more efficient estimations will be followed in future work.
    kl_wspn = tf.reduce_mean(wspn_output_in - wspn_output_ae2)

    # # 3.4. Total loss
    loss = mse_loss + FLAGS.wspn * (whittle_loss + 0.001*kl_wspn)

    # 4. Load Whittle AE model
    load_model(sess, FLAGS.lr, FLAGS.wspn, FLAGS.epochs)

    # 5. Test Whittle AE
    print('start Whittle AE test...')

    # 5.1. Test on training data
    start_time = time.time()
    test_batch = x_train
    test_encode_llh1 = sess.run(wspn_output_ae, feed_dict={inputs: test_batch})
    test_encode_llh2 = sess.run(wspn_output_in, feed_dict={inputs: test_batch})
    print("Training data")
    print("LL_in = ", np.mean(test_encode_llh2))
    print("LL_ae = ", np.mean(test_encode_llh1))
    print("--- %s seconds ---" % (time.time() - start_time))

    # 5.2. Test on positive data
    start_time = time.time()
    test_batch = x_test04
    encode_label = x_test_label[x_test_label<5]
    test_decode_test = sess.run(decoder_op, feed_dict={inputs: test_batch})
    test_encode_llh1_test = sess.run(wspn_output_ae, feed_dict={inputs: test_batch})
    test_encode_llh2_test = sess.run(wspn_output_in, feed_dict={inputs: test_batch})
    print("Test data")
    print("LL_in = ", np.mean(test_encode_llh2_test))
    print("LL_ae = ", np.mean(test_encode_llh1_test))
    print("--- %s seconds ---" % (time.time() - start_time))
    # Plot positive data
    plt.rcParams.update({'figure.max_open_warning': 0})
    for k in range(100):
        plot_Whittle_AE(test_decode_test, test_batch, k, test_encode_llh1_test, test_encode_llh2_test, flag="pos")

    # 5.3. Test on outlier 1
    start_time = time.time()
    test_batch = x_test59
    encode_label = x_test_label[x_test_label>4]
    test_decode_neg = sess.run(decoder_op, feed_dict={inputs: test_batch})
    test_encode_llh1_neg = sess.run(wspn_output_ae, feed_dict={inputs: test_batch})
    test_encode_llh2_neg = sess.run(wspn_output_in, feed_dict={inputs: test_batch})
    print("Outlier 1")
    print("LL_in = ", np.mean(test_encode_llh2_neg))
    print("LL_ae = ", np.mean(test_encode_llh1_neg))
    print("--- %s seconds ---" % (time.time() - start_time))
    # Plot outlier 1
    for k in range(100):
        plot_Whittle_AE(test_decode_neg, test_batch, k, test_encode_llh1_neg, test_encode_llh2_neg, flag="neg")

    # 5.4. Test on outlier 2
    start_time = time.time()
    test_batch = x_out
    encode_label = x_test_label[x_test_label>4]
    test_decode_ood = sess.run(decoder_op, feed_dict={inputs: test_batch})
    test_encode_llh1_ood = sess.run(wspn_output_ae, feed_dict={inputs: test_batch})
    test_encode_llh2_ood = sess.run(wspn_output_in, feed_dict={inputs: test_batch})
    print("Outlier 2")
    print("LL_in = ", np.mean(test_encode_llh2_ood))
    print("LL_ae = ", np.mean(test_encode_llh1_ood))
    print("--- %s seconds ---" % (time.time() - start_time))
    # Plot outlier 2
    for k in range(100):
        plot_Whittle_AE(test_decode_ood, test_batch, k, test_encode_llh1_ood, test_encode_llh2_ood, flag="out")

    print('-' * 20, ' Standard exit ', '-' * 20)


def main(_):
    # resize images from 28x28 to 14x14, and load data
    dim = 14
    f_path = path_base + "/mnist"
    (train_im, train_label), (test_im, test_label) = load_mnist(f_path, dim)
    # training 
    train_04 = train_im[train_label < 5]
    # test
    test_04 = test_im[test_label < 5]
    # outlier 1
    test_59 = test_im[test_label > 4]
    # outlier 2
    f_path = path_base + "/fashion_mnist"
    (train_im, train_label), (test_im, test_label) = load_mnist(f_path, dim)
    test_out = test_im[test_label > 4]

    test_Whittle_AE(dim, train_04, test_04, test_59, test_out, test_label, sess=tf.Session())


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
