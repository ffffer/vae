import tensorflow as tf
import math
import numpy
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import pylab

print "Start loading MNIST data "
mndata = input_data.read_data_sets('data/mnist')
train_image = mndata.train.images # 60000 * 784
test_image = mndata.test.images # 10000 * 784
test_label = mndata.test.labels
print "Finished loading MNIST data"

n_hidden = 512
n_input = 784
batch_size = 50
l_number = 1
len_test = 10000
len_train = 60000
len_sample = 1000


def create_variables(scope, number_output):
    if number_output == 1:
        with tf.variable_scope(scope, reuse=False):
            weights = {
                '_h1': tf.get_variable(scope+"_h1", [2, n_hidden], dtype=tf.float64, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
                '_h_out': tf.get_variable(scope + "_h_out", [n_hidden, n_input], dtype=tf.float64, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            }
            biases = {
                '_b1': tf.get_variable(scope + "_b1", [n_hidden], dtype=tf.float64, initializer=tf.constant_initializer(0.0)),
                '_b_out': tf.get_variable(scope + "_b_out", [n_input], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
            }
    elif number_output == 2:
        with tf.variable_scope(scope, reuse=None):
            weights = {
                '_h1': tf.get_variable("_h1", [n_input, n_hidden], dtype=tf.float64, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
                '_h_out_sigma': tf.get_variable(scope + "_h_out_sigma", [n_hidden, 2], dtype=tf.float64, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
                '_h_out_mu': tf.get_variable(scope + "_h_out_mu", [n_hidden, 2], dtype=tf.float64, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            }
            biases = {
                '_b1': tf.get_variable(scope + "_b1", [n_hidden], dtype=tf.float64, initializer=tf.constant_initializer(0.0)),
                '_b_out_sigma': tf.get_variable(scope + "_b_out_sigma", [2], dtype=tf.float64, initializer=tf.constant_initializer(0.0)),
                '_b_out_mu': tf.get_variable(scope + "_b_out_mu", [2], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
            }
    return weights, biases

print "Start creating variables"
wake_weights, wake_biases = create_variables('w', 2)
sleep_weights, sleep_biases = create_variables('s', 1)
decoder_weights, decoder_biases = create_variables('d', 2)
encoder_weights, encoder_biases = create_variables('e', 1)
print "Finished creating variables"


def qzx(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['_h1']), biases['_b1'])
    layer_1 = tf.nn.relu(layer_1)

    out_layer_mu = tf.matmul(layer_1, weights['_h_out_mu']) + biases['_b_out_mu']
    out_layer_sigma = tf.matmul(layer_1, weights['_h_out_sigma']) + biases['_b_out_sigma']

    return out_layer_mu, out_layer_sigma


def pxz(z, weights, biases):
    layer_1 = tf.add(tf.matmul(z, weights['_h1']), biases['_b1'])
    layer_1 = tf.nn.relu(layer_1)

    out_layer = tf.matmul(layer_1, weights['_h_out']) + biases['_b_out']
    prob = tf.nn.sigmoid(out_layer)

    return prob, out_layer


def sample_z(shape, mu, sigma):
    epi = tf.random_normal(shape=shape,dtype=tf.float64)
    z = mu + tf.multiply(sigma, epi)

    return z


def sample_x(x):
    result = tf.contrib.distributions.Bernoulli(dtype=tf.float64, p=x).sample()
    return result


def normal_prob(x, mu, sigma, standard=False):
    if standard:
        mu = tf.convert_to_tensor([[0.0, 0.0]], dtype=tf.float64)
        sigma = tf.convert_to_tensor([[1.0, 1.0]], dtype=tf.float64)
    var = tf.square(sigma)
    x_minus_mu = x - mu
    x_sigma_x = tf.reduce_sum(tf.multiply(tf.square(x_minus_mu), var ** -1.0), 1)
    e = tf.exp(-0.5 * x_sigma_x)
    constant = 1.0 / (2.0 * math.pi) * (tf.reduce_prod(sigma, 1) ** -1.0)

    return tf.multiply(constant, e)


def log_normal_prob(x, mu, sigma):
    return -math.log(2*math.pi) - 0.5 * tf.log(tf.reduce_prod(tf.square(sigma), 1)) - 0.5 * (tf.reduce_sum(tf.multiply(tf.square(x-mu), (tf.square(sigma) ** -1.0)), 1))


def evaluate(x, p_weights, p_biases, q_weights, q_biases):
    z_mu_eval, z_sigma_eval = qzx(x, q_weights, q_biases)
    z_gen_eval = sample_z([len_sample, 2], z_mu_eval, z_sigma_eval)
    x_gen_prob_eval, x_gen_eval = pxz(z_gen_eval, p_weights, p_biases)
    p1 = tf.exp(tf.reduce_sum(-1.0 * tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_gen_eval), 1))
    p2 = normal_prob(z_gen_eval, [], [], standard=True)
    q = normal_prob(z_gen_eval, z_mu_eval, z_sigma_eval)
    p = tf.multiply(p1, p2)
    l_xi = tf.log(tf.reduce_mean(p / q))
    return l_xi


def plot(samples, name, dim):
    fig = plt.figure(figsize=(dim, dim))
    gs = gridspec.GridSpec(dim, dim)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    fig.savefig(name)
    plt.show()
    return fig


def plot_scatter(samples, label, name):
    samples = numpy.matrix.transpose(samples)
    plt.scatter(samples[0], samples[1], c=label)
    plt.legend()
    plt.savefig(name)
    plt.show()


def max_min(coor):
    coor = numpy.matrix.transpose(coor)
    x_min = numpy.min(coor[0])
    x_max = numpy.max(coor[0])
    y_min = numpy.min(coor[1])
    y_max = numpy.max(coor[1])

    x = numpy.linspace(start=x_min, stop=x_max, num=20)
    y = numpy.linspace(start=y_min, stop=y_max, num=20)
    res = []
    for i in range(20):
        for j in range(20):
            res.append([x[i], y[j]])

    return res


def run():
    x = tf.placeholder(tf.float64, [None, 784])
    z = tf.placeholder(tf.float64, [None, 2])

    # train wake sleep
    z_mu_w, z_sigma_w = qzx(x, wake_weights, wake_biases)
    z_gen_w = sample_z([1, 2], z_mu_w, z_sigma_w)
    x_gen_prob_w, x_gen_w = pxz(z_gen_w, sleep_weights, sleep_biases)
    cross_entropy_w = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_gen_w), 1)
    var_list_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='s')
    train_w = tf.train.AdamOptimizer().minimize(cross_entropy_w, var_list=var_list_w)

    # train sleep
    x_mu_prob_s, x_mu_s = pxz(z, sleep_weights, sleep_biases)
    x_gen_s = sample_x(x_mu_prob_s)
    z_mu_s, z_sigma_s = qzx(x_gen_s, wake_weights, wake_biases)
    likelihood = log_normal_prob(z, z_mu_s, z_sigma_s)
    var_list_s = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='w')
    train_s = tf.train.AdamOptimizer().minimize(-likelihood, var_list=var_list_s)

    # train vae
    z_mu_v, z_sigma_v = qzx(x, decoder_weights, decoder_biases)
    z_gen_v = sample_z([1, 2], z_mu_v, z_sigma_v)
    x_gen_prob_v, x_gen_v = pxz(z_gen_v, encoder_weights, encoder_biases)

    cross_entropy_v = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_gen_v), 1)
    kl = 0.5 * tf.reduce_sum(tf.square(z_sigma_v) + z_mu_v ** 2 - 1.0 - tf.log(tf.square(z_sigma_v)), 1)
    lower_bound = tf.reduce_mean(cross_entropy_v + kl)
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    train_e = tf.train.AdamOptimizer().minimize(lower_bound, var_list=var_list)

    # evaluation
    eval_vae = evaluate(x, encoder_weights, encoder_biases, decoder_weights, decoder_biases)
    eval_ws = evaluate(x, sleep_weights, sleep_biases, wake_weights, wake_biases)

    # plot
    x_vae, _ = pxz(z, encoder_weights, encoder_biases)
    x_ws, _ = pxz(z, sleep_weights, sleep_biases)

    # scatter
    z_mu_scatter_vae, z_sigma_scatter_vae = qzx(x, decoder_weights, decoder_biases)
    z_scatter_vae = sample_z([len_test, 2], z_mu_scatter_vae, z_sigma_scatter_vae)
    z_mu_scatter_ws, z_sigma_scatter_ws = qzx(x, wake_weights, wake_biases)
    z_scatter_ws = sample_z([len_test, 2], z_mu_scatter_ws, z_sigma_scatter_ws)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(100):
            for b in range(1200):
                batch = train_image[b * batch_size : b * batch_size + batch_size]

                _, lb = sess.run([train_e, lower_bound], feed_dict={x: batch})

                _, loss_w = sess.run([train_w, cross_entropy_w], feed_dict={x: batch})
                _, ml = sess.run([train_s, likelihood], feed_dict={z: tf.random_normal([1, 2]).eval()})

                if b % 100 == 0:
                    print "e : " + str(e) + " | b : " + str(b)
                    print lb
                    print tf.reduce_mean(loss_w).eval()
                    print ml
                    print "\n"

            if e % 10 == 0:
                print "start evaluation"
                eval_train_vae = 0.0
                eval_train_ws = 0.0
                eval_test_vae = 0.0
                eval_test_ws = 0.0

                for i in range(len_test):
                    eval_test_vae += sess.run(eval_vae, feed_dict={x: test_image[i:i + 1]}) / float(len_test)
                    eval_test_ws += sess.run(eval_ws, feed_dict={x: test_image[i:i + 1]}) / float(len_test)
                for i in range(len_train):
                    eval_train_vae += sess.run(eval_vae, feed_dict={x: train_image[i:i + 1]}) / float(len_train)
                    eval_train_ws += sess.run(eval_ws, feed_dict={x: train_image[i:i + 1]}) / float(len_train)

                print str(eval_train_vae) + "\t" + str(eval_test_vae) + "\t" + str(eval_train_ws) + "\t" + str(eval_test_ws)
                print "-----------------------------------------"

        sample_vae = sess.run(x_vae, feed_dict={z: tf.random_normal([100, 2]).eval()})
        sample_ws = sess.run(x_ws, feed_dict={z: tf.random_normal([100, 2]).eval()})
        plot(sample_vae, "vae_10_10", 10)
        plot(sample_ws, "ws_10_10", 10)

        z_sample_vae = sess.run(z_scatter_vae, feed_dict={x: test_image})
        z_sample_ws = sess.run(z_scatter_ws, feed_dict={x: test_image})
        plot_scatter(z_sample_vae, test_label, "vae_z")
        plot_scatter(z_sample_ws, test_label, "ws_z")

        x_sample_vae = sess.run(x_vae, feed_dict={z: max_min(z_sample_vae)})
        x_sample_ws = sess.run(x_ws, feed_dict={z: max_min(z_sample_ws)})
        plot(x_sample_vae, "vae_20_20", 20)
        plot(x_sample_ws, "ws_20_20", 20)

run()
