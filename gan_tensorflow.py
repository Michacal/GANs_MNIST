import os, time, itertools, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# G(z)
def generator(x):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)
    # 1st hidden layer
    w0 = tf.get_variable('G_w0', [x.get_shape()[1], 128], initializer=w_init)
    b0 = tf.get_variable('G_b0', [128], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    # output hidden layer
    w1 = tf.get_variable('G_w1', [h0.get_shape()[1], 784], initializer=w_init)
    b1 = tf.get_variable('G_b1', [784], initializer=b_init)
    o = tf.nn.tanh(tf.matmul(h0, w1) + b1)
    return o
    ### Code:ToDO( Change the architecture as CW2 Guidance required)


# D(x)
def discriminator(x, drop_out):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)
    # 1st hidden layer
    w0 = tf.get_variable('D_w0', [x.get_shape()[1], 784], initializer=w_init)
    b0 = tf.get_variable('D_b0', [784], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    # output layer
    w1 = tf.get_variable('D_w1', [h0.get_shape()[1], 1], initializer=w_init)
    b1 = tf.get_variable('D_b1', [1], initializer=b_init)
    o = tf.sigmoid(tf.matmul(h0, w1) + b1)
    ###  Code: ToDO( Change the architecture as CW2 Guidance required)

    return o

def show_result(num_epoch, show = False, save = False, path = 'results.png'):
    z_ = np.random.normal(0, 1, (25, 100))    # z_ is the input of generator, every epochs will random produce input
    ##Code:ToDo complete the rest of part
    samples = []
    gen_sample = sess.run(G_z, feed_dict={z: z_})
    samples.append(gen_sample)
    samples = np.array(samples)
    print('samples of G: ', samples.shape)
    samples = np.squeeze(samples)
    print('samples of G: ', samples.shape)
    # f, a = plt.subplots(5, 5, figsize=(5, 5))
    # for i in range(5):
    #     for j in range(5):
    #         a[i][j].imshow(np.reshape(samples[i], (28, 28)))

    rand_sample = np.reshape(samples[2], (28, 28))
    # plt.imshow(rand_sample)
    # plt.show()
    #plt.plot(np.reshape(samples[2], (28, 28)))

    if show:
        f, a = plt.subplots(5, 5, figsize=(5, 5))
        for i in range(5):
            for j in range(5):
                a[i][j].imshow(np.reshape(samples[i], (28, 28)))
        #f.show()
        plt.show()

    if save:
        #plt.savefig(path)
        plt.imsave(path, rand_sample)
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 100
samples = []
losses = []

# load MNIST
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1
print('mnist images: ', train_set.shape)
plt.imshow(train_set[550].reshape((28, 28)))
plt.show()

# networks : generator
# the two networks interact with each other share created variables
with tf.variable_scope('G'):
    # 100 random pixels from the actual image
    z = tf.placeholder(tf.float32, shape=(None, 100))
    G_z = generator(z)
# networks : discriminator
with tf.variable_scope('D') as scope:
    drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
    # real images
    x = tf.placeholder(tf.float32, shape=(None, 784))
    # learn from actual images
    D_real = discriminator(x, drop_out)
    # same variable is reused --> to avoid value errors
    scope.reuse_variables()
    # identify fake image
    D_fake = discriminator(G_z, drop_out)


# loss for each network
eps = 1e-2
# computes the mean of elements across dimensions of a tensor
# D_loss = D_real + D_fake: log(D(x)) + log((1- D(G(z)))
D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
# G_loss = log(D(G(z)))
G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

# trainable variables for each network
t_vars = tf.trainable_variables()  #returns all variables created in the two variable scopes and makes trainable=True
D_vars = [var for var in t_vars if 'D_' in var.name]
G_vars = [var for var in t_vars if 'G_' in var.name]

# optimizer for each network
D_optim = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
G_optim = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.initialize_all_variables().run()

# results save in folder
if not os.path.isdir('MNIST_GAN_results'):
    os.mkdir('MNIST_GAN_results')
if not os.path.isdir('MNIST_GAN_results/results'):
    os.mkdir('MNIST_GAN_results/results')
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []


# training-loop
np.random.seed(int(time.time()))
start_time = time.time()
for epoch in range(train_epoch):
    # initialize losses for both networks
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(train_set.shape[0] // batch_size):
        # update discriminator
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 100))
        # run the optimizer
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: 0.3})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 100))
        # run optimizer
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: 0.3})
        G_losses.append(loss_g_)


    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))


    ### Code: TODO Code complet show_result function)

    p = 'MNIST_GAN_results/results/MNIST_GAN_' + str(epoch + 1) + '.png'
    show_result((epoch + 1), show=False, save=True, path=p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
tf.logging.set_verbosity(old_v)
with open('MNIST_GAN_results/train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)
# with open('MNIST_GAN_results/train_samples.pkl', 'rb') as f:
#     samples = pickle.load(f)
with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')

# with open('MNIST_GAN_results/train_hist.pkl', 'rb') as f:
#     samples = pickle.load(f)
# print(samples)
images = []
sess.close()