import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'axes.titlesize': 'small'})
from sklearn.datasets import load_digits
from lfw_dataset import load_lfw_dataset

data, attr = load_lfw_dataset(dimx=36, dimy=36)

# preprocess faces
data = np.float32(data) / 255.

IMG_SHAPE = data.shape[1:]
# plt.imshow(data[np.random.randint(data.shape[0])], cmap='gray', interpolation='none')
# plt.show()

import tensorflow as tf
import keras
from keras.models import Sequential
import keras.layers as L

CODE_SIZE = 256
generator = Sequential()
generator.add(L.InputLayer([CODE_SIZE], name='noise'))
generator.add(L.Dense(10*8*8, activation='elu', kernel_initializer="glorot_normal"))

generator.add(L.Reshape((8, 8, 10)))
generator.add(L.Deconv2D(64, kernel_size=[5, 5], activation='elu',kernel_initializer="glorot_normal"))
generator.add(L.Deconv2D(64, kernel_size=[5, 5], activation='elu',kernel_initializer="glorot_normal"))
generator.add(L.UpSampling2D(size=(2, 2)))
generator.add(L.Deconv2D(32, kernel_size=3, activation='elu',kernel_initializer="glorot_normal"))
generator.add(L.Deconv2D(32, kernel_size=3, activation='elu',kernel_initializer="glorot_normal"))
generator.add(L.Deconv2D(32, kernel_size=3, activation='elu',kernel_initializer="glorot_normal"))

generator.add(L.Conv2D(3, kernel_size=3, activation=None))
assert generator.output_shape[1:] == IMG_SHAPE
generator.summary()

discriminator = Sequential()
discriminator.add(L.InputLayer(IMG_SHAPE))

# discriminator.add(L.Conv2D(filters=32, kernel_size=[3, 3], kernel_initializer=tf.truncated_normal_initializer(), padding='same', activation='elu'))
# discriminator.add(L.MaxPooling2D(pool_size=[2, 2], padding='same'))
# discriminator.add(L.Conv2D(filters=64, kernel_size=[3, 3], kernel_initializer=tf.truncated_normal_initializer(), padding='same', activation='elu'))
# discriminator.add(L.MaxPooling2D(pool_size=[3, 3], padding='same'))
# discriminator.add(L.Conv2D(filters=128, kernel_size=[3, 3], kernel_initializer=tf.truncated_normal_initializer(), padding='same', activation='elu'))
# discriminator.add(L.MaxPooling2D(pool_size=[2, 2], padding='same'))
#
# discriminator.add(L.Flatten())
# discriminator.add(L.Dense(256, activation='tanh'))
# discriminator.add(L.Dense(2, activation=tf.nn.log_softmax))

discriminator.add(L.Convolution2D(filters=64, kernel_size=3, activation="selu", kernel_initializer="glorot_normal"))
discriminator.add(L.Convolution2D(filters=64, kernel_size=3, activation="selu", kernel_initializer="glorot_normal"))
discriminator.add(L.MaxPool2D())

discriminator.add(L.Convolution2D(filters=128, kernel_size=3, activation="selu", kernel_initializer="glorot_normal"))
discriminator.add(L.Convolution2D(filters=128, kernel_size=3, activation="selu", kernel_initializer="glorot_normal"))

discriminator.add(L.MaxPool2D())
discriminator.add(L.Convolution2D(filters=256, kernel_size=3, activation="selu", kernel_initializer="glorot_normal"))
discriminator.add(L.Convolution2D(filters=256, kernel_size=3, activation="selu", kernel_initializer="glorot_normal"))
discriminator.add(L.MaxPool2D())

discriminator.add(L.Flatten())
discriminator.add(L.Dense(256, activation='tanh'))
discriminator.add(L.Dense(2, activation=tf.nn.log_softmax))

noise = tf.placeholder('float32', [None, CODE_SIZE])
real_data = tf.placeholder('float32', [None, ] + list(IMG_SHAPE))

logp_real = discriminator(real_data)

generated_data = generator(noise)
logp_gen = discriminator(generated_data)

d_loss = -tf.reduce_mean(logp_real[:, 1] + logp_gen[:, 0])

# regularize
d_loss += tf.reduce_mean(discriminator.layers[-1].kernel**2)

# optimize
disc_optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(d_loss, var_list=discriminator.trainable_weights)

g_loss = -tf.reduce_mean(logp_gen[:, 1])
gen_optimizer = tf.train.AdamOptimizer(1e-4).minimize(g_loss, var_list=generator.trainable_weights)

s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

def sample_noise_batch(bsize):
    return np.random.normal(size=(bsize, CODE_SIZE)).astype('float32')

def sample_data_batch(bsize):
    idxs = np.random.choice(np.arange(data.shape[0]), size=bsize)
    return data[idxs]

def sample_images(nrow, ncol, sharp=False):
    images = generator.predict(sample_noise_batch(bsize=nrow*ncol))
    if np.var(images) != 0:
        images = images.clip(np.min(data), np.max(data))
    for i in range(ncol * nrow):
        plt.subplot(nrow, ncol, i+1)
        if sharp:
            plt.imshow(images[i].reshape(IMG_SHAPE), cmap='gray', interpolation='None')
        else:
            plt.imshow(images[i].reshape(IMG_SHAPE), cmap='gray')

    plt.show()

def sample_probas(bsize):
    plt.title('Generated vs real data')
    plt.hist(np.exp(discriminator.predict(sample_data_batch(bsize)))[:, 1],
             label='D(x)', alpha=0.5, range=[0, 1])
    plt.hist(np.exp((discriminator.predict(generator.predict(sample_noise_batch(bsize)))))[:, 1],
             label='D(G(z))', alpha=0.5, range=[0, 1])
    plt.legend(loc='best')
    plt.show()

from tqdm import tnrange

for epoch in tnrange(50000):

    feed_dict = {real_data:sample_data_batch(100), noise: sample_noise_batch(100)}

    for i in range(5):
        s.run(disc_optimizer, feed_dict)

    s.run(gen_optimizer, feed_dict)

    if epoch % 3000 == 0:
        sample_images(2, 3, True)
        sample_probas(1000)