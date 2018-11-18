import sys
sys.path.append("..")
import tensorflow as tf
import keras
import keras.layers as L
import keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from lfw_dataset import load_lfw_dataset
from matplotlib import pyplot as plt
from keras_tqdm import TQDMCallback

# !!! remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors!
def reset_tf_session():
    K.clear_session()
    tf.reset_default_graph()
    s = K.get_session()
    return s


X, attr = load_lfw_dataset(use_raw=True, dimx=32, dimy=32)
IMG_SHAPE = X.shape[1:]
print(IMG_SHAPE)

# center images
X = X.astype('float') / 255.0 - 0.5

# spilt
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))

# plt.title('sample images')
#
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     show_image(X[i])
# plt.show()

print("X shape:", X.shape)
print("attr shape:", attr.shape)

# del X
# import gc
# gc.collect()
#
# def build_pca_autoencoder(img_shape, code_size):
#
#     encoder = keras.models.Sequential()
#     encoder.add(L.InputLayer(img_shape))
#     encoder.add(L.Flatten())
#     encoder.add(L.Dense(code_size))
#
#     decoder = keras.models.Sequential()
#     decoder.add(L.InputLayer((code_size, )))
#     decoder.add(L.Dense(np.prod(img_shape)))
#     decoder.add(L.Reshape(img_shape))
#
#     return encoder, decoder
#
# s = reset_tf_session()
# encoder, decoder = build_pca_autoencoder(IMG_SHAPE, code_size=32)
#
# inp = L.Input(IMG_SHAPE)
# code = encoder(inp)
# reconstruction = decoder(code)
#
# autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
# autoencoder.compile(optimizer='adamax', loss='mse')
#
# autoencoder.fit(x=X_train, y=X_train, epochs=15, validation_data=[X_test, X_test],
#                 callbacks=[TQDMCallback()], verbose=0)


def visualize(img, encoder, decoder):
    """
    Draw original, encoded and decoded images
    """
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1, 3, 1)
    plt.title('Original')
    show_image(img)

    plt.subplot(1, 3, 2)
    plt.title('code')
    plt.imshow(code.reshape([code.shape[-1]//2, -1]))

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

# score = autoencoder.evaluate(X_test, X_test, verbose=0)
# print("PCA MSE:", score)
# for i in range(5):
#     img = X_test[i]
#     visualize(img, encoder, decoder)

def build_deep_autoencoder(img_shape, code_size):

    H, W, C = img_shape

    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(filters=32, kernel_size=[3, 3], activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=[3, 3], strides=2, padding='same'))
    encoder.add(L.Conv2D(filters=64, kernel_size=[3, 3], activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=[3, 3], strides=2, padding='same'))
    encoder.add(L.Conv2D(filters=128, kernel_size=[3, 3], activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=[3, 3], strides=2, padding='same'))
    encoder.add(L.Conv2D(filters=256, kernel_size=[3, 3], activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=[3, 3], strides=2, padding='same'))
    h, w, c = encoder.output_shape[1:]
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size, ))

    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size, )))
    decoder.add(L.Dense(h * w * c))
    decoder.add(L.Reshape((h, w, c)))
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=[3, 3], strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=[3, 3], strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=[3, 3], strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=3, kernel_size=[3, 3], strides=2, activation='elu', padding='same'))

    return encoder, decoder

s = reset_tf_session()
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
print(encoder.summary())
print(decoder.summary())

s = reset_tf_session()
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

model_filename = 'autoencoder.{0:03d}.hdf5'
last_finised_epoch = None

#### uncomment below to continue training from model checkpoint
#### fill `last_finished_epoch` with your latest finished epoch
# from keras.models import load_model
# s = reset_tf_session()
# last_finished_epoch = 4
# autoencoder = load_model(model_filename.format(last_finished_epoch))
# encoder = autoencoder.layers[1]
# decoder = autoencoder.layers[2]

autoencoder.fit(x=X_train, y=X_train, epochs=25, validation_data=[X_test, X_test],
                verbose=0,
                initial_epoch=last_finised_epoch or 0)

reconstruction_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
print("Convolutional autoencoder MSE:", reconstruction_mse)
for i in range(5):
    img = X_test[i]
    visualize(img, encoder, decoder)

encoder.save_weights("encoder.h5")
decoder.save_weights("decoder.h5")

s = reset_tf_session()
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
encoder.load_weights("encoder.h5")
decoder.load_weights("decoder.h5")

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

print(autoencoder.evaluate(X_test, X_test, verbose=0))
print(reconstruction_mse)