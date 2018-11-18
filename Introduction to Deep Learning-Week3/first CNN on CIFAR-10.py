import sys
sys.path.append("..")
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)

NUM_CLASSES = 10
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# show random images from train
# cols = 8
# rows = 2
# fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
# for i in range(cols):
#     for j in range(rows):
#         random_index = np.random.randint(0, len(y_train))
#         ax = fig.add_subplot(rows, cols, i * rows + j + 1)
#         ax.grid('off')
#         ax.axis('off')
#         ax.imshow(x_train[random_index, :])
#         ax.set_title(cifar10_classes[y_train[random_index, 0]])
# plt.show()

# normalize inputs
x_train2 = x_train / 255. - 0.5
x_test2 = x_test / 255 - 0.5
# convert class labels to one-hot encoded, should have shape (?, NUM_CLASSES)
y_train2 = keras.utils.to_categorical(y_train, NUM_CLASSES)
print(y_train2.shape)
y_test2 = keras.utils.to_categorical(y_test, NUM_CLASSES)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU

def make_model():

    model = Sequential()
    # stage1
    model.add(Conv2D(filters=16, kernel_size=[3, 3], padding='same', input_shape=(32, 32, 3)))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(filters=32, kernel_size=[3, 3], padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    # stage2
    model.add(Conv2D(filters=32, kernel_size=[3, 3], padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(filters=64, kernel_size=[3, 3], padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    # stage 3
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# describle model
K.clear_session() # clear default graph
model = make_model()
model.summary()

INIT_LR = 5e-3 # initial learning rate
BATCH_SIZE = 32
EPOCHS = 10

K.clear_session() # clear default graph
# don't call K.set_learning_phase() !!! (otherwise will enable dropout in train/test simultaneously)
model = make_model() # define our model

# prepare model for fitting (loss, optimizer, etc)
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
    metrics=['accuracy'] # report accuracy during training
)

# schedeler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch

# callback for printing of actual learning rate used by optimizer
class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Learning rate:", K.get_value(model.optimizer.lr))

model_filenames = 'cifar.{0:03d}.hdf5'
last_finised_epoch = 10

#### uncomment below to continue training from model checkpoint
#### fill `last_finished_epoch` with your latest finished epoch
# from keras.models import load_model
# s = reset_tf_session()
# last_finished_epoch = 7
# model = load_model(model_filename.format(last_finished_epoch))


# fit model
# model.fit(
#     x_train2, y_train2, # prepared data
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     # callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler),
#     #            LrHistory(),
#     #            keras_utils.Tqm]
#     validation_data=(x_test2, y_test2),
#     shuffle=True,
#     verbose=0,
#     initial_epoch=last_finised_epoch or 0
# )
# model.save_weights("weights.h5")
model.load_weights("weights.h5")

# make test predictions
y_pred_test = model.predict_proba(x_test2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probs = np.max(y_pred_test, axis=1)

#confusion matrix anc accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
plt.figure(figsize=(7, 6))
plt.title("Confusion matrix", fontsize=16)
plt.imshow(confusion_matrix(y_test, y_pred_test_classes))
plt.xticks(np.arange(10), cifar10_classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), cifar10_classes, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(y_test, y_pred_test_classes))

# inspect predictiono
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_test))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_test[random_index, :])
        pred_label = cifar10_classes[y_pred_test_classes[random_index]]
        pred_proba = y_pred_test_max_probs[random_index]
        true_label = cifar10_classes[y_test[random_index, 0]]
        ax.set_title("pred: {}\nscore: {:.3}\ntrue: {}".format(pred_label, pred_proba, true_label))

plt.show()

## Visuslize maximum stimuli
"""
We want to find input images that provide maximum activations for particular layers of our network
We will find those maximum stimuli via gradient descent in image space
For that task we load our model weights, calculate the layer output gradient with respect to
image input and shift input image in that direction
"""

K.clear_session()  # clear default graph
K.set_learning_phase(0) # disable dropout
model = make_model()
model.load_weights("weights.h5") # that were saved after model.fit

# all weights we have
model.summary()

def find_maximum_stimuli(layer_name, is_conv, filter_index, model, iterations=20, step=1., verbose=True):

    def image_values_to_rgb(x):
        # normalize x: center on 0 (np.mean(x_train2)), ensure std is 0.25 (np.std(x_train2))
        # so that it looks like a normalized image input for our network
        x = (x - np.mean(x_train2)) / np.std(x_train2)

        # do reverse normalization to RGB values: x = (x_norm + 0.5) * 255
        x = (x + 0.5) * 255

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # this is the placeholder for the input image
    input_img = model.input
    img_width, img_height = input_img.shape.as_list()[1:3]

    # find the layer output by name
    layer_output = list(filter(lambda x: x.name == layer_name, model.layers))[0].output

    # we will build a loss function that maximizes the activation
    # of the filter_index filter of the layer considered
    if is_conv:
        # mean over feature map values for convolutional layer
        loss = K.mean(layer_output[:, :, :, filter_index])
    else:
        loss = K.mean(layer_output[:, filter_index])

    # we compute the gradient of the loss wrt input image
    grads = K.gradients(loss, input_img)[0] # [0] because of the batch dimension

    # normalization trick: we normalize the gradient
    grads = grads / (K.sqrt(K.sum(K.sqrt(grads))) + 1e-10)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * (0.1 if is_conv else 0.001)

    # we run gradient ascent
    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        if verbose:
            print('Current loss value:', loss_value)

    # decode the resulting input image
    img = image_values_to_rgb(input_img_data[0])

    return img, loss_value

# sample maximum stimuli
def plot_filters_stimuli(layer_name, is_conv, model, iterations=20, step=1., verbose=False):

    cols = 8
    rows = 2
    filter_index = 0
    max_filter_index = list(filter(lambda x: x.name == layer_name, model.layers))[0].output.shape.as_list()[-1] - 1
    fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            if filter_index <= max_filter_index:
                ax = fig.add_subplot(rows, cols, i * rows + j + 1)
                ax.grid('off')
                ax.axis('off')
                loss = -1e20
                while loss < 0 and filter_index <= max_filter_index:
                    stimuli, loss = find_maximum_stimuli(layer_name, is_conv, filter_index, model, iterations, step, verbose=verbose)
                    filter_index += 1

                if loss > 0:
                    ax.imshow(stimuli)
                    ax.set_title("Filter #{}".format(filter_index))
    plt.show()


# maximum stimuli for convolutional neurons
conv_activation_layers = []
for layer in model.layers:
    if isinstance(layer, LeakyReLU):
        prev_layer = layer.inbound_nodes[0].inbound_layers[0]
        if isinstance(prev_layer, Conv2D):
            conv_activation_layers.append(layer)

for layer in conv_activation_layers:
    print(layer.name)
    plot_filters_stimuli(layer_name=layer.name, is_conv=True, model=model)

# maximum stimuli for last dense layer
last_dense_layer = list(filter(lambda x: isinstance(x, Dense), model.layers))[-1]
plot_filters_stimuli(layer_name=last_dense_layer.name, is_conv=False, iterations=200, step=0.1, model=model)

# K.clear_session()  # clear default graph
# K.set_learning_phase(0)  # disable dropout
# model = make_model()
# model.load_weights("weights.h5")  # that were saved after model.fit
# # all weights we have
# model.summary()
#
#
# def find_maximum_stimuli(layer_name, is_conv, filter_index, model, iterations=20, step=1., verbose=True):
#     def image_values_to_rgb(x):
#         # normalize x: center on 0 (np.mean(x_train2)), ensure std is 0.25 (np.std(x_train2))
#         # so that it looks like a normalized image input for our network
#         x = (x - np.mean(x_train2)) / np.std(x_train2)
#
#         # do reverse normalization to RGB values: x = (x_norm + 0.5) * 255
#         x = (x + 0.5) * 255
#
#         # clip values to [0, 255] and convert to bytes
#         x = np.clip(x, 0, 255).astype('uint8')
#         return x
#
#     # this is the placeholder for the input image
#     input_img = model.input
#     img_width, img_height = input_img.shape.as_list()[1:3]
#
#     # find the layer output by name
#     layer_output = list(filter(lambda x: x.name == layer_name, model.layers))[0].output
#
#     # we build a loss function that maximizes the activation
#     # of the filter_index filter of the layer considered
#     if is_conv:
#         # mean over feature map values for convolutional layer
#         loss = K.mean(layer_output[:, :, :, filter_index])
#     else:
#         loss = K.mean(layer_output[:, filter_index])
#
#     # we compute the gradient of the loss wrt input image
#     grads = K.gradients(loss, input_img)[0]  # [0] because of the batch dimension!
#
#     # normalization trick: we normalize the gradient
#     grads = grads / (K.sqrt(K.sum(K.square(grads))) + 1e-10)
#
#     # this function returns the loss and grads given the input picture
#     iterate = K.function([input_img], [loss, grads])
#
#     # we start from a gray image with some random noise
#     input_img_data = np.random.random((1, img_width, img_height, 3))
#     input_img_data = (input_img_data - 0.5) * (0.1 if is_conv else 0.001)
#
#     # we run gradient ascent
#     for i in range(iterations):
#         loss_value, grads_value = iterate([input_img_data])
#         input_img_data += grads_value * step
#         if verbose:
#             print('Current loss value:', loss_value)
#
#     # decode the resulting input image
#     img = image_values_to_rgb(input_img_data[0])
#
#     return img, loss_value
# # sample maximum stimuli
# def plot_filters_stimuli(layer_name, is_conv, model, iterations=20, step=1., verbose=False):
#     cols = 8
#     rows = 2
#     filter_index = 0
#     max_filter_index = list(filter(lambda x: x.name == layer_name, model.layers))[0].output.shape.as_list()[-1] - 1
#     fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
#     for i in range(cols):
#         for j in range(rows):
#             if filter_index <= max_filter_index:
#                 ax = fig.add_subplot(rows, cols, i * rows + j + 1)
#                 ax.grid('off')
#                 ax.axis('off')
#                 loss = -1e20
#                 while loss < 0 and filter_index <= max_filter_index:
#                     stimuli, loss = find_maximum_stimuli(layer_name, is_conv, filter_index, model,
#                                                          iterations, step, verbose=verbose)
#                     filter_index += 1
#                 if loss > 0:
#                     ax.imshow(stimuli)
#                     ax.set_title("Filter #{}".format(filter_index))
#     plt.show()
# # maximum stimuli for convolutional neurons
# conv_activation_layers = []
# for layer in model.layers:
#     if isinstance(layer, LeakyReLU):
#         prev_layer = layer.inbound_nodes[0].inbound_layers[0]
#         if isinstance(prev_layer, Conv2D):
#             conv_activation_layers.append(layer)
#
# for layer in conv_activation_layers:
#     print(layer.name)
#     plot_filters_stimuli(layer_name=layer.name, is_conv=True, model=model)
# # maximum stimuli for last dense layer
# last_dense_layer = list(filter(lambda x: isinstance(x, Dense), model.layers))[-1]
# plot_filters_stimuli(layer_name=last_dense_layer.name, is_conv=False,
#                      iterations=200, step=0.1, model=model)