import sys
sys.path.append("..")
import keras.backend as K
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import scipy.io
import os
import tarfile


# remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors!
def reset_tf_session():
    K.clear_session()
    tf.reset_default_graph()
    s = K.get_session()
    return s

# we will crop and resize input images to IMG_SIZE x IMG_SIZE
IMG_SIZE = 250
def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1) # 1 maybe means color image, 0 maybe means grayscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # for plt show
    return img

def image_center_crop(img):
    """
    Makes a square center crop of an img, which is a [h, w, 3] numpy array.
    Returns [min(h, w), min(h, w), 3] output with same width and height
    For cropping use numpy slicing
    """

    h, w = img.shape[0], img.shape[1]
    if w > h:
        cropped_img = img[:, (w - h) // 2 : (w - h) // 2 + h, :]
    else:
        cropped_img = img[(h - w) // 2 : (h - w) // 2 + w, :, :]

    return cropped_img

def prepare_raw_bytes_for_model(raw_bytes, normalize_for_model=True):
    img = decode_image_from_raw_bytes(raw_bytes) # decode image raw bytes to matrix
    img = image_center_crop(img)  # take squared center crop
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # resize for our model
    if normalize_for_model:
        img = img.astype("float32") # prepare for normalization
        img = keras.applications.inception_v3.preprocess_input(img) # normalize for model

    return img

# reads bytes directly from tar by filename (slow, but ok for testing, take about 6 sec)
def read_raw_from_tar(tar_fn, fn):
    with tarfile.open(tar_fn) as f:
        m = f.getmember(fn)
        return f.extractfile(m).read()

# test cropping
# raw_bytes = read_raw_from_tar("102flowers.tgz", "jpg/image_00001.jpg")

# img = decode_image_from_raw_bytes(raw_bytes)
# print(img.shape)
# plt.imshow(img)
# plt.show()
#
# img = prepare_raw_bytes_for_model(raw_bytes, normalize_for_model=False)
# print(img.shape)
# plt.imshow(img)
# plt.show()
# read all filenames and labels for them
# read filenames direactly from tar
def get_all_filenames(tar_fn):
    with tarfile.open(tar_fn) as f:
        return [m.name for m in f.getmembers() if m.isfile()]

all_files = sorted(get_all_filenames("102flowers.tgz")) # list all files in tar sorted by name, namely X
all_labels = scipy.io.loadmat('imagelabels.mat')['labels'][0] - 1 # read class labels (0, 1, 2, ...), namely Y
# all_files and all_labels are aligned now
N_CLASSES = len(np.unique(all_labels))
print(N_CLASSES)

# split into train/test
tr_files, te_files, tr_labels, te_labels = train_test_split(all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

# will yield raw image bytes from tar with corresponding label
def raw_generator_with_label_from_tar(tar_fn, files, labels):
    label_by_fn = dict(zip(files, labels)) # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    with tarfile.open(tar_fn) as f:
        while True:
            m = f.next()
            if m is None:
                break
            if m.name in label_by_fn:
                print(label_by_fn[m.name])
                yield f.extractfile(m).read(), label_by_fn[m.name]

# batch generator
BATCH_SIZE = 32
def batch_generator(items, batch_size):
    """
        Implement batch generator that yields items in batches of size batch_size.
        There's no need to shuffle input items, just chop them into batches.
        Remember about the last batch that can be smaller than batch_size!
        Input: any iterable (list, generator, ...). You should do `for item in items: ...`
            In case of generator you can pass through your items only once!
        Output: In output yield each batch as a list of items.
    """
    minibatch = []
    cnt = 0
    for item in items:
        minibatch.append(item)
        cnt = cnt + 1
        if (cnt == batch_size):
            yield minibatch
            minibatch = []
            cnt = 0
    if cnt != 0:
        yield minibatch

def train_generator(files, labels):
    while True: # so that Keras can loop through this as long as it wants
        for batch in batch_generator(raw_generator_with_label_from_tar("102flowers.tgz", files, labels), BATCH_SIZE):
            # prepare batch images
            batch_imgs = []
            batch_targets = []
            for raw, label in batch:
                img = prepare_raw_bytes_for_model(raw)
                batch_imgs.append(img)
                batch_targets.append(label)
            # stack images into 4D tensor [batch_size, img_size, img_size, 3]
            batch_imgs = np.stack(batch_imgs, axis=0)
            # convert targets into 2D tensor [batch_size, num_classes]
            batch_targets = keras.utils.np_utils.to_categorical(batch_targets, N_CLASSES)
            yield batch_imgs, batch_targets

s = reset_tf_session()
# don't call K.set_learning_phase() !!! (otherwise will enable dropout in train/test simultaneously)
def inception(use_imagenet=True):
    # load pre-trained model graph, don't add final layer
    model = keras.applications.InceptionV3(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet' if use_imagenet else None)
    # add gloabl pooling just like in InceptionV3
    new_output = keras.layers.GlobalAveragePooling2D()(model.output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(N_CLASSES, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model

model = inception()
print(model.summary())

# how many layers our model has
print(len(model.layers))
# set all layers trainable by default
for layer in model.layers:
    layer.trainable = True
# fix deep layers (fine-tuning only last 50)
for layer in model.layers[:-50]:
    layer.trainable = False

# compile new model
model.compile(
    loss = 'categorical_crossentropy',
    optimizer=keras.optimizers.adamax(lr=1e-2),
    metrics=['accuracy']
)
# we will save model checkpoints to continue training in case of kernel death
model_filename = 'flowers.{0:03d}.hdf5'
last_finished_epoch = None

#### uncomment below to continue training from model checkpoint
#### fill `last_finished_epoch` with your latest finished epoch
from keras.models import load_model
# s = reset_tf_session()
# last_finished_epoch = 10
# model = load_model(model_filename.format(last_finished_epoch))

# fine-tune for 2 epochs (full passes through all training data)
# we make 2*8 epochs, where epoch is 1/8 of our training data to see progress more often
from keras_tqdm import TQDMCallback
from keras.callbacks import ModelCheckpoint
# model.fit_generator(
#     train_generator(tr_files, tr_labels),
#     steps_per_epoch=len(tr_files) // BATCH_SIZE // 8,
#     epochs=2*8,
#     validation_data = train_generator(te_files, te_labels),
#     validation_steps=len(te_files) // BATCH_SIZE // 4,
#     callbacks=[TQDMCallback(), ModelCheckpoint('fine-tune-model.hdf5')],
#     verbose=0,
#     initial_epoch=last_finished_epoch or 0
# )
model = load_model('fine-tune-model.hdf5')

#test_accuracy = model.evaluate_generator(train_generator(te_files, te_labels), len(te_files) // BATCH_SIZE // 2)
test_accuracy = model.evaluate_generator(train_generator(te_files, te_labels), len(te_files) // BATCH_SIZE)
print(test_accuracy)
print(model.metrics_names)