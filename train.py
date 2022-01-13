### Multi-Label Classification

# Resnet + FC layers (224 x 224)
# RGB Channel
# Focal loss

!curl ipinfo.io  # us-central과 같은 대륙

!echo "deb http://packages.cloud.google.com/apt gcsfuse-`lsb_release -c -s` main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
!curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
!sudo apt-get -y -q update
!sudo apt-get -y -q install gcsfuse
!pip install tensorflow_addons

import os
import numpy as np
import pandas as pd
import cv2
import pickle
from tqdm.notebook import tqdm
import random

import tensorflow as tf
#from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.metrics import Recall, Precision, CategoricalAccuracy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
#from tensorflow_addons.metrics import MultiLabelConfusionMatrix


# Params

# Const Params
NUM_CLASSES = 19

# Hyper Params
#IMAGE_TYPE = 'jpg'
IMAGE_SIZE = 224 # 224 resnet default?
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'relu'
OUTPUT_LAYER_ACTIVATION = 'sigmoid'

THRESHOLDS_METRICS = 0.5
DENSE_LAYERS = [1024, 256, 64]
DROPOUT_RATE = 0.2

NUM_EPOCHS = 10
#EARLY_STOP_PATIENCE = 3
#STEPS_PER_EPOCH_TRAINING = 10
#STEPS_PER_EPOCH_VALIDATION = 10
BATCH_SIZE = 32


# Collect Data
#    HPA_generator : Generator 방식 Batch 생성기 (메모리 절약용)
#    DataHandling : 한번에 전부 로드

class DataHandling():
    labels = np.array(
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'])
    cross_valid_frac = 0.1
    shuffle_size = 100000

    def __init__(self, _max_sample=100000000):
        self.max_sample = _max_sample

    def data_generator(self):
        data_list_X = []
        data_list_Y = []
        sample_cnt = 0
        for filenames in tqdm(os.listdir(DATA_ROOT)):
            loaded = pickle.load(open(DATA_ROOT + filenames, 'rb'))
            for data in loaded:
                if (sample_cnt == self.max_sample):
                    return data_list_X, data_list_Y, sample_cnt

                img = cv2.imdecode(data['img'], cv2.IMREAD_UNCHANGED)
                if img.shape[0] != IMAGE_SIZE:
                    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = tf.cast(img, tf.float32)
                img = img / 255
                img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)

                class_id = data['label'].split('|')
                label = np.zeros((NUM_CLASSES,), dtype=int)
                # label = np.full((NUM_CLASSES,), -1, dtype=int)
                for id in class_id:
                    new_label = id == self.labels
                    # new_label = new_label.astype(int) * 2
                    label += new_label
                    # label = tf.clip_by_value(label, clip_value_min=0, clip_value_max=1)
                data_Y = tf.convert_to_tensor(label, dtype=tf.float32)

                data_list_X.append(img)
                data_list_Y.append(data_Y)
                sample_cnt += 1
            del loaded
        return data_list_X, data_list_Y, sample_cnt

    def get_dataset(self):
        data_list_X, data_list_Y, sample_cnt = self.data_generator()
        # tensor_X = tf.stack(data_list_X)
        # tensor_Y = tf.stack(data_list_Y)

        train_ds = tf.data.Dataset.from_tensor_slices((data_list_X, data_list_Y))
        train_ds = train_ds.shuffle(buffer_size=self.shuffle_size)
        valid_ds = train_ds.take(int(sample_cnt * self.cross_valid_frac))
        train_ds = train_ds.skip(int(sample_cnt * self.cross_valid_frac))
        train_ds = train_ds.batch(BATCH_SIZE)
        valid_ds = valid_ds.batch(BATCH_SIZE)
        train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_ds, valid_ds


class HPA_generator(tf.keras.utils.Sequence):
    labels = np.array(
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'])

    def __init__(self, path, batch_size=64, max_file=1):
        self.batch_size = batch_size
        self.max_file = max_file
        self.path = path

        self.dump_list = []
        for f in os.listdir(self.path):
            self.dump_list.append(f)
        self.dump_len = len(self.dump_list)

        self.current_idx = 0
        self.current_data = pickle.load(open(self.path + self.dump_list[self.current_idx], 'rb'))
        self.current_idx += 1

        self.current_cnt = 0
        self.current_len = len(self.current_data)
        self.file_length = self.max_file * int(self.current_len / batch_size)

    def __len__(self):
        return self.file_length

    def __getitem__(self, idx):
        if self.current_len <= (self.current_cnt + 1) * self.batch_size:
            self.current_data = pickle.load(open(self.path + self.dump_list[self.current_idx], 'rb'))
            self.current_len = len(self.current_data)
            self.current_idx += 1
            self.current_cnt = 0

        batch = self.current_data[self.current_cnt * self.batch_size: (self.current_cnt + 1) * self.batch_size]
        self.current_cnt += 1

        data_list_X = []
        data_list_Y = []
        for data in batch:
            img = cv2.imdecode(data['img'], cv2.IMREAD_UNCHANGED)
            if img.shape[0] != IMAGE_SIZE:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = tf.cast(img, tf.float32)
            img = img / 255
            img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)

            class_id = data['label'].split('|')
            label = np.zeros((NUM_CLASSES,), dtype=int)
            # label = np.full((NUM_CLASSES,), -1, dtype=int)
            for id in class_id:
                new_label = id == self.labels
                # new_label = new_label.astype(int) * 2
                label += new_label
                # label = tf.clip_by_value(label, clip_value_min=0, clip_value_max=1)
            data_Y = tf.convert_to_tensor(label, dtype=tf.float32)

            data_list_X.append(img)
            data_list_Y.append(data_Y)

        tensor_X = tf.stack(data_list_X)
        tensor_Y = tf.stack(data_list_Y)
        return tensor_X, tensor_Y

    def reset(self):
        self.current_cnt = 0
        self.current_idx = 0
        self.current_data = pickle.load(open(self.path + self.dump_list[self.current_idx], 'rb'))
        self.current_idx += 1


# Make Model

class HPA_Model(Model):
    def __init__(self, input_shape=None):
        super(HPA_Model, self).__init__()
        self.input_layer = tf.keras.layers.Input(input_shape)
        self.depth = len(DENSE_LAYERS)
        #self.rand_flip = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
        #self.rand_rot = tf.keras.layers.experimental.preprocessing.RandomRotation(1.0)
        #self.effnet = EfficientNetB4(include_top=False, pooling='avg', input_shape=input_shape)
        self.effnet = ResNet50(include_top=False, pooling='avg', input_shape=input_shape)
        self.effnet.trainable = False
        self.batch_normalization = []
        self.dense_1 = []
        #self.dropout = []
        self.batch_normalization.append(BatchNormalization())
        for layer_size in DENSE_LAYERS:
            self.dense_1.append(Dense(layer_size, activation = DENSE_LAYER_ACTIVATION)) #kernel_regularizer=tf.keras.regularizers.l2(0.001)
            #self.batch_normalization.append(BatchNormalization())
            #self.dropout.append(Dropout(DROPOUT_RATE))
        self.dense_2 = Dense(NUM_CLASSES, activation = OUTPUT_LAYER_ACTIVATION)
        self.out = self.call(self.input_layer)

    def call(self, x, training=None):
        #if training:
        #    x = self.rand_flip(x)
        #    x = self.rand_rot(x)
        x = self.batch_normalization[0](x)
        x = self.effnet(x)
        for i in range(self.depth):
            x = self.dense_1[i](x)
            #x = self.batch_normalization[i+1](x)
            #x = self.dropout[i](x)
        out = self.dense_2(x)
        return out

LABELS = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'])
DATA_ROOT = "/content/data/work/seg_v6/"

#train_ds = HPA_generator(DATA_ROOT + 'train/', BATCH_SIZE, 10)
#valid_ds = HPA_generator(DATA_ROOT + 'test/', BATCH_SIZE, 1)

data = DataHandling(5000)
train_ds, valid_ds = data.get_dataset()

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(3, 1, figsize=(20,50))
#ax[0].imshow(cv2.imdecode(valid_ds.current_data[0]['img'], cv2.IMREAD_UNCHANGED))
#ax[1].imshow(cv2.imdecode(valid_ds.current_data[1]['img'], cv2.IMREAD_UNCHANGED))
#ax[2].imshow(cv2.imdecode(valid_ds.current_data[2]['img'], cv2.IMREAD_UNCHANGED))

model = HPA_Model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

loss_object = SigmoidFocalCrossEntropy()
#optimizer = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()

train_acc_rec = Recall()
test_acc_rec = Recall()
train_acc_pre = Precision()
test_acc_pre = Precision()


@tf.function
def train_step(train_x, train_y):
    with tf.GradientTape() as tape:
        predictions = model(train_x, training=True)
        #tf.print(predictions)
        loss = loss_object(train_y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    inference = tf.math.equal(predictions, tf.math.reduce_max(predictions, axis=1, keepdims=True))
    inference = tf.bitwise.bitwise_or(tf.cast(inference, tf.int8), tf.cast(predictions>THRESHOLDS_METRICS, tf.int8))
    train_acc_rec.update_state(train_y>0, inference)
    train_acc_pre.update_state(train_y>0, inference)

@tf.function
def valid_step(valid_x, valid_y):
    predictions = model(valid_x, training=False)
    #tf.print(predictions)
    t_loss = loss_object(valid_y, predictions)
    test_loss(t_loss)
    inference = tf.math.equal(predictions, tf.math.reduce_max(predictions, axis=1, keepdims=True))
    inference = tf.bitwise.bitwise_or(tf.cast(inference, tf.int8), tf.cast(predictions>THRESHOLDS_METRICS, tf.int8))
    test_acc_rec.update_state(valid_y>0, inference)
    test_acc_pre.update_state(valid_y>0, inference)

for epoch in range(NUM_EPOCHS):
    pbar = tqdm(total=len(train_ds))
    for train_x, train_y in train_ds:
        train_step(train_x, train_y)
        pbar.update(1)

    for valid_x, valid_y in valid_ds:
        valid_step(valid_x, valid_y)

    template = 'epoch: {}, loss_train: {:.4f}, rec_train: {:.4f}, pre_train: {:.4f}, loss_test: {:.4f}, rec_test: {:.4f}, pre_test: {:.4f}'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_acc_rec.result(),
                           train_acc_pre.result(),
                           test_loss.result(),
                           test_acc_rec.result(),
                           test_acc_pre.result()
                           ))
model.summary()
tf.saved_model.save(model, '/content/data/work/model/model_v4')
#tf.saved_model.save(model, 'gs://blonix-tpu-bucket/kaggle_market/model_a')


# test

loaded_model = tf.saved_model.load("/content/data/work/model/model_v4")
labels_infer = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'])
infer_acc_rec = Recall()
infer_acc_pre = Precision()

for valid_x, valid_y in valid_ds:
    predictions = loaded_model(valid_x, training=False)
    #tf.print(predictions)
    inference = tf.math.equal(predictions, tf.math.reduce_max(predictions, axis=1, keepdims=True))
    inference = tf.bitwise.bitwise_or(tf.cast(inference, tf.int8), tf.cast(predictions>0.35, tf.int8))

    pred_strs = []
    for idx in range(len(inference)):
        class_id_list = []
        for i, proba in enumerate(inference[idx]):
            if proba == 1:
                class_id_list.append(labels_infer[i])
        class_id = '|'.join(class_id_list)
        pred_strs.append(f'{class_id}')

    key_strs = []
    for idx in range(len(valid_y)):
        class_id_list = []
        for i, proba in enumerate(valid_y[idx]):
            if proba == 1:
                class_id_list.append(labels_infer[i])
        class_id = '|'.join(class_id_list)
        key_strs.append(f'{class_id}')

    infer_acc_rec.update_state(valid_y>0, inference)
    infer_acc_pre.update_state(valid_y>0, inference)
    print(" ".join(pred_strs))
    print(" ".join(key_strs))
    print(f'rec : {infer_acc_rec.result()} / pre : {infer_acc_pre.result()}')

    cnt = 0.
    for i, key in enumerate(pred_strs):
        if key == key_strs[i]:
            cnt += 1.
    acc = cnt / len(pred_strs)
    print(f'accuracy : {acc}')