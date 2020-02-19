from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import TimeDistributed, Bidirectional
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from keras import Model
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd
import pickle, gzip
import datetime
import argparse
import os, cv2 as cv
import sys
import socket
from sklearn.utils import shuffle

import warnings

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import LSTM
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image
from keras.optimizers import Adam, RMSprop
from keras.losses import categorical_crossentropy
import h5py
from keras.utils.io_utils import HDF5Matrix


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

def make_train_test_data():

    _X = np.memmap('/mnt/nas/data/model/data/golf_pose_128x128_3ch_X.dat', dtype='uint8', mode='r', shape=(6121, 120, 128, 128, 3))

    with gzip.open('/mnt/nas/data/model/data/golf_pose_128x128_3ch_Y.pickle', 'rb') as f:
        _Y = pickle.load(f)
        
    X = np.asarray(_X).astype('uint8')
    Y = np.asarray(_Y)
    
    with h5py.File('pose_128x128_3ch.hdf5', 'w') as f:
        f.creat_dataset('video', data=X)
        f.creat_dataset('label', data=Y)
    
    ## subtract mean and normalize
    #X -= np.mean(X, axis=0)
    #X /= 128.

    k = int(X.shape[0] * 0.8)    # for no shuffling...

    return X[:k], Y[:k], X[k:], Y[k:]


def frame_augmentation(org_imgs, pose_values, max_size, nb_classes=17):
    _x = []
    _y = []
    for pad_size in range(1, max_size+1):
        new_imgs = np.delete(org_imgs, np.s_[-pad_size:], 0)
        new_pose_values = np.delete(pose_values, np.s_[-pad_size:], 0)
        new_imgs2 = np.insert(new_imgs, [ 0 for x in range(pad_size)], org_imgs[0], axis=0)
        new_pose_values2 = np.insert(new_pose_values, [ 0 for x in range(pad_size)], pose_values[0], axis=0)
        _x.append(new_imgs2)
        _y.append(np_utils.to_categorical(np.asarray(new_pose_values2).astype('uint8'), nb_classes))
    return _x, _y


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def make_all_dir(dir_arr):
    for dir_nm in dir_arr:
        if not os.path.exists(dir_nm):
            os.makedirs(dir_nm)


def get_model_name():
    file_name = sys.argv[0]
    return file_name.split('.')[0]


def make_imgs(video_path):

    imgs = []
    cap = cv.VideoCapture(video_path)

    i = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        else:
            imgs.append(to_grayscale(cv.resize(frame,(60,60))).reshape(60,60,1))

        i += 1

    return np.asarray(imgs)


def get_YYYYMMDD():
    now = datetime.datetime.now()
    return "%d%02d%02d" % (now.year, now.month, now.day)


def Video_normalier(X_in):
    # subtract mean and normalize
    X_in = X_in.astype('float16')
    
    X_in[:,:,:,0] -= np.mean(X_in[:,:,:,0])
    X_in[:,:,:,1] -= np.mean(X_in[:,:,:,1])
    X_in[:,:,:,2] -= np.mean(X_in[:,:,:,2])
    
    #X_in -= np.mean(X_in, axis=1)
    X_in /= 255.
    return X_in


def td_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None, 
                 is_first_layer = False, is_trainable=False, input_shape=(120,128,128,3) ):
    """Utility function to apply time distributed + conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
        
    x = TimeDistributed(Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            trainable=is_trainable,
            name=conv_name))(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def pre_conv_lstm(input_shape=(120, 128, 128, 3), classes=17, weight_load=False):
    """
    Arguments:
        input_shape: optional shape tuple
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 4
        
    img_input = Input(shape=input_shape)

    x = td_conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = td_conv2d_bn(x, 32, 3, 3, padding='valid')
    x = td_conv2d_bn(x, 64, 3, 3)
    x = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2)))(x)

    x = td_conv2d_bn(x, 80, 1, 1, padding='valid')
    x = td_conv2d_bn(x, 192, 3, 3, padding='valid')
    x = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2)))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = td_conv2d_bn(x, 64, 1, 1)

    branch5x5 = td_conv2d_bn(x, 48, 1, 1)
    branch5x5 = td_conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = td_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = td_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = td_conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = TimeDistributed(AveragePooling2D((3, 3), strides=(1, 1), padding='same'))(x)
    branch_pool = td_conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = td_conv2d_bn(x, 64, 1, 1)

    branch5x5 = td_conv2d_bn(x, 48, 1, 1)
    branch5x5 = td_conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = td_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = td_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = td_conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = TimeDistributed(AveragePooling2D((3, 3), strides=(1, 1), padding='same'))(x)
    branch_pool = td_conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = td_conv2d_bn(x, 64, 1, 1)

    branch5x5 = td_conv2d_bn(x, 48, 1, 1)
    branch5x5 = td_conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = td_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = td_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = td_conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = TimeDistributed(AveragePooling2D((3, 3), strides=(1, 1), padding='same'))(x)
    branch_pool = td_conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = td_conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = td_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = td_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = td_conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2)))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = td_conv2d_bn(x, 192, 1, 1)

    branch7x7 = td_conv2d_bn(x, 128, 1, 1)
    branch7x7 = td_conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = td_conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = td_conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = td_conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = td_conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = td_conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = td_conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = TimeDistributed(AveragePooling2D((3, 3), strides=(1, 1), padding='same'))(x)
    branch_pool = td_conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = td_conv2d_bn(x, 192, 1, 1)

        branch7x7 = td_conv2d_bn(x, 160, 1, 1)
        branch7x7 = td_conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = td_conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = td_conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = td_conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = td_conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = td_conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = td_conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = TimeDistributed(AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same'))(x)
        branch_pool = td_conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = td_conv2d_bn(x, 192, 1, 1)

    branch7x7 = td_conv2d_bn(x, 192, 1, 1)
    branch7x7 = td_conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = td_conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = td_conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = td_conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = td_conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = td_conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = td_conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = TimeDistributed(AveragePooling2D((3, 3), strides=(1, 1), padding='same'))(x)
    branch_pool = td_conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = td_conv2d_bn(x, 192, 1, 1)
    branch3x3 = td_conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = td_conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = td_conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = td_conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = td_conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2)))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')
    
    x = TimeDistributed(GlobalAveragePooling2D(name='avg_pool'))(x)
    x = LSTM(128, return_sequence = True)(x)
    x = LSTM(128, return_sequence = True)(x)
    softmax_out = TimeDistributed(Dense(classes, activation='softmax', name='predictions'))(x)
    
    # Create model.
    model = Model(img_input, softmax_out, name='inception_v3')
    
    if weight_load:
        # load weights
        weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model    
   
if __name__ == '__main__':
    CUR_YYYYMMDD = '20181213/'
    CSV_LOG_DIR = 'model/csv_logger/pose/' + CUR_YYYYMMDD
    TBBOARD_DIR = 'model/tensorboard/pose/' + CUR_YYYYMMDD
    H5_DIR = 'model/h5/pose/' + CUR_YYYYMMDD
    make_all_dir([CSV_LOG_DIR, TBBOARD_DIR, H5_DIR])

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=10, patience=50, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=300)
    csv_logger = CSVLogger(CSV_LOG_DIR + 'golf_pose_' + get_YYYYMMDD() + '.csv')

    n_batch = 4
    n_epoch = 1000


    print("training start")
    
    idx = 0

    x_tr = HDF5Matrix('pose_128x128_3ch_train.hdf5', 'video', normalizer=Video_normalier)
    y_tr =  HDF5Matrix('pose_128x128_3ch_train.hdf5', 'label', normalizer=None)
    
    x_test = HDF5Matrix('pose_128x128_3ch_test.hdf5', 'video', normalizer=Video_normalier)
    y_test =  HDF5Matrix('pose_128x128_3ch_test.hdf5', 'label', normalizer=None)    
    
    model = pre_conv_lstm(weight_load=True)
    
    #adam_opt=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    rmsp_opt=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model = ModelMGPU(model, gpus=2)
    model.compile(loss='categorical_crossentropy',
              optimizer=rmsp_opt,
              metrics=['accuracy'])
    
    weights_file = H5_DIR + "%d_fold" % idx + '.h5'
    model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True, save_weights_only=False,
                                   mode='auto')  # save model every epoch..
    tb_hist = TensorBoard(log_dir=TBBOARD_DIR, histogram_freq=0, write_graph=True, write_images=True)

    model.fit(x_tr, y_tr,
              batch_size=n_batch,
              epochs=n_epoch,
              validation_data=(x_test, y_test),
              shuffle="batch",
              verbose = 2,
              callbacks=[lr_reducer, early_stopper, csv_logger, model_checkpoint, tb_hist])
        
    y_pred = model.predict(x_test, batch_size=1).argmax(axis=-1)
    pred_df = pd.DataFrame(columns=['y_true', 'y_pred'])
    for i in range(y_pred.shape[0]):
        pred_df = pred_df.append({'y_true': np.argmax(y_test[i], axis=1), 'y_pred': y_pred[i]},
                                 ignore_index=True)
        pred_df.to_csv(CSV_LOG_DIR + "%d_fold" % idx + '_predict.csv')

    scores = model.evaluate(x_test, y_test, batch_size=n_batch)
    test_df = pd.DataFrame(data={'test_loss': [scores[0]], 'test_acc': [scores[1]]})
    test_df.to_csv(CSV_LOG_DIR + "%d_fold" % idx + '_test_score.csv')

    K.clear_session()
