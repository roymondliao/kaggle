#!/usr/bin/env python3.5

import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import Model, Input
from keras import models, layers
from keras import initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from distutils.version import StrictVersion
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K

if StrictVersion(keras.__version__) < StrictVersion('2.2.0'):
    from keras.applications.imagenet_utils import _obtain_input_shape
else:
    from keras_applications.imagenet_utils import _obtain_input_shape

# Global setting
weights_collection = [
    # ResNet34
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        'name': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        'md5': 'a7b3fe01876f51b976af0dea6bc144eb',
    },
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'name': 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'md5': 'a268eb855778b3df3c7506639542a6af',
    }]


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_normal',
        'use_bias': True
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True
    }
    default_bn_params.update(params)
    return default_bn_params


class ResNet50:
    def __init__(self, conv_params, bn_params, weights_tag,
                 repetitions=(2, 2, 2, 2), include_top=True, input_tensor=None,
                 input_shape=None, n_classes=1000):
        self.repetitions = repetitions # disabele paramter
        self.include_top = include_top
        self.input_tensor = input_tensor
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.init_filters = 64
        self.conv_params = conv_params
        self.bn_params = bn_params
        self.weights_tag = weights_tag

    @staticmethod
    def handle_block_names(stage, block):
        name_base = 'stage{}_block{}_'.format(stage + 1, block + 1)
        conv_name = name_base + 'conv'
        bn_name = name_base + 'bn'
        relu_name = name_base + 'relu'
        shortcut_name = name_base + 'shortcut'
        return conv_name, bn_name, relu_name, shortcut_name

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        conv_name, bn_name, relu_name, shortcut_name = self.handle_block_names(stage, block)
        filter1, filter2, filter3 = filters

        # First component of main path
        x = layers.Conv2D(filter1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name + '2a')(input_tensor)
        x = layers.BatchNormalization(name=bn_name + '2a', **self.bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2a')(x)

        # Second component of main path
        x = layers.Conv2D(filter2, kernel_size=kernel_size, strides=(1, 1), padding='same', name=conv_name + '2b')(x)
        x = layers.BatchNormalization(name=bn_name + '2b', **self.bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2b')(x)

        # Third component of main path
        x = layers.Conv2D(filter3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name + '2c')(x)
        x = layers.BatchNormalization(name=bn_name + '2c', **self.bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2c')(x)

        # Final step: Add shortcut value to main path and pass it throught a relu
        x = layers.Add()([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        conv_name, bn_name, relu_name, shortcut_name = self.handle_block_names(stage, block)
        filter1, filter2, filter3 = filters

        # First component of main path
        x = layers.Conv2D(filter1, (1, 1), strides=strides, name=conv_name + '2a')(input_tensor)
        x = layers.BatchNormalization(name=bn_name + '2a', **self.bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2a')(x)

        # Second component of main path
        x = layers.Conv2D(filter2, kernel_size, strides=(1, 1), padding='same', name=conv_name + '2b')(x)
        x = layers.BatchNormalization(name=bn_name + '2b', **self.bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2b')(x)

        # Third component of main path
        x = layers.Conv2D(filter3, (1, 1), strides=(1, 1), name=conv_name + '2c')(x)
        x = layers.BatchNormalization(name=bn_name + '2c', **self.bn_params)(x)

        # Shortcut path
        x_shortcut = layers.Conv2D(filter3, (1, 1), strides=strides, name=conv_name + '1')(input_tensor)
        x_shortcut = layers.BatchNormalization(name=bn_name + '1')(x_shortcut)

        # Final step
        x = layers.Add()([x, x_shortcut])
        x = layers.Activation('relu')(x)
        return x

    def load_model_weights(self, weights_collection, model, dataset):

        def find_weights(weights_collection, model_name, dataset, include_top):
            w = list(filter(lambda x: x['model'] == model_name, weights_collection))
            w = list(filter(lambda x: x['dataset'] == dataset, w))
            w = list(filter(lambda x: x['include_top'] == include_top, w))
            return w

        weights = find_weights(weights_collection, model.name, dataset, self.include_top)

        if weights:
            weights = weights[0]
            if self.include_top and weights['classes'] != self.n_classes:
                raise ValueError('If using `weights` and `include_top`'
                                 ' as true, `classes` should be {}'.format(weights['classes']))

            weights_path = get_file( weights['name'],
                                    weights['url'],
                                    cache_subdir='models',
                                    md5_hash=weights['md5'])
        else:
            raise ValueError('There is no weights for such configuration: ' +
                             'model = {}, dataset = {}, '.format(model.name, dataset) +
                             'classes = {}, include_top = {}.'.format(self.n_classes, self.include_top))
        return weights_path

    def build(self, pooling=None):
        input_shape = _obtain_input_shape(self.input_shape, default_size=224, min_size=32,
                                          data_format='channels_last', require_flatten=self.include_top)

        if self.input_tensor is None:
            img_input = Input(shape=self.input_shape, name='data')
        else:
            if not K.is_keras_tensor(self.input_tensor):
                img_input = Input(tensor=self.input_tensor, shape=self.input_shape)
            else:
                img_input = self.input_tensor

        # stage 1
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_padding')(img_input)
        x = layers.Conv2D(self.init_filters, (7, 7), strides=(2, 2), name='conv1', **self.conv_params)(x)
        x = layers.BatchNormalization(name='bn_conv1', **self.bn_params)(x)
        x = layers.Activation('relu', name='relu1')(x)
        x = layers.MaxPool2D((3, 3), strides=(2, 2), padding='valid', name='pooling1')(x)

        # stage 2 - 3 blocks
        x = self.conv_block(x, 3, [64, 64, 256], strides=(1, 1), stage=2, block=1)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block=2)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block=3)

        # stage 3 - 4 blocks
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block=1)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block=2)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block=3)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block=4)

        # stage 4 - 6 blocks
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

        # stage 5 - 3 blocks
        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block=3)
        x = layers.AveragePooling2D((2, 2), name='avg_pool')(x)
        
        if self.include_top:
            x = layers.Flatten()(x)
            x = layers.Dense(self.n_classes, activation='softmax', name='fc1000')(x)
        else:
            if pooling == 'avg':
                x = layers.GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = layers.GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account any potential predecessors of `input_tensor`.
        if self.input_tensor is not None:
            inputs = get_source_inputs(self.input_tensor)
        else:
            inputs = img_input

        model = Model(inputs=inputs, outputs=x, name='resnet50')

        w_path = self.load_model_weights(weights_collection, model, dataset=self.weights_tag)
        model.load_weights(w_path)

        return model
