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
        'epsilon': 1.001e-5,
        'center': True,
        'scale': True
    }
    default_bn_params.update(params)
    return default_bn_params

class ResNext:
    def __init__(self, conv_params, bn_params, repetitions, conv_shortcut,
                 weights_tag='imagenet', include_top=True, input_tensor=None,
                 input_shape=None, n_classes=1000):
        '''
        Parameters
        ----------
        conv_params   dict       : convolution layer parameters setting
        bn_params     dict       : batch normalization layer parameters setting
        weights_tag   string     : pre-training weight dataset name, default imagenet
        repetitions   list       : The list of number of block in each stages
        conv_shortcut boolean    : use convolution in shortcut path
        include_top   boolean    : use top layer in model
        input_tensor  tensor     : input layer
        input_shape   tuple      : the input shape of image
        n_classes     integer    : the number of classification

        Returns
           Output keras model
        -------
        '''
        self.repetitions = repetitions
        self.include_top = include_top
        self.input_tensor = input_tensor
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.init_filters = 64
        self.conv_shortcut = conv_shortcut
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
        return name_base, conv_name, bn_name, relu_name, shortcut_name

    def group_conv_block(self, input_tensor, filters, stage, block,
                         kernel_size=3, stride=1, groups=32):
        '''
        Parameters
        ----------
        input_tensor tensor             : input tensor
        filters      integer            : filters of the bottleneck layer
        kernel_size  tuple/list/integer : kernel size of the bottleneck layer, default = 3
        stride       tuple/list/integer : stride of the first layer, default = 1
        groups       integer            : group size of the first layers, default = 32
        stage        integer            : stage label
        block        integer            : block label

        Returns
        Output tensor for the residual block.
        -------
        '''
        name_base, conv_name, bn_name, relu_name, shortcut_name = self.handle_block_names(stage, block)
        # bn_axis = 3 if K.image_data_format() == 'channel_last' else 1

        if self.conv_shortcut is True:
            shortcut = layers.Conv2D((64 // groups) * filters,
                                     kernel_size=(kernel_size, kernel_size),
                                     strides=(stride, stride),
                                     name=shortcut_name + 'conv',
                                     **self.conv_params)(input_tensor)
            shortcut = layers.BatchNormalization(name=shortcut_name + 'bn',
                                                 **self.bn_params)(shortcut)
        else:
            shortcut = input_tensor

        x = layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), name=conv_name + 'a',
                          **self.conv_params)(input_tensor)
        x = layers.BatchNormalization(name=bn_name + 'a',  **self.bn_params)(x)
        x = layers.Activation('relu', name=relu_name + 'a')(x)

        cardinality = filters // groups
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name_base + 'pad')(x)
        # output channels will be equal to filters_in * depth_multiplier
        x = layers.DepthwiseConv2D(kernel_size=(kernel_size, kernel_size),
                                   strides=(stride, stride),
                                   depth_multiplier=cardinality,
                                   use_bias=False, name=name_base + 'group')(x)
        x_shape = K.int_shape(x)[1: -1]
        x = layers.Reshape(x_shape + (groups, cardinality, cardinality))(x)
        x = layers.Lambda(lambda x: sum([x[:, :, :, :, i] for i in range(cardinality)]),
                          name=name_base + 'reduce')(x)
        x = layers.Reshape(x_shape + (filters, ))(x)
        x = layers.BatchNormalization(name=bn_name + 'b', **self.bn_params)(x)
        x = layers.Activation('relu', name=relu_name + 'b')(x)

        x = layers.Conv2D((64 // groups) * filters, kernel_size=(1, 1), strides=(1, 1),
                          name=conv_name + 'c')(x)
        x = layers.BatchNormalization(name=bn_name + 'c', **self.bn_params)(x)

        x = layers.Add(name=name_base + 'add')([shortcut, x])
        x = layers.Activation('relu', name=relu_name + 'c')(x)
        return x

    def stack_blocks(self):
        pass





