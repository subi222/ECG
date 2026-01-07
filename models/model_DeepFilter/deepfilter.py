#============================================================
#
#  Deep Learning BLW Filtering
#  Deep Learning models (CLEANED VERSION)
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#  Cleaned by: Removing experimental model variants
#  Kept only: LANLFilter modules and main DeepFilter models
#
#===========================================================


import keras
from keras.models import Model
from keras.layers import Conv1D, Dropout, BatchNormalization, concatenate, Input

import keras.backend as K

##########################################################################

###### MODULES #######

def LANLFilter_module(x, layers):
    """
    Linear And Non-Linear Filter Module (LANL)
    
    8-branch parallel architecture:
    - 4 Linear branches (kernel sizes: 3, 5, 9, 15)
    - 4 Non-linear branches with ReLU (kernel sizes: 3, 5, 9, 15)
    """
    # Linear branches (4)
    LB0 = Conv1D(filters=int(layers / 8),
                 kernel_size=3,
                 activation='linear',
                 strides=1,
                 padding='same')(x)
    LB1 = Conv1D(filters=int(layers / 8),
                kernel_size=5,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 8),
                kernel_size=9,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 8),
                kernel_size=15,
                activation='linear',
                strides=1,
                padding='same')(x)

    # Non-linear branches with ReLU (4)
    NLB0 = Conv1D(filters=int(layers / 8),
                  kernel_size=3,
                  activation='relu',
                  strides=1,
                  padding='same')(x)
    NLB1 = Conv1D(filters=int(layers / 8),
                 kernel_size=5,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 8),
                 kernel_size=9,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 8),
                 kernel_size=15,
                 activation='relu',
                 strides=1,
                 padding='same')(x)

    # Concatenate all 8 branches
    x = concatenate([LB0, LB1, LB2, LB3, NLB0, NLB1, NLB2, NLB3])

    return x


def LANLFilter_module_dilated(x, layers):
    """
    LANL Filter Module with Dilation Rate = 3
    
    6-branch parallel architecture (kernel 3 removed for dilated version):
    - 3 Linear branches (kernel sizes: 5, 9, 15)
    - 3 Non-linear branches with ReLU (kernel sizes: 5, 9, 15)
    All with dilation_rate=3
    """
    # Linear branches with dilation=3 (3)
    LB1 = Conv1D(filters=int(layers / 6),
                kernel_size=5,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 6),
                kernel_size=9,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 6),
                kernel_size=15,
                dilation_rate=3,
                activation='linear',
                padding='same')(x)

    # Non-linear branches with ReLU and dilation=3 (3)
    NLB1 = Conv1D(filters=int(layers / 6),
                 kernel_size=5,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 6),
                 kernel_size=9,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 6),
                 kernel_size=15,
                 dilation_rate=3,
                 activation='relu',
                 padding='same')(x)

    # Concatenate all 6 branches
    x = concatenate([LB1, LB2, LB3, NLB1, NLB2, NLB3])

    return x


###### MODELS #######

def deep_filter_I_LANL(signal_size=512):
    """
    DeepFilter Main Model (Paper Version)
    
    Architecture:
    - 6 sequential LANLFilter_module blocks
    - BatchNormalization after each block
    - Final Conv1D to produce single-channel output
    
    Recommended for: General use, faster training
    """
    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    tensor = LANLFilter_module(input, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = BatchNormalization()(tensor)
    predictions = Conv1D(filters=1,
                    kernel_size=9,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def deep_filter_model_I_LANL_dilated(signal_size=512):
    """
    DeepFilter Enhanced Model (Dilated + Dropout Version)
    
    Architecture:
    - 6 blocks alternating between LANLFilter_module and LANLFilter_module_dilated
    - Dropout(0.4) + BatchNormalization after each block
    - Final Conv1D to produce single-channel output
    
    Recommended for: Overfitting prevention, better generalization
    """
    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    tensor = LANLFilter_module(input, 64)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 64)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 32)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 16)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    predictions = Conv1D(filters=1,
                    kernel_size=9,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model
