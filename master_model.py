import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from tensorflow.keras.layers import Convolution2D, BatchNormalization, Activation, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, ReLU,LeakyReLU,Add,Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D, UpSampling2D


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

"""
def conv_block(X, filters, block):
    # resiudal block with dilated convolutions
    # add skip connection at last after doing convoluion operation to input X

    b = 'block_' + str(block) + '_'
    f1, f2, f3 = filters
    X_skip = X
    # block_a
    X = Convolution2D(filters=f1, kernel_size=(1, 1), dilation_rate=(1, 1),
                      padding='same', kernel_initializer='he_normal', name=b + 'a')(X)
    X = BatchNormalization(name=b + 'batch_norm_a')(X)
    X = LeakyReLU(alpha=0.2, name=b + 'leakyrelu_a')(X)
    # block_b
    X = Convolution2D(filters=f2, kernel_size=(3, 3), dilation_rate=(2, 2),
                      padding='same', kernel_initializer='he_normal', name=b + 'b')(X)
    X = BatchNormalization(name=b + 'batch_norm_b')(X)
    X = LeakyReLU(alpha=0.2, name=b + 'leakyrelu_b')(X)
    # block_c
    X = Convolution2D(filters=f3, kernel_size=(1, 1), dilation_rate=(1, 1),
                      padding='same', kernel_initializer='he_normal', name=b + 'c')(X)
    X = BatchNormalization(name=b + 'batch_norm_c')(X)
    # skip_conv
    X_skip = Convolution2D(filters=f3, kernel_size=(3, 3), padding='same', name=b + 'skip_conv')(X_skip)
    X_skip = BatchNormalization(name=b + 'batch_norm_skip_conv')(X_skip)
    # block_c + skip_conv
    X = Add(name=b + 'add')([X, X_skip])
    X = ReLU(name=b + 'relu')(X)
    return X
"""


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def base_feature_maps(input_layer):
    # base convolution module to get input image feature maps
    """
    # block_1
    base = conv_block(input_layer, [32, 32, 64], '1')
    # block_2
    base = conv_block(base, [64, 64, 128], '2')
    # block_3
    base = conv_block(base, [128, 128, 256], '3')
    return base

    """
    s1, p1 = encoder_block(input_layer, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    # s5, p5 = encoder_block(p4, 1024)
    return p4, s4, s3, s2, s1


def pyramid_feature_maps(input_layer):
    # pyramid pooling module
    #
    base, s4, s3, s2, s1 = base_feature_maps(input_layer)
    # base = base_feature_maps(input_layer)
    # red
    red = GlobalAveragePooling2D(name='red_pool')(base)
    red = tf.keras.layers.Reshape((1, 1, 512))(red)
    red = Convolution2D(filters=64, kernel_size=(1, 1), name='red_1_by_1')(red)
    red = UpSampling2D(size=32, interpolation='bilinear', name='red_upsampling')(red)
    # yellow
    yellow = AveragePooling2D(pool_size=(2, 2), name='yellow_pool')(base)
    yellow = Convolution2D(filters=64, kernel_size=(1, 1), name='yellow_1_by_1')(yellow)
    yellow = UpSampling2D(size=2, interpolation='bilinear', name='yellow_upsampling')(yellow)
    # blue
    blue = AveragePooling2D(pool_size=(4, 4), name='blue_pool')(base)
    blue = Convolution2D(filters=64, kernel_size=(1, 1), name='blue_1_by_1')(blue)
    blue = UpSampling2D(size=4, interpolation='bilinear', name='blue_upsampling')(blue)
    # green
    green = AveragePooling2D(pool_size=(8, 8), name='green_pool')(base)
    green = Convolution2D(filters=64, kernel_size=(1, 1), name='green_1_by_1')(green)
    green = UpSampling2D(size=8, interpolation='bilinear', name='green_upsampling')(green)
    # base + red + yellow + blue + green
    return tf.keras.layers.concatenate([base, red, yellow]), s4, s3, s2, s1


def last_conv_module(input_layer):

    X, s4, s3, s2, s1 = pyramid_feature_maps(input_layer)
    # d0 = decoder_block(X, s5, 1024)
    d1 = decoder_block(X, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    X = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    # X = tf.keras.layers.Flatten(name='last_conv_flatten')(X)
    """
    X = pyramid_feature_maps(input_layer)
    X = Convolution2D(filters=3, kernel_size=3, padding='same', name='last_conv_3_by_3')(X)
    X = BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)
    X = Activation('softmax', name='last_conv_relu')(X)
    
    # X = tf.reshape(X, [1, 1024, 1024, 3])
    """
    return X


def build_model():
    input_layer = tf.keras.Input((512, 512, 3), name='input')
    output_layer = last_conv_module(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # model.summary()
    return model

build_model()