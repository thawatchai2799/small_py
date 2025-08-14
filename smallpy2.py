#--- 2025-08-14 17-57 – by Dr. Thawatchai Chomsiri  

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.models import Model
import datetime

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
import datetime
import numpy as np
import os
import re
import math
import pickle

def sechlu(x):
    Pi = math.pi
    return x * (1/Pi * tf.math.atan(tf.math.sinh(x)) + 0.5)

# Define relu6
#def relu6(x):
#    return tf.keras.activations.relu(x, max_value=6)

# Define paralu
def paralu1500(x):
    k = 15.00
    return tf.where(
        x >= 0,
        x,
        tf.where(
            x >= -1 * k,
            x * (1/tf.pow(k,3)) * tf.pow(x + k, 3),
            tf.zeros_like(x)
        )
    )

def paralu2(x):
    Pi = math.pi
    eight_pi_cubed = 8 * tf.pow(Pi, 3)
    return tf.where(
        x >= 0,
        x,
        tf.where(
            x >= -2 * Pi,
            x * (1/eight_pi_cubed) * tf.pow(x + (2*Pi), 3),
            tf.zeros_like(x)
        )
    )
    
def paralu(x):
    cube_root_of_three = 3 ** (1/3)
    return tf.where(
        x >= 0,
        x,
        tf.where(
            x >= -cube_root_of_three,
            x * (1/3) * tf.pow(x + cube_root_of_three, 3),
            tf.zeros_like(x)
        )
    )

def paralu2(x):
    Pi = math.pi
    eight_pi_cubed = 8 * tf.pow(Pi, 3)
    return tf.where(
        x >= 0,
        x,
        tf.where(
            x >= -2 * Pi,
            x * (1/eight_pi_cubed) * tf.pow(x + (2*Pi), 3),
            tf.zeros_like(x)
        )
    )

def lsgelu(x):    # Left-Shifted GELU with 1 range
    return x * 0.5 * (1 + tf.math.erf((x + 1.5) / tf.sqrt(2.0)))

def xsinelu(x):
    Pi = math.pi
    return tf.where(
        x >= Pi,
        x,
        tf.where(
            x >= -1 * Pi,
            x * ( ((tf.sin(x)) + x + Pi ) / (2*Pi) ),
            tf.zeros_like(x)
        )
    )  

def xsinelu2pi(x):
    Pi = math.pi
    return tf.where(
        x >= 0,
        x,
        tf.where(
            x >= -2 * Pi,
            x * ( ((tf.sin(x+Pi)) + (x+Pi) + Pi ) / (2*Pi) ),
            tf.zeros_like(x)
        )
    )  
  
# Function สำหรับสร้างโมเดล
def build_model(activation_fn):
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), strides=2, padding='same', activation=None)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)

    def bottleneck_block(x, filters):
        dw = DepthwiseConv2D((3, 3), padding='same')(x)
        dw = BatchNormalization()(dw)
        dw = Activation(activation_fn)(dw)

        pw = Conv2D(filters, (1, 1), padding='same')(dw)
        pw = BatchNormalization()(pw)
        pw = Activation(activation_fn)(pw)
        return pw

    x = bottleneck_block(x, 64)
    x = bottleneck_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(100, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# ฟังก์ชันโหลด batch จากไฟล์
def load_cifar100_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data']
    labels = batch[b'fine_labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


print(f"\nHello at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
