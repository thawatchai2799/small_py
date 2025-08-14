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

folder_path = '../cifar-100-python'

# โหลดไฟล์ training batches
train_images_list = []
train_labels_list = []

for i in range(1, 6):
    file_path = os.path.join(folder_path, 'train')
    imgs, lbls = load_cifar100_batch(file_path)

x_train = np.array(imgs)
y_train = np.array(lbls)

# โหลดไฟล์ test batch
test_batch_path = os.path.join(folder_path, 'test')
x_test, y_test = load_cifar100_batch(test_batch_path)

# แปลงข้อมูลเป็น float และ normalize 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# แปลง label เป็น one-hot
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

activations_list = {
    #"XSINELU": xsinelu,
    #"XSINELU2Pi": xsinelu2pi,
    #'ELU': tf.nn.elu,    
    #'Swish': tf.nn.swish,
    #'LSGELU': lsgelu,
    #'GELU': tf.nn.gelu,
    #'PARALU': paralu,
    #'ReLU6': 'relu6',
    #'ReLU': tf.nn.relu,
    #'PARALU2': paralu2,
    'SECHLU': sechlu,

}

epochs = 501  ################
num_runs = 1 ###############
batch_size = 64 ###############
results = {
    'activation': [],
    'accuracy_per_epoch': []
}
accuracy_summary = {}

for run_idx in range(num_runs):
    print(f"\n--- Run {run_idx:03d} of {num_runs:03d} ---")
    results = {}
    accuracy_results = {}
    loss_results = {}
    
    for act_name, act_fn in activations_list.items():
        print(f"Running: Activation={act_name} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        model = build_model(act_fn)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        
        history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)
        
        print(f"Running: Activation={act_name} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")   
        results[act_name] = {key: np.array(val) for key, val in history.history.items()}
        print(f"Details in results for round {run_idx:03d}:")
        for act_name, metrics_dict in results.items():
            print(f"\nActivation: {act_name}")
            for metric_name, metric_values in metrics_dict.items():
                print(f"  {metric_name}: {metric_values}")

        np.savez(f"accuracy_{run_idx:03d}_{act_name}.npz", accuracy=np.array(history.history['accuracy']))
        np.savez(f"loss_{run_idx:03d}_{act_name}.npz", loss=np.array(history.history['loss']))
        
        accuracy_results[act_name] = np.array(history.history['accuracy'])
        loss_results[act_name] = np.array(history.history['loss'])

    print(f" ----- Data -------- ")
    results[act_name] = {key: np.array(val) for key, val in history.history.items()}
    print(f"Details in results for round {run_idx:03d}:")
    for act_name, metrics_dict in results.items():
        print(f"\nActivation: {act_name}")
        for metric_name, metric_values in metrics_dict.items():
            print(f"  {metric_name}: {metric_values}")

    print(f" ------------------- ")
   
print(f"\nEND at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
