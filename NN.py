import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA usage

import tensorflow as tf
import pandas as pd
import numpy as np

for i in range(10):
    print("\n")


def MPP(features, labels, hparams):    # Model Prediction and Performance

    hidden_units = hparams.get('hidden_units')
    num_layers = hparams.get('num_layers',)
    activation = hparams.get('activation')
    epochs = hparams.get('epochs')
    optimizer = hparams.get('optimizer')
    loss = hparams.get('loss')
    
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Input(shape=(features.shape[1],)))

    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(hidden_units, activation=activation))
    
    model.add(tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax'))

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])    

    return model

def CNN(input_shape, num_classes):  # Convolutional Neural Network

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":

    # Creating example data foro MPP

    train_dataset = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'feature4': np.random.rand(100),
        'weights': np.random.randint(0, 2, size=100)  # Target (for example, two classes: 0 and 1)
    })
    features = train_dataset.iloc[:, :-1].values  # All columns except the last
    labels = train_dataset.iloc[:, -1].values.astype(int)  # The last column corresponds to labels and is cast to int

    hparams = {
        'hidden_units': 256,
        'num_layers': 4,
        'activation': tf.nn.relu,
        'epochs': 100,
        'optimizer': 'adam',
        'loss': 'sparse_categorical_crossentropy'
    }
    
    model = MPP(features, labels, hparams)
