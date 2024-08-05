from pathlib import Path
import random
import sys
import os
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as implt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
import tensorflow as tf
import tensorflow.data as tfd
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
from torchmetrics.text import CharErrorRate, WordErrorRate

from ctc_layer import CTCLayer
from hwr_dataset import IAM_dataset
from callbacks import SavePredictions

def calculate_accuracy(preds, labels):
    acuraccy = sum(1 for pred, label in zip(preds, labels)
        if pred == label)
    acuraccy /= len(labels)
    return acuraccy
    
def custom_serialize(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist() 
    raise TypeError(f'Tipo {type(obj)} no serializable')

def perform_experiment(**kwargs):
    '''
    This function encapsules all the logic begind of a experiment.
    '''
    experiment_path = kwargs.get('experiment_path',Path('.'))
    seed = kwargs.get('seed', 42)
    epochs = kwargs.get('epochs', 200)
    img_width = kwargs.get('img_width', 128)
    img_height = kwargs.get('img_height', 1024)     
    img_size = (img_width, img_height)
    dataset_path = kwargs.get('dataset_path', Path(r'\data\IAM'))
    task = kwargs.get('task', False)
    reduced = kwargs.get('reduced', True)
    batch_size = kwargs.get('batch_size', 16)
    dropout_CNN = kwargs.get('dropout_CNN', 0.1)
    dropout_dense = kwargs.get('dropout_dense', 0.4)
    dropout_RNN = kwargs.get('dropout_RNN', 0.4)
    CNN = kwargs.get('CNN', ((64, 32), (64, 32)))
    dense = kwargs.get('dense', (128,))
    RNN = kwargs.get('RNN', (128, 64, 64))
    earlyStopping_patience = kwargs.get('earlyStopping_patience', 10)
    reduceLROnPlateau_patience = kwargs.get('reduceLROnPlateau_patience', False)
    savePredictions = kwargs.get('savePredictions', False)
    batchNormalization = kwargs.get('batchNormalization', False)

    experiment_path.mkdir(parents=True, exist_ok=True)
    with open(str(experiment_path/'hyperparameters.json'), 'w') as f:
        json.dump(kwargs, f, default=custom_serialize)

    tf.keras.utils.set_random_seed(seed)
    
    # Dataset load
    iam_dataset = IAM_dataset(dataset_path, img_size, reduced=reduced, task=task, seed=seed)
    train_ds, val_ds, test_ds = iam_dataset.get_partitions(batch_size)

    # Model definition
    input_images = Input(shape=(img_width, img_height, 1), name='image')
    input_labels = Input(shape=(None, ), name='label')
    x = input_images
    for i, block in enumerate(CNN):
        for j, layer in enumerate(block):
            x = Conv2D(layer, 3, strides=1, padding='same', kernel_initializer='he_normal', name=f'conv_{i}_{j}')(x)
            if batchNormalization:
                x = BatchNormalization()(x)   
            x = Activation('relu')(x)
            x = Dropout(dropout_CNN, name=f'conv_{i}_{j}_dropout')(x)     
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=f'MaxPool_{i}')(x)
    target_shape = (img_width // 2**len(CNN), CNN[-1][-1] * img_height // 2**len(CNN))
    x = Reshape(target_shape=target_shape, name='reshape_layer')(x)
    for i, layer in enumerate(dense):
        x = Dense(layer, kernel_initializer='he_normal', activation='relu', name=f'enconding_dense_{i}')(x)
        x = Dropout(dropout_dense, name=f'enconding_dense_{i}_dropout')(x)
    for i, layer in enumerate(RNN):
        x = Bidirectional(LSTM(layer, return_sequences=True, dropout=dropout_RNN), name=f'bidirectional_lstm_{i}')(x)
    output = Dense(len(iam_dataset.char_to_num.get_vocabulary())+1, activation='softmax', name='output_dense')(x)
    ctc_loss_layer = CTCLayer()(input_labels, output) 
    model = Model(inputs=[input_images, input_labels], outputs=[ctc_loss_layer])
    model.compile(optimizer='adam')
    print(model.summary())

    # Model taining  
    callbacks = [
        EarlyStopping(patience=earlyStopping_patience, restore_best_weights=True),
        ModelCheckpoint(filepath=str(experiment_path/'model.keras'), save_best_only=True),
        CSVLogger(str(experiment_path/'training.log'))
    ]
    if savePredictions:
        callbacks.append(SavePredictions(iam_dataset, val_ds, save_dir=str(experiment_path/'predictions')))
    if reduceLROnPlateau_patience:
        callbacks.append(ReduceLROnPlateau(patience=reduceLROnPlateau_patience))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # Training visualization
    sns.set_style('white')
    sns.set_palette('deep')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    pd.DataFrame(history.history).plot(ax=ax)
    ax.legend(fontsize=12)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('CTC Loss Score')
    ax.set_title('Training and Validation Losses', fontsize=15)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig(str(experiment_path/'model_result.png'), facecolor='white', edgecolor='none')
    plt.show()

    # Metrics visualization
    metrics = {}
    metrics['loss'] = model.evaluate(val_ds)
    labels, preds = iam_dataset.predict(val_ds, model)
    metrics['accuracy'] = calculate_accuracy(preds, labels)
    metrics['wer'] = WordErrorRate()(preds, labels)
    metrics['cer'] = CharErrorRate()(preds, labels)

    print(f"Evaluation loss: {metrics['loss']:.0f}")
    print(f"Character error rate: {100*metrics['cer']:.1f} %")
    print(f"Word error rate: {100*metrics['wer']:.0f} %")
    print(f"Precisión: {100*metrics['accuracy']:.0f} %")
    with open(str(experiment_path/'metrics.json'), 'w') as f:
        json.dump(metrics, f, default=custom_serialize)