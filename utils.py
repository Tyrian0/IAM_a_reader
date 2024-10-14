from pathlib import Path
import random
import sys
import os
import json
import csv

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as implt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import tensorflow as tf
import tensorflow.data as tfd
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
from torchmetrics.text import CharErrorRate, WordErrorRate

from ctc_layer import CTCLayer, WeightedCTCLayer
from hwr_dataset import IAM_dataset
from callbacks import SavePredictions

def load_hyperparameters(path):
    with open(path/'hyperparameters.json') as f:
        hyperparameters = json.load(f)
    hyperparameters['experiment_path'] = path
    hyperparameters['dataset_path'] = Path(hyperparameters['dataset_path']) 
    return hyperparameters

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

def find_current_lr(path, patience):    
    current_lr = 0.001
    if not patience:
        return current_lr
    log_path = path / "training.log"  
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    lr_reductions = 0
    
    with open(log_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            val_loss = float(row['val_loss'])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                current_lr /= 10
                epochs_without_improvement = 0    
    return current_lr

def find_current_epoch(path):
    log_path = path / "training.log"    
    with open(log_path, 'r') as f:
        last_epoch = int(pd.read_csv(f, sep=',').iloc[-1,0])
    return last_epoch + 1 #Log starts with 0

def perform_experiment(model=None, initial_epoch=0, **kwargs):
    '''
    This function encapsules all the logic begind of a experiment.
    '''
    experiment_path = kwargs.get('experiment_path', Path('.'))
    seed = kwargs.get('seed', 42)
    keras_seed = kwargs.get('keras_seed', seed)
    epochs = kwargs.get('epochs', 200)
    img_width = kwargs.get('img_width', 1024)
    img_height = kwargs.get('img_height', 128)     
    img_size = (img_width, img_height)
    dataset_path = kwargs.get('dataset_path', Path(r'data\IAM'))
    task = kwargs.get('task', False)
    include_all = kwargs.get('include_all', False)
    reduced = kwargs.get('reduced', True)
    batch_size = kwargs.get('batch_size', 16)
    dropout_CNN = kwargs.get('dropout_CNN', 0.1)
    dropout_MLP = kwargs.get('dropout_MLP', 0.4)
    dropout_RNN = kwargs.get('dropout_RNN', 0.4)
    CNN = kwargs.get('CNN', ((64, 32), (64, 32)))
    MLP = kwargs.get('MLP', (128,))
    RNN = kwargs.get('RNN', (128, 64, 64))
    negative_slope = kwargs.get('negative_slope', 0)
    earlyStopping_patience = kwargs.get('earlyStopping_patience', 10)
    reduceLROnPlateau_patience = kwargs.get('reduceLROnPlateau_patience', False)
    savePredictions = kwargs.get('savePredictions', False)
    batchNormalization = kwargs.get('batchNormalization', False)
    preserve_aspect_ratio = kwargs.get('preserve_aspect_ratio', False)
    extra_spaces = kwargs.get('extra_spaces', False)
    img_negative = kwargs.get('img_negative', False)
    ctc_shortcut = kwargs.get('ctc_shortcut', False)
    flattening_method = kwargs.get('flattening_method', 'reshape')
    augmentation_factor = kwargs.get('augmentation_factor', 0)

    experiment_path.mkdir(parents=True, exist_ok=True)
    with open(str(experiment_path/'hyperparameters.json'), 'w') as f:
        json.dump(kwargs, f, default=custom_serialize)

    tf.keras.utils.set_random_seed(keras_seed)
    
    # Dataset load
    iam_dataset = IAM_dataset(dataset_path, img_size, reduced=reduced, 
                              task=task, include_all=include_all, seed=seed, 
                              preserve_aspect_ratio=preserve_aspect_ratio,
                              extra_spaces=extra_spaces, img_negative=img_negative,
                              augmentation_factor=augmentation_factor)
    train_ds, val_ds, test_ds = iam_dataset.get_partitions(batch_size)

    # Model definition
    if model:
        append_log = True
    else:
        append_log = False
        input_images = Input(shape=(img_width, img_height, 1), name='image')
        input_labels = Input(shape=(None, ), name='label')
        x = input_images
        for i, block in enumerate(CNN):
            for j, layer in enumerate(block):
                x = Conv2D(layer, 3, strides=1, padding='same', kernel_initializer='he_normal', name=f'conv_{i}_{j}')(x)
                if batchNormalization:
                    x = BatchNormalization()(x)   
                x = LeakyReLU(negative_slope)(x)
                x = Dropout(dropout_CNN, name=f'conv_{i}_{j}_dropout')(x)     
            if i < len(CNN) -1 or flattening_method == 'reshape':    
                x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=f'MaxPool_{i}')(x)
        if flattening_method == 'reshape':
            target_shape = (img_width // 2**len(CNN), CNN[-1][-1] * img_height // 2**len(CNN))
        elif flattening_method == 'maxPooling':
            pool_size = (1, img_height // 2**(len(CNN)-1))
            x = MaxPool2D(pool_size=pool_size, strides=(1, 1), name='MaxPool_columns')(x)    
            target_shape = (-1, x.shape[3]) 
        output_cnn = Reshape(target_shape=target_shape, name='reshape_layer')(x)
        x = output_cnn
        for i, layer in enumerate(MLP):
            x = Dense(layer, kernel_initializer='he_normal', activation='relu', name=f'MLP_{i}')(x)
            x = Dropout(dropout_MLP, name=f'MLP_{i}_dropout')(x)
        output_MLP = x
        for i, layer in enumerate(RNN):
            x = Bidirectional(LSTM(layer, return_sequences=True, dropout=dropout_RNN), name=f'bidirectional_lstm_{i}')(x)
        output_rnn = Dense(len(iam_dataset.char_to_num.get_vocabulary())+1, activation='softmax', name='output_RNN')(x)
        ctc_loss_rnn = CTCLayer()(input_labels, output_rnn)
        if ctc_shortcut:
            output_shortcut = Conv1D(len(iam_dataset.char_to_num.get_vocabulary())+1, 3, padding='same', activation='softmax', name='output_shortcut')(output_MLP)
            ctc_loss_shortcut = WeightedCTCLayer(0.1)(input_labels, output_shortcut)
            model = Model(inputs=[input_images, input_labels], outputs=[ctc_loss_rnn, ctc_loss_shortcut])
        else:
            model = Model(inputs=[input_images, input_labels], outputs=[ctc_loss_rnn])
        model.compile(optimizer='adam')
    draw_DNN(**kwargs)

    # Model taining  
    callbacks = [
        EarlyStopping(patience=earlyStopping_patience, restore_best_weights=True),
        ModelCheckpoint(filepath=str(experiment_path/'model.keras'), save_best_only=True),
        CSVLogger(str(experiment_path/'training.log'), append=append_log)
    ]
    if savePredictions:
        callbacks.append(SavePredictions(iam_dataset, val_ds, save_dir=str(experiment_path/'predictions')))
    if reduceLROnPlateau_patience:
        callbacks.append(ReduceLROnPlateau(patience=reduceLROnPlateau_patience))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, initial_epoch=initial_epoch)

    # Training visualization
    history_df = pd.read_csv(str(experiment_path/'training.log'))
    sns.set_style('white')
    sns.set_palette('deep')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    history_df.plot(y=['loss', 'val_loss'], ax=ax)
    ax.legend(['Entrenamiento', 'Validación'], fontsize=12)
    ax.set_ylabel('Pérdidas CTC')
    ax.set_xlabel('Época')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig(str(experiment_path/'model_result.png'), facecolor='white', edgecolor='none')
    plt.show()

    # Metrics visualization
    pred_model = Model(inputs=model.get_layer(name="image").output, outputs=model.get_layer(name='output_RNN').output)
    if ctc_shortcut:
        eval_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        eval_model.compile(optimizer='adam')
    else:
        eval_model = model
    metrics = {}
    metrics['loss'] = eval_model.evaluate(val_ds)
    labels, preds = iam_dataset.predict(val_ds, pred_model)
    metrics['accuracy'] = calculate_accuracy(preds, labels)
    metrics['wer'] = WordErrorRate()(preds, labels)
    metrics['cer'] = CharErrorRate()(preds, labels)

    print(f"Evaluation loss: {metrics['loss']:.0f}")
    print(f"Character error rate: {100*metrics['cer']:.1f} %")
    print(f"Word error rate: {100*metrics['wer']:.0f} %")
    print(f"Precisión: {100*metrics['accuracy']:.0f} %")
    with open(str(experiment_path/'metrics.json'), 'w') as f:
        json.dump(metrics, f, default=custom_serialize)
        
def continue_experiment(path, epochs):
    '''
    This function encapsules all the logic to continue experiment.
    '''
    hyperparameters = load_hyperparameters(path) 
    
    current_epoch = find_current_epoch(path)   

    hyperparameters['epochs'] = current_epoch + epochs    
    patience = hyperparameters.get('reduceLROnPlateau_patience', False)
    learning_rate = find_current_lr(path, patience)
    model = load_model(str(path/'model.keras'))
    model.optimizer.learning_rate.assign(learning_rate)
    perform_experiment(model=model, initial_epoch=current_epoch, **hyperparameters)

def draw_layer(ax, centers, sizes, color, label='', linestyle='solid'):
    x_center, y_center, z_center = centers
    x_size,  y_size, z_size = sizes
    x_center += x_size/2
    vertexes = [[x_center - x_size/2, y_center - y_size/2, z_center - z_size/2], 
                [x_center + x_size/2, y_center - y_size/2, z_center - z_size/2],
                [x_center + x_size/2, y_center + y_size/2, z_center - z_size/2], 
                [x_center - x_size/2, y_center + y_size/2, z_center - z_size/2],
                [x_center - x_size/2, y_center - y_size/2, z_center + z_size/2], 
                [x_center + x_size/2, y_center - y_size/2, z_center + z_size/2],
                [x_center + x_size/2, y_center + y_size/2, z_center + z_size/2], 
                [x_center - x_size/2, y_center + y_size/2, z_center + z_size/2]]
    edges = [[vertexes[j] for j in [0, 1, 2, 3]],
             [vertexes[j] for j in [4, 5, 6, 7]],
             [vertexes[j] for j in [0, 1, 5, 4]],
             [vertexes[j] for j in [2, 3, 7, 6]],
             [vertexes[j] for j in [1, 2, 6, 5]],
             [vertexes[j] for j in [4, 7, 3, 0]]]
    layer = Poly3DCollection(edges, facecolors='white', linewidths=1, edgecolors=color, alpha=1, linestyle=linestyle)
    ax.add_collection3d(layer)
    ax.text(x_center, y_center, z_size/2, label, 'y', color='black', 
                    fontsize=10, ha='center', va='center', fontdict={'family': 'Arial'})

def draw_DNN(space_layer=10, image_path=None, space_image=10, label=None, space_label=40, **kwargs):
    experiment_path = kwargs.get('experiment_path', Path('.'))
    seed = kwargs.get('seed', 42)
    img_width = kwargs.get('img_width', 1024)
    img_height = kwargs.get('img_height', 128)     
    img_size = (img_width, img_height)    
    max_size = max(img_width, img_height)  
    dataset_path = kwargs.get('dataset_path', Path(r'data\IAM'))
    task = kwargs.get('task', False)
    include_all = kwargs.get('include_all', False)
    reduced = kwargs.get('reduced', True)
    CNN = kwargs.get('CNN', ((64, 32), (64, 32)))
    MLP = kwargs.get('MLP', (128,))
    RNN = kwargs.get('RNN', (128, 64, 64))
    preserve_aspect_ratio = kwargs.get('preserve_aspect_ratio', False)
    extra_spaces = kwargs.get('extra_spaces', False)
    img_negative = kwargs.get('img_negative', False)
    ctc_shortcut = kwargs.get('ctc_shortcut', False)
    flattening_method = kwargs.get('flattening_method', 'reshape')

    iam_dataset = IAM_dataset(dataset_path, img_size, reduced=reduced, 
                              task=task, include_all=include_all, seed=seed, 
                              preserve_aspect_ratio=preserve_aspect_ratio,
                              extra_spaces=extra_spaces, img_negative=img_negative)

                              
    
    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    # Draw input image
    if image_path is not None:
        pos = -CNN[0][0]/2 - space_image
        image = iam_dataset.load_image(image_path)
        image = tf.transpose(image, perm = [1, 0, 2])
        image = tf.squeeze(image, axis=2).numpy()
        image = np.flipud(image) #Corrects the z inversion of mapig to mgrid
        colors = plt.get_cmap('grey')(image)
        nz, ny = image.shape
        zi, yi = np.mgrid[-nz/2:nz/2 + 1, -ny/2:ny/2 + 1]
        xi = np.full_like(yi, pos)
        ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, facecolors=colors, shade=False)

    # Draw DNN
    x_center = 0
    layer_color_set =  {'brown', 'green'}
    for i, block in enumerate(CNN):
        for filters in block:
            layer_label = rf'{filters} $\times$ (3, 3)'
            draw_layer(ax, (x_center, 0, 0), (filters,  img_width, img_height), 'black', layer_label)
            layer_color_set.add('black')
            x_center += filters + space_layer
        if i < len(CNN) -1 or flattening_method == 'reshape':   
            img_width //= 2
            img_height //= 2
            draw_layer(ax, (x_center, 0, 0), (img_height,  img_width, img_height), 'red')
            layer_color_set.add('red')
            x_center += img_height + space_layer
    if flattening_method == 'reshape':
        draw_layer(ax, (x_center, 0, 0), (filters*img_height, img_width, 1), 'purple')
        layer_color_set.add('purple')
        x_center += filters*img_height + space_layer
    elif flattening_method == 'maxPooling':
        draw_layer(ax, (x_center, 0, 0), (filters, img_width, 1), 'olive')
        layer_color_set.add('olive')
        x_center += filters + space_layer
    for neurons in MLP:
        layer_label = f'{neurons}'
        draw_layer(ax, (x_center, 0, 0), (neurons, img_width, 1), 'orange', layer_label)
        layer_color_set.add('orange')
        x_center += neurons + space_layer
    if ctc_shortcut:
        y_center = (max(RNN) + space_layer)/2
        x_size = len(iam_dataset.char_to_num.get_vocabulary())+1
        draw_layer(ax, (x_center, y_center, 0), (x_size, img_width, 1), 'cyan', linestyle='dotted')
        layer_color_set.add('cyan')
        x_center_shortcut = x_center + x_size + space_layer
        draw_layer(ax, (x_center_shortcut, y_center, 0), (x_size, img_width, 1), 'brown', linestyle='dotted')
        x_center_shortcut += x_size + space_layer    
        draw_layer(ax, (x_center_shortcut, y_center, 0), (1, iam_dataset.max_label_length, 1), 'green', linestyle='dotted')
    else:
        y_center = 0 
    for cells in RNN:
        layer_label = rf'2 $\times$ {cells}'
        draw_layer(ax, (x_center, -y_center, 0), (2*cells, img_width, 1), 'blue', layer_label)
        layer_color_set.add('blue')
        x_center += 2*cells + space_layer
    x_size = len(iam_dataset.char_to_num.get_vocabulary())+1
    draw_layer(ax, (x_center, -y_center, 0), (x_size, img_width, 1), 'brown')
    x_center += x_size + space_layer    
    draw_layer(ax, (x_center, -y_center, 0), (1, iam_dataset.max_label_length, 1), 'green')
    x_center += 1

    # Draw output label
    x_center += space_label
    if label:
        ax.text(x_center, 0, 0, label, 'y', color='black', 
                fontsize=15, ha='center', va='center', fontdict={'family': 'Arial'})
    
    # Ajust axis aspect ratio  
    ax.set_xlim([0, x_center])
    ax.set_ylim([-max_size/2, max_size/2])
    ax.set_zlim([-max_size/2, max_size/2])
    ax.set_axis_off()

    # Draw legend
    legend = {'black': 'Convolucional 2D', 
              'cyan': 'Convolucional 1D', 
              'red': r'$\it Max\ pooling$', 
              'purple': 'Redimension',
              'olive': r'$\it Max\ pooling$ por columnas',
              'orange': 'Densa',
              'brown': 'Softmax',
              'blue': 'BiLSTM',
              'green': 'Decodificador CTC'}
    size = 32
    x_center, y_center, z_center = x_center/2, max_size/2, 0
    for color in layer_color_set:
        draw_layer(ax, (x_center, y_center, z_center), (size, size, size), color)
        ax.text(x_center + size + space_layer, y_center + size/2, z_center, legend[color], color=color, 
                    fontsize=10, ha='left', va='center', fontdict={'family': 'Arial'})
        z_center += size + space_layer


    plt.savefig(str(experiment_path/'model.png'), facecolor='white', edgecolor='none')
    plt.show()