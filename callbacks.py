
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model

class SavePredictions(Callback):
    def __init__(self, dataset, data, save_dir='predictions'):
        super(SavePredictions, self).__init__()
        self.dataset = dataset
        self.data = data
        self.save_dir = save_dir
        
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

    def on_epoch_end(self, epoch, logs=None):
        _, predictions = self.dataset.predict(self.data, self.model)
        
        with open(os.path.join(self.save_dir, f'epoch_{epoch+1}.txt'), 'w', encoding='utf-8') as f:
            for prediction in predictions:
                f.write(prediction + '\n')