from pathlib import Path
import random
import math

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow.data as tfd
from tensorflow.keras.layers import StringLookup
from sklearn.model_selection import train_test_split
from PIL import Image

class HWR_dataset(ABC):
    def __init__(self, path, img_size, preserve_aspect_ratio, img_negative):
        self.path = path
        self.img_size = img_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.img_negative = img_negative

    def load_image(self, image_path):
        """
        This function gets the image path and 
        reads the image using TensorFlow, Then the image will be decoded and 
        will be converted to float data type. next resize and transpose will be applied to it.
        In the final step the image will be converted to a Numpy Array using tf.cast
        """
        # read the image
        image = tf.io.read_file(image_path)
        # decode the image
        decoded_image = tf.image.decode_jpeg(contents=image, channels=1)
        # convert image data type to float32
        convert_imgs = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)
        # resize and transpose 
        resized_image = tf.image.resize(
            images=convert_imgs, 
            size=self.img_size[::-1],
            preserve_aspect_ratio=self.preserve_aspect_ratio)
        if self.preserve_aspect_ratio:
            pad_height = self.img_size[1] - tf.shape(resized_image)[0]
            pad_width = self.img_size[0] - tf.shape(resized_image)[1]            
            if pad_height % 2 != 0:
                height = pad_height // 2
                pad_height_top = height + 1
                pad_height_bottom = height
            else:
                pad_height_top = pad_height_bottom = pad_height // 2
            if pad_width % 2 != 0:
                width = pad_width // 2
                pad_width_left = width + 1
                pad_width_right = width
            else:
                pad_width_left = pad_width_right = pad_width // 2
            flattened_image = tf.reshape(resized_image, [-1])
            unique_values, _, counts = tf.unique_with_counts(flattened_image)
            pad_value = unique_values[tf.argmax(counts)]
            resized_image = tf.pad(
                resized_image,
                paddings=[
                    [pad_height_top, pad_height_bottom],
                    [pad_width_left, pad_width_right],
                    [0, 0],
                ],
                constant_values=pad_value
            )
        image = tf.transpose(resized_image, perm = [1, 0, 2])

        # to numpy array (Tensor)
        image_array = tf.cast(image, dtype=tf.float32)
        if self.img_negative:
            image_array = 1.0 - image_array

        return image_array 

    def encode_single_sample(self, image_path, label:str):    
        '''
        The function takes an image path and label as input and returns a dictionary containing the processed image tensor and the label tensor. 
        First, it loads the image using the load_image function, which decodes and resizes the image to a specific size. Then it converts the given
        label string into a sequence of Unicode characters using the unicode_split function. Next, it uses the self.char_to_num layer to convert each
        character in the label to a numerical representation. It pads the numerical representation with a special class (self.n_classes)
        to ensure that all labels have the same length (self.max_label_length). Finally, it returns a dictionary containing the processed image tensor
        and the label tensor.
        
        '''    
        # Get the image
        image = self.load_image(image_path)
        # Convert the label into characters
        chars = tf.strings.unicode_split(label, input_encoding='UTF-8')
        # Convert the characters into vectors
        vecs = self.char_to_num(chars)
        
        # Pad label
        pad_size = self.max_label_length - tf.shape(vecs)[0]
        if pad_size > 0:
            vecs = tf.pad(vecs, paddings = [[0, pad_size]], constant_values=self.n_classes+1)
        else:
            vecs = vecs[:self.max_label_length]
        
        return {'image':image, 'label':vecs}   

    def decoder_prediction(self, pred_label):
        """
        This function has the job to decode the prediction that the model had.
        The model predicts each character and then this function makes it readable. 
        """
        # Input length
        input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]
        
        # CTC decode
        decode = tf.keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=True)[0][0][:,:self.max_label_length]
        
        # Converting numerics back to their character values
        chars = self.num_to_char(decode)
        # Join all the characters
        texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]
        
        # Remove the unknown token
        filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]
        
        return filtered_texts

    def predict(self, data, model):
        """
        This function ask the model for the predictions of the data, returning
        the labels and the predictions decoded.
        """
        y, y_pred = [], []

        # Loading Data 
        iterador = iter(data)
        for batch in iterador:
            images, labels = batch['image'], batch['label']
            
            # Iterate over the data 
            for index, (image, label) in enumerate(zip(images, labels)):
                # Label processing
                text_label = self.num_to_char(label)
                text_label = tf.strings.reduce_join(text_label).numpy().decode('UTF-8')
                text_label = text_label.replace("[UNK]", " ").strip()
                y.append(text_label)

        y_pred = model.predict(data)
        y_pred = self.decoder_prediction(y_pred)
    
        return y, y_pred    

        @abstractmethod
        def get_partitions(self, batch_size):
            pass

        @abstractmethod
        def _load(self):
            pass

class IAM_dataset(HWR_dataset):
    def _load(self, extra_spaces):
        rows = []
        with open(self.path/'ascii/lines_spaces.txt', 'r') as f:
            for line in f:
                if line and not line.startswith('#'):
                    line = line.strip()
                    line = line.split(' ')
                    if line[1] == 'ok':
                        line_id = line[0]
                        doc, doc_line, form_line  = line[0].split('-')
                        form = doc + '-' + doc_line
                        path = str(self.path/'lines'/doc/form/(line[0]+'.png'))
                        label = ' '.join(line[8:])
                        rows.append([form, line_id, path, label])    
        columns = ['form','line', 'path', 'label']
        df = pd.DataFrame(rows, columns=columns)
        if extra_spaces == 'all':
            df['label'] = ' ' + df['label'] + ' '
        elif  extra_spaces == 'selective':
            sizes = [Image.open(path).size for path in df['path'].tolist()]  
            sizes_df = pd.DataFrame(sizes, columns=['Lx', 'Ly'])
            mask = sizes_df["Ly"] > (128/1024 * sizes_df["Lx"])
            df['label'][mask] = ' ' + df['label'][mask] + ' '
        return df

    def __init__(self, path, size, seed=42, reduced=False, test_size=0.2, val_size=0.2,
                 task=False, include_all=False, preserve_aspect_ratio=False, 
                 extra_spaces=False, img_negative=False, augmentation_factor=0):
        '''
        path: Roort directory of dataset.
        size: Size of images.
        seed=42: Random seed.
        reduced=False: Selecsts one line for formulary.
        test_size=0.2: Fraction of dataset destined to test.
        val_size=0.2: (1-test_size)*val_size is the fraction of datasewt destined to validation.
        task=False: Selects the partitions
        include_all=False: train contaisn all forms that are not in validation or test
        '''
        super().__init__(path, size, preserve_aspect_ratio, img_negative)
        self.df = self._load(extra_spaces).sample(frac=1, random_state=seed)
        if reduced:
            self.df['form'] = self.df['line'].str.split('-').str[:2].str.join('-')
            self.df = self.df.groupby(['form']).first().reset_index(drop=True)
        if task:
            X_val1, y_val1 = self.__select_taskset('validationset1.txt')
            X_val2, y_val2 = self.__select_taskset('validationset2.txt')
            self.X_val = X_val1 + X_val2
            self.y_val = y_val1 + y_val2
            self.X_test, self.y_test = self.__select_taskset('testset.txt')
            if include_all:
                val_forms = set(form.split('\\')[-2] for form in self.X_val)
                test_forms = set(form.split('\\')[-2] for form in self.X_test)
                train_df = self.df[~self.df['form'].isin(val_forms | test_forms)]             
                self.X_train = train_df['path'].tolist()
                self.y_train = train_df['label'].tolist()
            else:
                self.X_train, self.y_train = self.__select_taskset('trainset.txt')
        else:                                                                  
            X = self.df['path'].tolist()
            y = self.df['label'].tolist()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_size, shuffle=False)
        self.__data_augmentation(augmentation_factor)
        self.y_test = [label.strip() for label in self.y_test] #No extra spaces in test
        self.unique_characters = set(char for word in self.y_train for char in word)
        self.unique_characters = sorted(list(self.unique_characters))
        self.n_classes = len(self.unique_characters)
        self.char_to_num = StringLookup(vocabulary=list(self.unique_characters), mask_token=None)
        self.num_to_char = StringLookup(vocabulary = self.char_to_num.get_vocabulary(), mask_token = None, invert = True)
        self.max_label_length = max(map(len, self.df['label']))

    def __select_taskset(self, file):
        with open(self.path/'task'/file, 'r') as f:
            lines = f.read().splitlines()
        df = self.df[self.df['line'].isin(lines)]
        X = df['path'].tolist()
        y = df['label'].tolist()
        return X, y

    def get_partitions(self, batch_size):   
        train_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(self.X_train), np.array(self.y_train))
        ).map(
            lambda x, y: self.encode_single_sample(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(self.X_val), np.array(self.y_val))
        ).map(
            lambda x, y: self.encode_single_sample(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tfd.AUTOTUNE)
        test_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(self.X_test), np.array(self.y_test))
        ).map(
            lambda x, y: self.encode_single_sample(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tfd.AUTOTUNE)
        return train_ds, val_ds, test_ds

    def __data_augmentation(self, factor):
        tam_X_train = len(self.X_train)
        iteration = 1
        if type(factor) == float:
            num_samples = int(factor*tam_X_train)
        else:
            num_samples = factor            
        (self.path/'data augmentation').mkdir(parents=True, exist_ok=True)

        while num_samples > 0:
            if num_samples > tam_X_train:
                img_paths  = self.X_train
            else:
                img_paths  = self.X_train[:num_samples]
            for origin_path, label in zip(img_paths, self.y_train):
                with Image.open(origin_path) as img:
                    target_path = self.path/'data augmentation'/(Path(origin_path).stem + f'_{iteration}.png')
                    if not target_path.exists():
                        img_array = np.array(img)
                        fill_color = int(np.bincount(img_array.flatten()).argmax())

                        w , h = img.size
                        shear_factor = random.uniform(-1, 1)
                        new_width = int(w + abs(shear_factor)*h)
                        if shear_factor > 0:
                            skewed_img = img.transform((new_width,h), Image.AFFINE, (1, shear_factor, -shear_factor*h, 0, 1, 0), fillcolor=fill_color)
                        else:
                            skewed_img = img.transform((new_width,h), Image.AFFINE, (1, shear_factor, 0, 0, 1, 0), fillcolor=fill_color)

                        angle = random.uniform(-10, 10)
                        rotated_img = skewed_img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill_color)
                                
                        rotated_img.save(target_path)
                self.X_train.append(str(target_path))
                self.y_train.append(label)
            num_samples -= tam_X_train
            iteration += 1

class Written_names_dataset(HWR_dataset):
    def _load(self):
        rows = []
        with open(self.path/'ascii/lines_spaces.txt', 'r') as f:
            for line in f:
                if line and not line.startswith('#'):
                    line = line.strip()
                    line = line.split(' ')
                    if line[1] == 'ok':
                        line_id = line[0]
                        doc, doc_line, form_line  = line[0].split('-')
                        form = doc + '-' + doc_line
                        path = str(self.path/'lines'/doc/form/(line[0]+'.png'))
                        label = ' '.join(line[8:])
                        rows.append([form, line_id, path, label])
    
        columns = ['form','line', 'path', 'label']
        return pd.DataFrame(rows, columns=columns)

    def __init__(self, path, size, preserve_aspect_ratio=False, img_negative=False):
        super().__init__(path, size, preserve_aspect_ratio, img_negative)
        self.train_csv = pd.read_csv(path/'CSV/train.csv')
        self.test_csv = pd.read_csv(path/'CSV/test.csv')
        self.val_csv = pd.read_csv(path/'CSV/validation.csv')
        self.train_csv['FILENAME'] = [str(path/'train'/filename) for filename in self.train_csv['FILENAME']]
        self.val_csv['FILENAME'] = [str(path/'validation'/filename) for filename in self.val_csv['FILENAME']]
        self.test_csv['FILENAME']  = [str(path/'test'/filename) for filename in self.test_csv['FILENAME']]
        train_labels = [str(word) for word in self.train_csv["IDENTITY"].to_numpy()]
        self.unique_characters = set(char for word in train_labels for char in word)
        self.unique_characters = sorted(list(self.unique_characters))
        self.n_classes = len(self.unique_characters)
        self.char_to_num = StringLookup(vocabulary=list(self.unique_characters), mask_token=None)
        self.num_to_char = StringLookup(vocabulary = self.char_to_num.get_vocabulary(), mask_token = None, invert = True)
        self.max_label_length = max(map(len, train_labels))

    def get_partitions(self, batch_size, seed):  
        train_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(self.train_csv['FILENAME'].to_list()), np.array(self.train_csv['IDENTITY'].to_list()))
        ).shuffle(seed).map(
            lambda x, y: self.encode_single_sample(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tfd.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(self.val_csv['FILENAME'].to_list()), np.array(self.val_csv['IDENTITY'].to_list()))
        ).map(
            lambda x, y: self.encode_single_sample(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tfd.AUTOTUNE)
        test_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(self.test_csv['FILENAME'].to_list()), np.array(self.test_csv['IDENTITY'].to_list()))
        ).map(
            lambda x, y: self.encode_single_sample(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tfd.AUTOTUNE)
        return train_ds, val_ds, test_ds