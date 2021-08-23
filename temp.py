import os

current_path = os.path.abspath(os.getcwd())
data_path = current_path + '/figure'
csv_path = current_path + '/train.csv'
benign = data_path + '/benign'
malignant = data_path + '/malignant'

files = os.listdir(data_path)

import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
#from tensorflow.keras.optimizers import Nadame
from keras.utils import np_utils,get_file
from keras.preprocessing import image
import matplotlib.pyplot as plt
K.image_data_format() == 'channels_last'


def image_resize(image, size):
    new_image = cv.resize(image, size, interpolation=cv.INTER_AREA)
    return new_image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def processing_data(data_path, height, width, batch_size=32, test_split=0.1):


    train_data = ImageDataGenerator(

            rescale=1. / 255,  

            shear_range=0.1,  

            zoom_range=0.1,

            width_shift_range=0.1,

            height_shift_range=0.1,

            horizontal_flip=True,

            vertical_flip=True,

            validation_split=test_split  
    )


    test_data = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=test_split)

    train_generator = train_data.flow_from_directory(

            data_path, 

            target_size=(height, width),

            batch_size=batch_size,

            class_mode='binary',
            classes=['benign','malignant'],

            subset='training', 
            seed=0)
    test_generator = test_data.flow_from_directory(
            data_path,
            target_size=(height, width),
            batch_size=batch_size,
            class_mode='binary',
            classes=['benign','malignant'],
            subset='validation',
            seed=0)

    return train_generator, test_generator

def save_model(model, checkpoint_save_path, model_dir):
    """
    保存模型，每迭代3次保存一次
    :param model: 训练的模型
    :param checkpoint_save_path: 加载历史模型
    :param model_dir: 
    :return: 
    """
    if os.path.exists(checkpoint_save_path):
        print("loading model")
        model.load_weights(checkpoint_save_path)
        print("loading is finished")
    checkpoint_period = ModelCheckpoint(

        model_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',

        monitor='accuracy',

        #mode='max',

        save_weights_only=False,

        save_best_only=True,

        period=3
    )
    return checkpoint_period

from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model

def model():
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def model2():
    model = models.Sequential()
    model.add(Conv2D(32, (11, 11), activation='relu', input_shape=(500, 500, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64,(7,7),activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

model_name = 'mobilenet_1_0_224_tf_no_top.h5'
weights_path = current_path + '/weight/' + model_name


#model = model()

import tensorflow as tf

model = model()



model.compile(optimizer = "SGD", loss = 'binary_crossentropy', metrics=['accuracy'])
print(weights_path)
model.load_weights(weights_path,by_name=True)
checkpoint_save_path = "results/temp.h5"
model_dir = "results/"
checkpoint_period = save_model(model, checkpoint_save_path, model_dir)

reduce_lr = ReduceLROnPlateau(
                        monitor='accuracy',  #
                        factor=0.5,     # 
                        patience=3,     # 
                        verbose=1       # 
                    )
early_stopping = EarlyStopping(
                            monitor='val_loss',  
                            min_delta=0,         
                            patience=10,         
                            verbose=2           
                        )
batch_size = 8





train_generator,test_generator = processing_data(data_path, height=500, width=500, batch_size=batch_size, test_split=0.2)

model.compile(loss='binary_crossentropy',  
              optimizer= SGD(learning_rate=0.001),           
              metrics=['accuracy'])        

history = model.fit_generator(train_generator,    
                    epochs=180,
                    
                    steps_per_epoch= 904// batch_size,
                    validation_data=test_generator,
                    validation_steps=max(1, 224//batch_size),
                    initial_epoch=120,
                    callbacks=[checkpoint_period, reduce_lr])

model.save_weights(model_dir + 'temp.h5')


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim([0.5, 1])
plt.show()


