import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
import os
import matplotlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers import Nadame
from keras.utils import np_utils,get_file
import os
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing import image

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
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model
weights_path = "results/temp.h5"
image_path = 'ISIC_0624498.jpg'   #'ISIC_0149568.jpg'
my_model = model()
#my_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
my_model.load_weights(weights_path)

image_pre  = image.load_img(image_path,target_size=(500,500))
image_pre = image.img_to_array(image_pre)
image_pre = np.expand_dims(image_pre, axis = 0)




print(my_model.predict(image_pre))