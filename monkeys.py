# Applying cifar10 project to monkey data
import os
from pathlib import Path
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD

# GPU Acceleration
from keras import backend as K
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
K.tensorflow_backend._get_available_gpus()

cwd = os.getcwd()
training_dir = Path(cwd + '/rgb/training/')
validation_dir = Path(cwd + '/rgb/validation')
labels_dir = pd.read_csv(cwd + '/monkey_labels.csv')
labels_df = pd.DataFrame(labels_dir, columns=['Label', 'Latin Name', 'Common Name', 'Train Images', 'Validaiton Images'])
label = labels_df["Common Name"]

# Setting Variables
rescale = 1.0/255.0
rot_range = 40
shift = 0.2
wid_hgt = 64
batch = 32
dim = 3
train_n = 1097
test_n = 272

#  Image Data Generators
train_imgdatagen = ImageDataGenerator(
                    rescale=rescale,
                    rotation_range=rot_range,
                    width_shift_range=shift,
                    height_shift_range=shift,
                    shear_range=shift,
                    zoom_range=shift,
                    horizontal_flip=True)

test_imgdatagen = ImageDataGenerator(rescale=rescale)

# Generators
train_generator = train_imgdatagen.flow_from_directory(
                    training_dir,
                    target_size=(wid_hgt, wid_hgt),
                    batch_size=batch,
                    shuffle=True,
                    class_mode='categorical',)

test_generator = test_imgdatagen.flow_from_directory(
                    validation_dir,
                    target_size=(wid_hgt, wid_hgt),
                    batch_size=batch,
                    shuffle=False,
                    class_mode='categorical')


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(wid_hgt, wid_hgt, dim), activation='relu', padding='same',
                 kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=2048, activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(rate=0.5))

model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer=SGD(lr=.01), loss='categorical_crossentropy', metrics=['accuracy'])


# Needs to use fit_generators instead of 'fit'
model.fit_generator(train_generator,
                    steps_per_epoch=np.ceil(train_n/batch),   # Number of train samples divided by batch size
                    epochs=50,
                    verbose=1,                                     # 1 = progress bar
                    validation_data=test_generator,
                    validation_steps=np.ceil(test_n/batch))   # Number of test samples divided by batch size

cwd = os.getcwd()
model.save(filepath=cwd + "/monkey_classifier.h5")
