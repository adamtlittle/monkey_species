# Applying cifar10 project to monkey data
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

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


model = load_model(filepath=cwd + "/monkey_classifier.h5")

predict = model.predict_generator(test_generator, steps=np.ceil(test_n/batch), verbose=1)
max_pred = np.argmax(predict, axis=1)

print(len(test_generator.labels))
print(len(max_pred))

# Function taken from sklearn documentation
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.BuPu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


plot_confusion_matrix(y_true=test_generator.classes, y_pred=max_pred, normalize=True, classes=label, title="Monkey Species Conf. Matrix")

plt.show()