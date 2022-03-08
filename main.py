import os
import random
import numpy as np
import re
from pyedflib import highlevel
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import tensorflow as tf

"""Preprocessing
# Some basic data preprocessing includes obtaining signals in frequency domain using fft and shaping the data to ...
# arrange as labels and training+test data. Run all of this. Very important chunk.

# First we will read all the ".edf" and ".txt" files in the directory and stack them.
"""

path = "chb-mit-scalp-eeg-database-1.0.0"

edfFiles = []
txtFiles = []
# r = root, d = directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if file[-4:] == '.edf':
            edfFiles.append(os.path.join(r, file))
        elif file[-4:] == '.txt':
            txtFiles.append(os.path.join(r, file))

edfFiles = sorted(edfFiles)
txtFiles = sorted(txtFiles)

# for f in edfFiles:
#     print(f)
#
# for f in txtFiles:
#     print(f)

""" Reading EDF & TXT files and stacking them in batches."""


def generateLabels(edfFileName):
    sub = edfFileName[32:38]    # \chb01~\chb23
    # chb-mit-scalp-eeg-database-1.0.0/chb01/chb01-summary.txt
    filepath = 'chb-mit-scalp-eeg-database-1.0.0' + sub + sub + '-summary.txt'
    f = open(filepath, 'r')
    file_contents = f.read()    # Contents of chb01~chb23 Summary

    file_list = file_contents.split('\n')
    judge = edfFileName[42:44]
    if judge == '17':
        sub = edfFileName[39:48]
    else:
        sub = edfFileName[39:47]
    sub = 'File Name: ' + sub + '.edf'
    ind = file_list.index(sub)
    if judge == '24':
        seizures = list(map(int, re.findall(r'\d+', file_list[ind + 1])))[0]
    else:
        seizures = list(map(int, re.findall(r'\d+', file_list[ind + 3])))[0]
    start = []
    end = []

    if judge == '24':
        for i in range(seizures):
            start.append(list(map(int, re.findall(r'\d+', file_list[ind + 2 * i + 2])))[0])
            end.append(list(map(int, re.findall(r'\d+', file_list[ind + 2 * i + 3])))[0])
    else:
        for i in range(seizures):
            start.append(list(map(int, re.findall(r'\d+', file_list[ind + 2 * i + 4])))[0])
            end.append(list(map(int, re.findall(r'\d+', file_list[ind + 2 * i + 5])))[0])
        # print(start, end)

    if judge == '10':
        if seizures == 0:
            labels = np.zeros(7200)
        else:
            labels = np.ones(7200)
            labels[end[-1]:] *= 0
            for i in range(len(start)):
                labels[start[i]:end[i]] *= 2
    elif judge == '04' or judge == '06' or judge == '07' or judge == '09' or judge == '23':
        if seizures == 0:
            labels = np.zeros(14400)
        else:
            labels = np.ones(14400)
            labels[end[-1]:] *= 0
            for i in range(len(start)):
                labels[start[i]:end[i]] *= 2
    else:
        if seizures == 0:
            labels = np.zeros(3600)
        else:
            labels = np.ones(3600)
            labels[end[-1]:] *= 0
            for i in range(len(start)):
                labels[start[i]:end[i]] *= 2

    return labels


""" Shuffling and partitioning list"""

totalData = len(edfFiles)
random.shuffle(edfFiles)
partition = int(len(edfFiles) * 2 / 3)
edfFilesVal = edfFiles[partition:]
edfFilesTrain = edfFiles[:partition]
trainData = len(edfFilesTrain)
valData = len(edfFilesVal)

print(totalData, trainData, valData)

""" Frequency Domain"""


def stackDFTTrain(nbatch=2):
    count = 0

    stackedDFT = np.zeros((1, 23, 256, 3))
    stackedLabels = np.zeros((1))
    rejected = []

    while True:
        for f in edfFilesTrain:
            if stackedDFT.shape[0] >= nbatch * 3600 // 3 + 1:
                print(stackedLabels.shape)
                if stackedDFT[1:nbatch * 3600 // 3 + 1, :, :, :].shape == (
                        3600 * nbatch // 3, 23, 256, 3) and to_categorical(stackedLabels[1:nbatch * 3600 // 3 + 1],
                                                                           num_classes=3).shape == (
                        3600 * nbatch // 3, 3):
                    yield (stackedDFT[1:nbatch * 3600 // 3 + 1, :, :, :],
                           to_categorical(stackedLabels[1:nbatch * 3600 // 3 + 1], num_classes=3))
                stackedDFT = stackedDFT[nbatch * 3600 // 3:, :, :, :]
                stackedLabels = stackedLabels[nbatch * 3600:]
                print('extra', stackedDFT.shape, stackedLabels.shape)

            signals, signal_headers, header = highlevel.read_edf(f)
            if signals.shape[-1] % 3600 != 0 or signals.shape[0] != 23:
                rejected.append(f[54:59])
                continue

            count += 1
            print(f, signals.shape)
            s = int(signals.shape[1] / 256)
            signals = np.reshape(signals, (23, 256, 3, s // 3))
            signals = signals.transpose(3, 0, 1, 2)
            stackedDFT = np.concatenate((stackedDFT, np.fft.fft(signals, axis=1)), axis=0)
            genLabels = generateLabels(f)
            stackedLabels = np.concatenate((stackedLabels, genLabels), axis=-1)


def stackDFTVal(nbatch=1):
    count = 0

    stackedDFT = np.zeros((1, 23, 256, 3))
    stackedLabels = np.zeros((1))
    rejected = []

    while True:

        for f in edfFilesVal:
            # print(f[54:-4])
            if stackedDFT.shape[0] >= nbatch * 3600 // 3 + 1:
                if stackedDFT[1:nbatch * 3600 // 3 + 1, :, :, :].shape == (
                        3600 * nbatch // 3, 23, 256, 3) and to_categorical(stackedLabels[1:nbatch * 3600 // 3 + 1],
                                                                           num_classes=3).shape == (
                        3600 * nbatch // 3, 3):
                    yield (stackedDFT[1:nbatch * 3600 // 3 + 1, :, :, :],
                           to_categorical(stackedLabels[1:nbatch * 3600 // 3 + 1], num_classes=3))
                stackedDFT = stackedDFT[nbatch * 3600 // 3:, :, :, :]
                stackedLabels = stackedLabels[nbatch * 3600:]

            signals, signal_headers, header = highlevel.read_edf(f)
            if signals.shape[-1] % 3600 != 0 or signals.shape[0] != 23:
                rejected.append(f[54:59])
                continue

            count += 1
            print(f, signals.shape)
            s = int(signals.shape[1] / 256)
            signals = np.reshape(signals, (23, 256, 3, s // 3))
            signals = signals.transpose(3, 0, 1, 2)
            stackedDFT = np.concatenate((stackedDFT, np.fft.fft(signals, axis=1)), axis=0)
            genLabels = generateLabels(f)
            stackedLabels = np.concatenate((stackedLabels, genLabels), axis=-1)


""" Building a CNN for Frequency domain"""

nBatch = 2

in1 = Input(shape=(23, 256, 3))
c1 = Conv2D(16, (5, 5), activation='relu')(in1)
m1 = AveragePooling2D()(c1)
f1 = Flatten()(m1)
o = Dense(3, activation='sigmoid')(f1)

model = Model(inputs=in1, outputs=o)
print(model.summary())

""" Now after we have constructed oue model let's train it."""

model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
testSteps = int(trainData / 10)
valSteps = int(valData / 5)
history_cnn = model.fit_generator(generator=stackDFTTrain(),
                                  steps_per_epoch=testSteps,
                                  epochs=5,
                                  validation_data=stackDFTVal(),
                                  validation_steps=valSteps)

# When accuracy reaches ACCURACY_THRESHOLD
ACCURACY_THRESHOLD = 0.95


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('acc') > ACCURACY_THRESHOLD:
            print("\nReached %2.2f%% accuracy, so stopping training!!" % (ACCURACY_THRESHOLD * 100))
            self.model.stop_training = True


# Instantiate a callback object
callbacks = myCallback()

history_cnn.history

""" Results for Frequency domain"""

loss = history_cnn.history['loss']
val_loss = history_cnn.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.grid()
plt.axis([1, 5, 0, 5])
plt.plot(epochs, loss, '*y-', label='Training loss')
plt.plot(epochs, val_loss, '*r-', label='Validation loss')
plt.title('Training and validation loss for frequency domain')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history_cnn.history['accuracy']
val_acc = history_cnn.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.grid()
plt.axis([1, 5, 0, 1])
plt.plot(epochs, acc, '*y-', label='Training accuracy')
plt.plot(epochs, val_acc, '*r-', label='Validation accuracy')
plt.title('Training and validation accuracies for frequency domain')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


