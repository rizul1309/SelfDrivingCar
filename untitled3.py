# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:53:28 2020

@author: sanjeev
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
import cv2
np.random.seed(0)

with open('C:/Users/sanjeev/german-traffic-signs/train.p','rb') as f:
    train_data = pickle.load(f)
with open('C:/Users/sanjeev/german-traffic-signs/valid.p','rb') as f:
   val_data = pickle.load(f)
with open('C:/Users/sanjeev/german-traffic-signs/test.p','rb') as f:
    test_data = pickle.load(f)

print(type(train_data))

x_train,y_train = train_data['features'],train_data['labels']
x_val,y_val = val_data['features'],val_data['labels']
x_test,y_test = test_data['features'],test_data['labels']

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

assert(x_train.shape[0] == y_train.shape[0]), "The no of images are not eaual"
data = pd.read_csv('C:/Users/sanjeev/german-traffic-signs/signnames.csv')
print(data)


num_of_samples = []
 
cols = 5
num_classes = 43
 
fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 50))
fig.tight_layout()
for i in range(cols):
    for j,row in data.iterrows():
        x_selected = x_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["SignName"])
            num_of_samples.append(len(x_selected))
            

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")


plt.imshow(x_train[1000])
plt.axis("off")
print(x_train[1000].shape)


def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

img = grayscale(x_train[1000])









