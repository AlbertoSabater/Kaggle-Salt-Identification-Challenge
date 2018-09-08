#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, merge, UpSampling2D, Conv2DTranspose, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
import numpy as np
from keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def tune_base_score(pred, y):
	best_bs = 0.5
	best_score = 0
	bs_pipeline = [(0.05, 8, 9),
					   (0.05/3, 2, 3),
					   (0.05/3/3, 2, 3),
					   (0.05/3/3/3, 2, 3)]
	
	for params in bs_pipeline:
		print(params, best_bs-params[0]*params[1], best_bs+params[0]*params[2], params[0])
		for bs in np.arange(best_bs-params[0]*params[1], best_bs+params[0]*params[2], params[0]):
			score = accuracy_score(np.where(pred>bs, 1, 0), y)
			print(bs, score)
			if score > best_score:
				best_score = score
				best_bs = bs
		print('-'*60)
		print(best_bs, best_score)
		print('-'*60)
		
	return best_bs, best_score


def unet(input_size, dropout=True, batchnorm=True):

	input_img = Input(input_size)
	
	conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	if batchnorm: pool1 = BatchNormalization()(pool1)
	if dropout: pool1 = Dropout(0.25)(pool1)

	conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	if batchnorm: pool2 = BatchNormalization()(pool2)
	if dropout: pool2 = Dropout(0.25)(pool2)

	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	if batchnorm: pool3 = BatchNormalization()(pool3)
	if dropout: pool3 = Dropout(0.25)(pool3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	if batchnorm: pool4 = BatchNormalization()(pool4)
	if dropout: pool4 = Dropout(0.25)(pool4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
	
	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	if batchnorm: up6 = BatchNormalization()(up6)
	if dropout: up6 = Dropout(0.25)(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
	if batchnorm: up7 = BatchNormalization()(up7)
	if dropout: up7 = Dropout(0.25)(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
	if batchnorm: up8 = BatchNormalization()(up8)
	if dropout: up8 = Dropout(0.25)(up8)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(32, (	3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	if batchnorm: up9 = BatchNormalization()(up9)
	if dropout: up9 = Dropout(0.25)(up9)
	conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)
	if batchnorm: conv9 = BatchNormalization()(conv9)

	conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
	
	#model.summary()

	return Model(input_img, conv10)