#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout,\
							Conv2DTranspose, concatenate,\
							RepeatVector, Reshape, BatchNormalization
#from tensorflow.keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
import numpy as np
from keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score


def bce(y_true, y_pred):
	return binary_crossentropy(y_true, y_pred)

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
	bs_pipeline = [(0.05, 4, 5),
					   (0.05/3, 2, 3),
					   (0.05/3/3, 2, 3),
					   (0.05/3/3/3, 2, 3)]
	
	for params in bs_pipeline:
#		print(params, best_bs-params[0]*params[1], best_bs+params[0]*params[2], params[0])
		for bs in np.arange(best_bs-params[0]*params[1], best_bs+params[0]*params[2], params[0]):
#			score = accuracy_score(np.where(pred>bs, 1, 0), y)
			score = iou_precision(y, np.where(pred>bs, 1, 0))
#			print(bs, score)
			if score > best_score:
				best_score = score
				best_bs = bs
#		print('-'*60)
#		print(best_bs, best_score)
#		print('-'*60)
		
	return best_bs, best_score


def unet(input_size, include_depth=[], dropout=True, batchnorm=True):

	input_img = Input(input_size, name='input_image')
	input_depth = Input((1,), name='input_depth')

	if 0 in include_depth:
		print(' - Introducing depth 0')
		depth = RepeatVector(input_size[0]*input_size[1])(input_depth)
		depth = Reshape((input_size[0], input_size[1], 1))(depth)
		input_img = concatenate([input_img, depth], 0)
		
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
	

	if 1 in include_depth:
		print(' - Introducing depth 1')
		depth1 = RepeatVector(64)(input_depth)
		depth1 = Reshape((8,8, 1))(depth1)
		pool4 = concatenate([pool4, depth1], -1)
	

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

	conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='output')(conv9)
	
	#model.summary()

	if len(include_depth) > 0:
		print(' - Model with depth')
		return Model(inputs=[input_img, input_depth], outputs=conv10)
	else:
		return Model(input_img, conv10)



import keras.backend as K
import tensorflow as tf

# https://www.kaggle.com/dibgerge/another-tf-keras-implementation-of-score-metric
def iou_precision(y_true, y_pred):
	"""
	Computes the mean precision at different iou threshold levels.
	:param y_true:
	:param y_pred:
	:return:
	"""
	y_true = tf.to_int32(y_true)
	y_pred = tf.to_int32(tf.round(y_pred))

	n_batch = tf.shape(y_true)[0]

	y_true = tf.reshape(y_true, shape=[n_batch , -1])
	y_pred = tf.reshape(y_pred, shape=[n_batch, -1])

	intersection = K.sum(tf.bitwise.bitwise_and(y_true, y_pred), -1)
	union = K.sum(tf.bitwise.bitwise_or(y_true, y_pred), -1)
	#iou = tf.where(union == 0, tf.ones(n_batch), tf.to_float(intersection/union))
	SMOOTH = tf.constant(1e-6)
	iou = tf.add(tf.to_float(intersection), SMOOTH)/tf.add(tf.to_float(union), SMOOTH)

	precision = tf.zeros(n_batch)
	thresholds = np.arange(0.5, 1.0, 0.05)
	for thresh in thresholds:
		precision = precision + tf.to_float(iou > thresh)
	precision = precision/len(thresholds)

	mean_precision = K.mean(precision)
	
	with tf.Session() as sess:
		iou = sess.run(mean_precision)
		
	return iou

















	