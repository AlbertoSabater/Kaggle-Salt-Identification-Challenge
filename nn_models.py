#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout,\
							Conv2DTranspose, concatenate,\
							RepeatVector, Reshape, BatchNormalization,\
							Activation, Add
#from tensorflow.keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
import tensorflow as tf
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


###############################################################################
###############################################################################
	

def unet(input_size, size_base=16, include_depth=[], dropout=True, batchnorm=True):

	input_img = Input(input_size, name='input_image')
	input_depth = Input((1,), name='input_depth')

	if 0 in include_depth:
		print(' - Introducing depth 0')
		depth = RepeatVector(input_size[0]*input_size[1])(input_depth)
		depth = Reshape((input_size[0], input_size[1], 1))(depth)
		input_img = concatenate([input_img, depth], 0)
		
	conv1 = Conv2D(size_base*1, (3, 3), activation='relu', padding='same')(input_img)
	conv1 = Conv2D(size_base*1, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	if batchnorm: pool1 = BatchNormalization()(pool1)
	if dropout: pool1 = Dropout(0.25)(pool1)

	conv2 = Conv2D(size_base*2, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(size_base*2, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	if batchnorm: pool2 = BatchNormalization()(pool2)
	if dropout: pool2 = Dropout(0.25)(pool2)

	conv3 = Conv2D(size_base*4, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(size_base*4, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	if batchnorm: pool3 = BatchNormalization()(pool3)
	if dropout: pool3 = Dropout(0.25)(pool3)

	conv4 = Conv2D(size_base*8, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(size_base*8, (3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	if batchnorm: pool4 = BatchNormalization()(pool4)
	if dropout: pool4 = Dropout(0.25)(pool4)
	

	if 1 in include_depth:
		print(' - Introducing depth 1')
		depth1 = RepeatVector(64)(input_depth)
		depth1 = Reshape((8,8, 1), name='depth_1')(depth1)
		pool4 = concatenate([pool4, depth1], -1)
	

	conv5 = Conv2D(size_base*16, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(size_base*16, (3, 3), activation='relu', padding='same')(conv5)
	
	
	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	if batchnorm: up6 = BatchNormalization()(up6)
	if dropout: up6 = Dropout(0.25)(up6)
	conv6 = Conv2D(size_base*8, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(size_base*8, (3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
	if batchnorm: up7 = BatchNormalization()(up7)
	if dropout: up7 = Dropout(0.25)(up7)
	conv7 = Conv2D(size_base*4, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(size_base*4, (3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
	if batchnorm: up8 = BatchNormalization()(up8)
	if dropout: up8 = Dropout(0.25)(up8)
	conv8 = Conv2D(size_base*2, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(size_base*2, (	3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	if batchnorm: up9 = BatchNormalization()(up9)
	if dropout: up9 = Dropout(0.25)(up9)
	conv9 = Conv2D(size_base*2, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(size_base*2, (3, 3), activation='relu', padding='same')(conv9)
	if batchnorm: conv9 = BatchNormalization()(conv9)

	conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='output')(conv9)
	
	#model.summary()

	if len(include_depth) > 0:
		print(' - Model with depth')
		return Model(inputs=[input_img, input_depth], outputs=conv10)
	else:
		return Model(input_img, conv10)


###############################################################################
###############################################################################


def BatchActivate(x):
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
	x = Conv2D(filters, size, strides=strides, padding=padding)(x)
	if activation == True:
		x = BatchActivate(x)
	return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
	x = BatchActivate(blockInput)
	x = convolution_block(x, num_filters, (3,3) )
	x = convolution_block(x, num_filters, (3,3), activation=False)
	x = Add()([x, blockInput])
	if batch_activate:
		x = BatchActivate(x)
	return x


#https://www.kaggle.com/deepaksinghrawat/introduction-to-u-net-with-simple-resnet-blocks
def residual_unet(input_size, size_base=16, include_depth=[], DropoutRatio = 0.5):

	input_img = Input(input_size, name='input_image')
	input_depth = Input((1,), name='input_depth')
	

	# 101 -> 50
	conv1 = Conv2D(size_base * 1, (3, 3), activation=None, padding="same")(input_img)
	
	conv1 = residual_block(conv1,size_base * 1)
	conv1 = residual_block(conv1,size_base * 1, True)
	pool1 = MaxPooling2D((2, 2))(conv1)
	pool1 = Dropout(DropoutRatio/2)(pool1)
	
	if 0 in include_depth:
		print(' - Introducing depth 0', input_size)
		depth0 = RepeatVector(pow(50, 2))(input_depth)
		depth0 = Reshape((50,50,1), name='depth_0')(depth0)
		pool1 = concatenate([pool1, depth0], -1)

	# 50 -> 25
	conv2 = Conv2D(size_base * 2, (3, 3), activation=None, padding="same")(pool1)
	conv2 = residual_block(conv2,size_base * 2)
	conv2 = residual_block(conv2,size_base * 2, True)
	pool2 = MaxPooling2D((2, 2))(conv2)
	pool2 = Dropout(DropoutRatio)(pool2)

	# 25 -> 12
	conv3 = Conv2D(size_base * 4, (3, 3), activation=None, padding="same")(pool2)
	conv3 = residual_block(conv3,size_base * 4)
	conv3 = residual_block(conv3,size_base * 4, True)
	pool3 = MaxPooling2D((2, 2))(conv3)
	pool3 = Dropout(DropoutRatio)(pool3)

	# 12 -> 6
	conv4 = Conv2D(size_base * 8, (3, 3), activation=None, padding="same")(pool3)
	conv4 = residual_block(conv4,size_base * 8)
	conv4 = residual_block(conv4,size_base * 8, True)
	pool4 = MaxPooling2D((2, 2))(conv4)
	pool4 = Dropout(DropoutRatio)(pool4)


	if 1 in include_depth:
		print(' - Introducing depth 1')
		depth1 = RepeatVector(pow(6, 2))(input_depth)
		depth1 = Reshape((6, 6, 1), name='depth_1')(depth1)
		pool4 = concatenate([pool4, depth1], -1)


	# Middle
	convm = Conv2D(size_base * 16, (3, 3), activation=None, padding="same")(pool4)
	convm = residual_block(convm,size_base * 16)
	convm = residual_block(convm,size_base * 16, True)
	
	# 6 -> 12
	deconv4 = Conv2DTranspose(size_base * 8, (3, 3), strides=(2, 2), padding="same")(convm)
	uconv4 = concatenate([deconv4, conv4])
	uconv4 = Dropout(DropoutRatio)(uconv4)
	
	uconv4 = Conv2D(size_base * 8, (3, 3), activation=None, padding="same")(uconv4)
	uconv4 = residual_block(uconv4,size_base * 8)
	uconv4 = residual_block(uconv4,size_base * 8, True)
	
	# 12 -> 25
	#deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
	deconv3 = Conv2DTranspose(size_base * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
	uconv3 = concatenate([deconv3, conv3])	
	uconv3 = Dropout(DropoutRatio)(uconv3)
	
	uconv3 = Conv2D(size_base * 4, (3, 3), activation=None, padding="same")(uconv3)
	uconv3 = residual_block(uconv3,size_base * 4)
	uconv3 = residual_block(uconv3,size_base * 4, True)

	# 25 -> 50
	deconv2 = Conv2DTranspose(size_base * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
	uconv2 = concatenate([deconv2, conv2])
		
	uconv2 = Dropout(DropoutRatio)(uconv2)
	uconv2 = Conv2D(size_base * 2, (3, 3), activation=None, padding="same")(uconv2)
	uconv2 = residual_block(uconv2,size_base * 2)
	uconv2 = residual_block(uconv2,size_base * 2, True)
	
	# 50 -> 101
	#deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
	deconv1 = Conv2DTranspose(size_base * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
	uconv1 = concatenate([deconv1, conv1])
	
	uconv1 = Dropout(DropoutRatio)(uconv1)
	uconv1 = Conv2D(size_base * 1, (3, 3), activation=None, padding="same")(uconv1)
	uconv1 = residual_block(uconv1,size_base * 1)
	uconv1 = residual_block(uconv1,size_base * 1, True)
	
	#uconv1 = Dropout(DropoutRatio/2)(uconv1)
	#output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
	output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
	output_layer = Activation('sigmoid')(output_layer_noActi)
	
	
	if len(include_depth) > 0:
		print(' - Model with depth')
		return Model(inputs=[input_img, input_depth], outputs=output_layer)
	else:
		print(' - Model without depth')
		return Model(input_img, output_layer)
	

###############################################################################
###############################################################################


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

















	