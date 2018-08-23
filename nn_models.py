#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, merge, UpSampling2D, Conv2DTranspose, concatenate
from keras.models import Model

def unet(input_size):

	input_img = Input(input_size)
	
	conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
	
	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

	conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
	
	#model.summary()

	return Model(input_img, conv10)