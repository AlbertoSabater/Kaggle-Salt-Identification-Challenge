#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
from joblib import Parallel, delayed 

from skimage.transform import resize
from keras.preprocessing.image import load_img

from tqdm import tqdm
import numpy as np
import os
import datetime
import sys

BASE_MODEL_DIR = './models/'


def create_new_model_folder():
	folder_num = len(os.listdir(BASE_MODEL_DIR))
	folder_path = BASE_MODEL_DIR + '{}_model_{}/'.format(datetime.datetime.today().strftime('%m%d_%H%M'), folder_num)
	os.mkdir(folder_path)
	return folder_path


# deprecated
def get_train_generator(batch_size, target_size):

	seed = 1
	
	data_gen_args = dict(rescale = 1./255)
	
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)
	
	image_generator = image_datagen.flow_from_directory(
		'./data/train/images',
		target_size=target_size,
		color_mode = 'grayscale',
		class_mode=None,
		batch_size = batch_size,
		seed=seed)
	
	mask_generator = mask_datagen.flow_from_directory(
		'./data/train/masks',
		target_size=target_size,
		color_mode = 'grayscale',
		class_mode=None,
		batch_size = batch_size,
		seed=seed)
	
	# combine generators into one which yields image and masks
	train_generator = zip(image_generator, mask_generator)
	
	return train_generator, image_datagen.n


def load_image(f, folder_path): return np.array(load_img(folder_path + f, color_mode='grayscale')) / 255
def process_image(img, target_size): return resize(img, target_size+(1,), mode='constant', preserve_range=True, anti_aliasing=True)
def load_and_process_image(f, folder_path, target_size):
	img = load_image(f, folder_path)
	return process_image(img, target_size)

def load_folder_images(folder_path, target_size):
	print(' * Loading images from {}'.format(folder_path))
	sys.stdout.flush()
	image_names = os.listdir(folder_path)
	images = Parallel(n_jobs=8)(delayed(load_and_process_image)(f, folder_path, target_size) for f in tqdm(image_names, total=len(image_names)))
	return np.array(images), image_names


def get_train_generator_on_memory(train_images, train_masks, batch_size, data_gen_args):
	
	seed = 123

#	data_gen_args = dict()
	
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)
	
	train_generator = zip(image_datagen.flow(train_images, batch_size=batch_size, shuffle=True, seed=seed), 
						   mask_datagen.flow(train_masks, batch_size=batch_size, shuffle=True, seed=seed))
	
	return train_generator, train_images.shape[0]


def rle(img, base_score):
	
	flat_img = img.flatten()
	flat_img = np.where(flat_img > base_score, 1, 0).astype(np.uint8)
	flat_img = np.insert(flat_img, [0, len(flat_img)], [0, 0])

	starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
	ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
	starts_ix = np.where(starts)[0] + 1
	ends_ix = np.where(ends)[0] + 1
	lengths = ends_ix - starts_ix
	
	return ' '.join([ str(s)+' '+str(l) for s, l in zip(starts_ix, lengths) ])
	

def get_result(pred, base_score):
	pred = resize(pred, (101, 101), mode='constant', preserve_range=True, anti_aliasing=True)
	pred = rle(pred, base_score)
	return pred

def get_prediction_result(model, images, target_size, base_score):
	return Parallel(n_jobs=8)(delayed(get_result)(model.predict(img.reshape((1,)+target_size+(1,)))[0,:,:,0], base_score) 
						for img in tqdm(images, total=len(images)))
