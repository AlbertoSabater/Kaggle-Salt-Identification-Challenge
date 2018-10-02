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
import platform
import pickle 
import re


BASE_MODEL_DIR = './models/'
BASE_PREPROCESSED_DATA_DIR = './data_preprocessed/'
if not os.path.exists(BASE_MODEL_DIR): os.makedirs(BASE_MODEL_DIR)
if not os.path.exists(BASE_PREPROCESSED_DATA_DIR): os.makedirs(BASE_PREPROCESSED_DATA_DIR)

N_JOBS = 1 if platform.system() == 'Windows' else 8


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
	pickle_file = '{}{}_{}_{}.pckl'.format(BASE_PREPROCESSED_DATA_DIR, 
				folder_path.split('/')[2], folder_path.split('/')[3], target_size)
	if os.path.isfile(pickle_file):
		print(' * Loading images from pickle: {}'.format(pickle_file))
		data, image_names = pickle.load(open(pickle_file, 'rb'))
		return data, image_names
	else: 	
		print(' * Loading and storing images: {}'.format(pickle_file))
		image_names = os.listdir(folder_path)
		images = Parallel(n_jobs=N_JOBS)(delayed(load_and_process_image)(f, folder_path, target_size) for f in tqdm(image_names, total=len(image_names)))
		data = np.array(images)
		pickle.dump((data, image_names), open(pickle_file, 'wb'))
		return data, image_names


def get_image_generator_on_memory(images, masks, batch_size, data_gen_args):
	
	seed = 123

	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)
	
	generator = zip(image_datagen.flow(images, batch_size=batch_size, shuffle=True, seed=seed), 
						   mask_datagen.flow(masks, batch_size=batch_size, shuffle=True, seed=seed))
	
	return generator, images.shape[0]


def get_image_depth_generator_on_memory(images, masks, depths, batch_size, data_gen_args):
	
	seed = 123

	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)
#	depth_datagen = ImageDataGenerator({})
	
	generator = zip(image_datagen.flow(images, batch_size=batch_size, shuffle=True, seed=seed),
#					   depth_datagen.flow(depths, batch_size=batch_size, shuffle=True, seed=seed), 
					   mask_datagen.flow(masks, batch_size=batch_size, shuffle=True, seed=seed))
	
	return generator, images.shape[0]


def get_image_depth_generator_on_memory_v2(images, masks, depths, batch_size, data_gen_args):
	seed = 123
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)	
	
	image_f = image_datagen.flow(images, depths, batch_size=batch_size, shuffle=True, seed=seed)
	mask_f = mask_datagen.flow(masks, batch_size=batch_size, shuffle=True, seed=seed)
	
	while True:
		image_n = image_f.next()
		mask_n = mask_f.next()
		
#		yield {'input_image': image_n[0], 'input_depth': image_n[1]}, {'output': mask_n}
#		yield [image_n[0], image_n[1]], mask_n
		yield np.concatenate([image_n[0], image_n[1]], axis=0), mask_n


# deprecated
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


## RLE encoding, as suggested by Tadeusz Hupało
def rle_encoding(x, base_score):
	x = np.where(x > base_score, 1, 0)
	dots = np.where(x.T.flatten() == 1)[0]
	run_lengths = []
	prev = -2
	for b in dots:
		if (b>prev+1): run_lengths.extend((b + 1, 0))
		run_lengths[-1] += 1
		prev = b
	return re.sub(r'[\[\],]','', str(run_lengths))
	

def get_result(pred, base_score):
	pred = resize(pred, (101, 101), mode='constant', preserve_range=True, anti_aliasing=True)
#	pred = rle(pred, base_score)
	pred = rle_encoding(pred, base_score)
	return pred

def get_prediction_result(model, images, target_size, base_score):
	return Parallel(n_jobs=N_JOBS)(delayed(get_result)(model.predict(img.reshape((1,)+target_size+(1,)))[0,:,:,0], base_score) 
						for img in tqdm(images, total=len(images)))
