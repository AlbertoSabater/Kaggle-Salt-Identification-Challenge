#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator

import nn_models
import nn_utils
import matplotlib.pyplot as plt

#from PIL import Image
import numpy as np
import pandas as pd
import os
import json
import datetime
import platform
import pickle
#from tqdm import tqdm

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.model_selection import train_test_split


# TODO: update automatically with model_params and scores in description
# TODO: tune base_score

TENSORBOARD_DIR = './logs/'



model_params = {
			'datetime': str(datetime.datetime.now()),
			'target_size': (128,128),
			'data_gen_args': {
					'horizontal_flip': True, 
					'vertical_flip': True,
				},
			'validation_split': 0.0,
			'batch_size': 32,
			'epochs': 250,
			'loss': 'binary_crossentropy',
			'optimizer': 'adam',
			'metrics': ["accuracy"],
			'monitor': 'val_acc',
			'model_architecture_file': 'model_architecture',
			'base_score': 0.5
		}

if not os.path.exists(TENSORBOARD_DIR): os.makedirs(TENSORBOARD_DIR)


# %%
	
# =============================================================================
# Create train data generator and validation set
# =============================================================================

# Load images
full_train_images, full_train_image_names = nn_utils.load_folder_images("./data/train/images/", model_params['target_size'])
full_train_masks, _ = nn_utils.load_folder_images("./data/train/masks/", model_params['target_size'])

# Create train and validation dataset
if model_params['validation_split'] > 0:
	train_images, val_images, train_masks, val_masks= train_test_split(
			full_train_images, full_train_masks, test_size=model_params['validation_split'], random_state=123)
else:
	train_images = val_images = full_train_images.copy()
	train_masks = val_masks = full_train_masks.copy()

model_params['num_train'] = len(train_images)
model_params['num_val'] = len(val_images)

# Augmented train dataset
#train_generator, num_samples_train = get_train_generator(batch_size, target_size)
train_generator, num_samples_train = nn_utils.get_train_generator_on_memory(
		train_images, train_masks,
		model_params['batch_size'], model_params['data_gen_args'])
model_params['num_samples_train'] = num_samples_train


# %%
	
# =============================================================================
# Create and train Model
# =============================================================================

model_params['model_folder'] = nn_utils.create_new_model_folder()
if not os.path.exists(model_params['model_folder']): os.makedirs(model_params['model_folder'])
model_params['model_weights_file']= 'model_weights'

print(model_params['model_folder'])


model = nn_models.unet(model_params['target_size'] + (1,))
print(model.summary())

print(' *  Data generator ready')

model.compile(loss=model_params['loss'], optimizer=model_params['optimizer'], metrics=model_params['metrics'])

callbacks = [
			ModelCheckpoint(model_params['model_folder'] + model_params['model_weights_file'] + '.h5', 
				   monitor=model_params['monitor'], verbose=1, save_best_only=True, mode='auto'),
				   
			EarlyStopping(monitor=model_params['monitor'], min_delta=0.00001, verbose=1, mode='auto', patience=4),
			
			TensorBoard(log_dir='{}{}'.format(TENSORBOARD_DIR, model_params['model_folder'].split('/')[-2]), 
						  histogram_freq=0, write_graph=True, 
						  write_grads=1, batch_size=model_params['batch_size'], write_images=True),
			   
			CSVLogger(model_params['model_folder']+'log.csv', separator=',', append=True)
		]

hist = model.fit_generator(
			generator = train_generator,
			steps_per_epoch = num_samples_train // model_params['batch_size'], #################### 3072 num_samples_train
			epochs = model_params['epochs'],
			validation_data = (val_images, val_masks),
#			validation_steps = num_samples_val // batch_size,
			callbacks = callbacks,
			use_multiprocessing = False if platform.system() == 'Windows' else True,
			verbose = 1)


# Store model architecture
model_architecture = model.to_json()
with open(model_params['model_folder'] + model_params['model_architecture_file'] + '.json', 'w') as f:
	json.dump(model_architecture, f)
	

# Store model params
with open(model_params['model_folder'] + 'model_params.json', 'w') as f:
	json.dump(model_params, f)


# Rename model_folder with monitor and its value, and tensorboard folder
new_model_folder = model_params['model_folder'][:-1] + \
				"_{}_{:.4f}/".format(model_params['monitor'], max(hist.history[model_params['monitor']]))
os.rename(model_params['model_folder'], new_model_folder)
os.rename(model_params['model_folder'].replace('./models/', './logs/'), new_model_folder.replace('./models/', './logs/'))
model_params['model_folder'] = new_model_folder
	

# %%

print(' * Calculating and storing train predictions')
train_preds = nn_utils.get_prediction_result(model, full_train_images, model_params['target_size'], model_params['base_score'])
pickle.dump((train_preds, full_train_image_names), open(model_params['model_folder']+'preds_train.pckl', 'wb'))


# %%

# =============================================================================
# Predict data
# =============================================================================

test_dir = './data/test/'
test_images, test_image_names = nn_utils.load_folder_images(test_dir, model_params['target_size'])


# %%

print(' * Calculating and storing test predictions')
test_preds = nn_utils.get_prediction_result(model, test_images, model_params['target_size'], model_params['base_score'])
pickle.dump((test_preds, test_image_names), open(model_params['model_folder']+'preds_test.pckl', 'wb'))

csv_df = pd.DataFrame.from_dict({'id': [ n.split('.')[0] for n in test_image_names ],
									'rle_mask': test_preds })

csv_path = model_params['model_folder']+model_params['model_folder'].split('/')[-2]+'.csv'
csv_df.to_csv(csv_path, index=False)


print(' * Submitting predictions')
#comment = '\n'.join([ '{}: {}'.format(k,v) for k,v in model_params.items() ])
comment = str(model_params)
os.system('kaggle competitions submit -c tgs-salt-identification-challenge -f {} -m "{}"'.format(csv_path, comment))
print(' * Predictions submitted')


# %%


# Plot train results
if False:
	# %%
#	for i in train_generator:
	inds = np.random.permutation(range(len(val_images)))
	for i in zip(val_images[inds].reshape((1,)+val_images.shape), val_masks[inds].reshape((1,)+val_masks.shape)):
		pred = model.predict(i[0][0,:,:,:].reshape((1,)+model_params['target_size']+(1,)))[0,:,:,0]
		
		fig = plt.figure()
		plt.subplot(221)
		plt.imshow(i[0][0,:,:,0])
		plt.subplot(222)
		plt.imshow(i[1][0,:,:,0])
		plt.subplot(223)
		plt.imshow(pred)
		plt.subplot(224)
		plt.imshow(np.where(pred > model_params['base_score'], 1, 0))
		
		break
	
	# %%
	
# Plot test_results
if False:
	
	# %%
	test_dir = 'data/test/'
	test_image_names = os.listdir(test_dir)
	img = nn_utils.load_and_process_image(np.random.choice(test_image_names, 1)[0], test_dir, model_params['target_size'])
	pred = model.predict(img.reshape((1,)+model_params['target_size']+(1,)))[0,:,:,0]
	
	fig = plt.figure()
	plt.subplot(131)
	plt.imshow(img[:,:,0])
	plt.subplot(132)
	plt.imshow(pred)
	plt.subplot(133)
	plt.imshow(np.where(pred > model_params['base_score'], 1, 0))
	
	
	