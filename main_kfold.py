#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import time
#from tqdm import tqdm

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
#from keras.models import load_model, model_from_json
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import segmentation_models
from segmentation_models.utils import set_trainable


# TODO: support a depth


TENSORBOARD_DIR = './logs/'

model_params = {
			'datetime': str(datetime.datetime.now()),
			'stratify': True,
			'n_models': 5,
			'model_type': 'rrunet', 		# resnet34 / rrunet
			'freeze_encoder': True,
			'retrain_resnet': True,
			'target_size': (101,101),
#			'target_size': (256,256),
			'nn_size_base': 16,
			'include_depth': [],
			'dropout': True,
			'batchnorm': True,
			'data_gen_args': {
					'horizontal_flip': True,
					'vertical_flip': False,
					'rotation_range': 15,
					'width_shift_range': 0.1,
					'height_shift_range': 0.1,
					'zoom_range': [0.9, 1.2]
				},
			'validation_split': 0.0,
			'batch_size': 32,
			'epochs': 250,
			'es_patience': 10,
			'rlr_patience': 5,																																																																												
			'loss': 'bcedice', 			# bce / -dice- / bcedice / -bceiou-
			'optimizer': 'adam',
			'metrics': ["accuracy", nn_models.iou_tf],
			'monitor': 'val_loss', 'monitor_mode': 'min',
			'model_architecture_file': 'model_architecture',
			'base_score': 0.5,
			'tta': True
		}

if not os.path.exists(TENSORBOARD_DIR): os.makedirs(TENSORBOARD_DIR)

depths = pd.DataFrame.from_csv('./data/depths.csv')
depths.z = preprocessing.MinMaxScaler().fit_transform(depths[['z']].values.astype(float))


# %%
	
# =============================================================================
# Create train data generator and validation set
# =============================================================================

# Load images
full_train_images, full_train_image_names = nn_utils.load_folder_images("./data/train/images/", model_params['target_size'])
full_train_masks, _ = nn_utils.load_folder_images("./data/train/masks/", model_params['target_size'])
coverage_class = nn_utils.get_coverage_class(full_train_masks)


# %%

models = []

model_params['model_folder'] = nn_utils.create_new_model_folder()
if not os.path.exists(model_params['model_folder']): os.makedirs(model_params['model_folder'])
model_params['model_weights_file']= 'model_weights'

print(model_params['model_folder'])


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=model_params['n_models'], shuffle=True)

for train_index, test_index in skf.split(full_train_images, coverage_class):
	
	print(' *  FOLD: {}/{}'.format(len(models)+1, model_params['n_models']))	

# =============================================================================
# Create fold training-test dataset
# =============================================================================
	
	train_images, test_images = full_train_images[train_index], full_train_images[test_index]
	train_masks, test_masks = full_train_masks[train_index], full_train_masks[test_index]
	
	train_generator = nn_utils.get_image_generator_on_memory_v2(
			train_images, train_masks,
			model_params['batch_size'], model_params['data_gen_args'])
	
	val_data = full_train_images[test_index]
	val_masks = full_train_masks[test_index]
	
	model_params['num_samples_train'] = len(train_index)
	
	print(train_images.shape, test_images.shape)
	print(val_data.shape, val_masks.shape)


# =============================================================================
# Create and train Model
# =============================================================================
	
	print(' *  Building', model_params['model_type'])

	if model_params['model_type'] == 'unet':
		model = nn_models.unet(input_size = model_params['target_size'] + (1,),
							   size_base = model_params['nn_size_base'],
							   include_depth = model_params['include_depth'],
							   dropout=model_params['dropout'],
							   batchnorm=model_params['batchnorm'],)
	elif model_params['model_type'] == 'rrunet':
		model = nn_models.residual_unet(input_size = model_params['target_size'] + (1,),
								size_base = model_params['nn_size_base'],
								include_depth = model_params['include_depth'])
	elif model_params['model_type'] == 'resnet34':
		model = segmentation_models.Unet(backbone_name='resnet34', 
								   encoder_weights='imagenet', freeze_encoder=model_params['freeze_encoder'])
#		img_size_target = 128	
#		model = UResNet34(input_shape=(1,)+model_params['target_size'], freeze_encoder=True)
	elif model_params['model_type'] == 'resnet18':
		model = segmentation_models.Unet(backbone_name='resnet18', 
								   encoder_weights='imagenet', freeze_encoder=model_params['freeze_encoder'])
	else:
		raise ValueError('Model type Error')	
	
	
	print('Trainable layers: {}/{}'.format(sum([l.trainable for l in model.layers]), len(model.layers)))

	
	nn_loss = 'binary_crossentropy'
	if model_params['loss'] == 'bce':
		nn_loss = nn_models.bce
	elif model_params['loss'] == 'dice':
		nn_loss = nn_models.dice_loss
	elif model_params['loss'] == 'bcedice':
		nn_loss = nn_models.bce_dice_loss
		
	model.compile(loss=nn_loss, optimizer=model_params['optimizer'], metrics=model_params['metrics'])
	
	model_checkpoint_file = model_params['model_folder'] + str(len(models)) + '_' + model_params['model_weights_file'] + '.h5'
	callbacks = [
				ModelCheckpoint(model_checkpoint_file, 
					   monitor=model_params['monitor'], verbose=1, save_best_only=True, mode=model_params['monitor_mode']),
					   
				EarlyStopping(monitor=model_params['monitor'], min_delta=0.00001, 
					 verbose=1, mode=model_params['monitor_mode'], patience=model_params['es_patience']),
				
				TensorBoard(log_dir='{}{}'.format(TENSORBOARD_DIR, model_params['model_folder'].split('/')[-2]), 
							  histogram_freq=0, write_graph=True, 
							  write_grads=1, batch_size=model_params['batch_size'], write_images=True),
				   
				CSVLogger(model_params['model_folder']+'log.csv', separator=',', append=True),
				
				ReduceLROnPlateau(monitor=model_params['monitor'], mode=model_params['monitor_mode'], 
								  factor=0.5, patience=model_params['rlr_patience'], 
								  min_lr=0.00001, verbose=1)
			]
	
	print(' *  Model {}/{} ready'.format(len(models)+1, model_params['n_models']))



	hist = model.fit_generator(
				generator = train_generator,
				steps_per_epoch = 1.4*(model_params['num_samples_train'] // model_params['batch_size']), #################### 3072 num_samples_train
				epochs = model_params['epochs'],
				validation_data = (val_data, val_masks),
	#			validation_steps = num_samples_val // batch_size,
				shuffle = True,
				callbacks = callbacks,
				use_multiprocessing = False if platform.system() == 'Windows' else True,
				verbose = 1)
	


	if model_params['retrain_resnet'] and model_params['freeze_encoder'] and \
			model_params['model_type'] in ['resnet34', 'resnet18']:
		print(' * Unfreezing model')
		set_trainable(model)
		print('Trainable layers: {}/{}'.format(sum([l.trainable for l in model.layers]), len(model.layers)))
		hist = model.fit_generator(
					generator = train_generator,
					steps_per_epoch = 1.4*(model_params['num_samples_train'] // model_params['batch_size']), #################### 3072 num_samples_train
					epochs = model_params['epochs'],
					validation_data = (val_data, val_masks),
		#			validation_steps = num_samples_val // batch_size,
					shuffle = True,
					callbacks = callbacks,
					use_multiprocessing = False if platform.system() == 'Windows' else True,
					verbose = 1)
		
	
	print(' *  Model {}/{} trained'.format(len(models)+1, model_params['n_models']))

	# Storing best model
	model.load_weights(model_checkpoint_file)
	models.append(model)	
	
	val_loss = min(hist.history['val_loss'])
	val_iou = min(hist.history['val_iou_tf'])
	os.rename(model_checkpoint_file, 
		   model_checkpoint_file.replace('.h5', '_{:.4f}_{:.4f}.h5'.format(val_loss, val_iou)))
	
	
# %%

# =============================================================================
# Store model_architecture and params
# =============================================================================

# Store model architecture
model_architecture = model.to_json()
with open(model_params['model_folder'] + model_params['model_architecture_file'] + '.json', 'w') as f:
	json.dump(model_architecture, f)

del model_params['metrics']
# Store model params
with open(model_params['model_folder'] + 'model_params.json', 'w') as f:
	json.dump(model_params, f)


# Rename model_folder with monitor and its value, and tensorboard folder
val = min(hist.history[model_params['monitor']]) if 'loss' in model_params['monitor'] else max(hist.history[model_params['monitor']])
new_model_folder = model_params['model_folder'][:-1] + \
				"_{}_{:.4f}".format(model_params['monitor'], val)
new_model_folder = new_model_folder + '_{}_nm{}_{}_l{}_sb{}_{}_d{}_{}_{}/'.format(
						model_params['model_type'],
						model_params['n_models'],
						'stry' if model_params['stratify'] and model_params['validation_split']>0.0 else 'nstry',
						model_params['loss'],
						model_params['nn_size_base'],
						'd' if model_params['dropout'] else 'nd',
						'bn' if model_params['batchnorm'] else 'nbn',
						''.join([ str(v) for v in model_params['include_depth'] ]) if len(model_params['include_depth'])>0 else 'nd',
						'tta' if model_params['tta'] else 'notta'
		)

os.rename(model_params['model_folder'], new_model_folder)
os.rename(model_params['model_folder'].replace('./models/', './logs/'), new_model_folder.replace('./models/', './logs/'))
model_params['model_folder'] = new_model_folder


# %%

# =============================================================================
# Tune base_score and store full predictions
# =============================================================================

full_preds = nn_models.predict_models(models, full_train_images, tta=model_params['tta'])
full_preds = nn_utils.process_image(full_preds, (len(full_preds),)+(101,101))

print('Tuning base score', datetime.datetime.now().strftime('%H:%M:%S'))
t = time.time()
best_bs, best_score = nn_models.tune_base_score(full_preds, full_train_masks.astype(int))
model_params['base_score'] = best_bs
print('Base score tuned. {}. {:.2f}m. threshold: {:.4f} | score: {:.4f}'.format(datetime.datetime.now().strftime('%H:%M:%S'), 
	  (time.time()-t)/60, best_bs, best_score))

# Store full predictions
pickle.dump((full_preds, full_train_image_names), open(model_params['model_folder']+'preds_train.pckl', 'wb'))


# %%

# =============================================================================
# Predict data
# =============================================================================

test_dir = './data/test/'
test_data, test_image_names = nn_utils.load_folder_images(test_dir, model_params['target_size'])
test_data = test_data.reshape((len(test_data),)+model_params['target_size']+(1,))
if len(model_params['include_depth']) > 0:
	test_data = [test_data, depths.loc[[ n[:-4] for n in test_image_names ]].values]

	
print(' * Calculating and storing test predictions')

#test_preds = model.predict(test_data)
test_preds = nn_models.predict_models(models, test_data, tta=model_params['tta'])
test_preds = [ nn_utils.get_result(tp, model_params['base_score']) for tp in test_preds ]

pickle.dump((test_preds, test_image_names), open(model_params['model_folder']+'preds_test.pckl', 'wb'))

csv_df = pd.DataFrame.from_dict({'id': [ n.split('.')[0] for n in test_image_names ],
									'rle_mask': test_preds })

csv_path = model_params['model_folder']+model_params['model_folder'].split('/')[-2]+'.csv.gz'
csv_df.to_csv(csv_path, compression='gzip', index=False)


print(' * Submitting predictions')

val_loss = max(hist.history['val_loss'])
comment = 'l{}_vl{:.4f}_vs{}'.format(model_params['loss'], val_loss, model_params['validation_split'])
command = 'kaggle competitions submit -c tgs-salt-identification-challenge -f {} -m "{}"'.format(csv_path.replace('|', '\|'), comment)
print(command)
os.system(command)
print(' * Predictions submitted')






