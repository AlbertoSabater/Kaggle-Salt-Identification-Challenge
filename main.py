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
#from tqdm import tqdm

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from keras.models import load_model, model_from_json
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# TODO: update automatically with model_params and scores in description
# TODO: add depth after first and/or last CNN layer
# TODO: Parallel in tune base_score

TENSORBOARD_DIR = './logs/'

'''
Ubuntu updated to Cuda 9.0
tensorflow-gpu updated
multi-input model created. Depths as input concatenated in different layers
stored model name udpated
'''


model_params = {
			'datetime': str(datetime.datetime.now()),
			'target_size': (128,128),
			'include_depth': [1],
			'dropout': True,
			'batchnorm': True,
			'data_gen_args': {
					'horizontal_flip': True, 
					'vertical_flip': True,
				},
			'validation_split': 0.0,
			'batch_size': 64,
			'epochs': 5,
			'es_patience': 10,
			'loss': 'bcedice', 			# bce / dice / bcedice
			'optimizer': 'adam',
			'metrics': ["accuracy"],
			'monitor': 'val_loss',
			'model_architecture_file': 'model_architecture',
			'base_score': 0.5
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
full_train_masks, full_test_image_names = nn_utils.load_folder_images("./data/train/masks/", model_params['target_size'])

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
if len(model_params['include_depth']) > 0:
	
	print(' * Including depth')
	train_depths = depths.loc[[ n[:-4] for n in full_train_image_names ]].values
	val_depths = depths.loc[[ n[:-4] for n in full_test_image_names ]].values
#	depths_train = np.concatenate( [ np.resize(d, (1, 128,128,1)) for d in depths_train ])
#	train_generator, num_samples_train = nn_utils.get_image_depth_generator_on_memory(
	train_generator = nn_utils.get_image_depth_generator_on_memory_v2(
			train_images, train_masks, train_depths,
			model_params['batch_size'], model_params['data_gen_args'])
	model_params['num_samples_train'] = train_images.shape[0]
	
	train_data = [train_images, train_depths]
	val_data = [val_images, val_depths]
	
else:
	print(' * No including depth')
	train_generator, num_samples_train = nn_utils.get_image_generator_on_memory(
			train_images, train_masks,
			model_params['batch_size'], model_params['data_gen_args'])
	model_params['num_samples_train'] = num_samples_train
	train_data = train_images
	val_data = val_images


# %%
	
# =============================================================================
# Create and train Model
# =============================================================================

model_params['model_folder'] = nn_utils.create_new_model_folder()
if not os.path.exists(model_params['model_folder']): os.makedirs(model_params['model_folder'])
model_params['model_weights_file']= 'model_weights'

print(model_params['model_folder'])


model = nn_models.unet(model_params['target_size'] + (1,),
					   include_depth = model_params['include_depth'],
					   dropout=model_params['dropout'],
					   batchnorm=model_params['batchnorm'],)
print(model.summary())

print(' *  Data generator ready')


nn_loss = 'binary_crossentropy'
if model_params['loss'] == 'bce':
	nn_loss = nn_models.bce
elif model_params['loss'] == 'dice':
	nn_loss = nn_models.dice_loss
elif model_params['loss'] == 'bcedice':
	nn_loss = nn_models.bce_dice_loss
	
model.compile(loss=nn_loss, optimizer=model_params['optimizer'], metrics=model_params['metrics'])

callbacks = [
			ModelCheckpoint(model_params['model_folder'] + model_params['model_weights_file'] + '.h5', 
				   monitor=model_params['monitor'], verbose=1, save_best_only=True, mode='auto'),
				   
			EarlyStopping(monitor=model_params['monitor'], min_delta=0.00001, 
				 verbose=1, mode='auto', patience=model_params['es_patience']),
			
			TensorBoard(log_dir='{}{}'.format(TENSORBOARD_DIR, model_params['model_folder'].split('/')[-2]), 
						  histogram_freq=0, write_graph=True, 
						  write_grads=1, batch_size=model_params['batch_size'], write_images=True),
			   
			CSVLogger(model_params['model_folder']+'log.csv', separator=',', append=True)
		]

hist = model.fit_generator(
			generator = train_generator,
			steps_per_epoch = model_params['num_samples_train'] // model_params['batch_size'], #################### 3072 num_samples_train
			epochs = model_params['epochs'],
			validation_data = (val_data, val_masks),
#			validation_steps = num_samples_val // batch_size,
			shuffle = True,
			callbacks = callbacks,
			use_multiprocessing = False if platform.system() == 'Windows' else True,
			verbose = 1)


# %%



# %%


# Store model architecture
model_architecture = model.to_json()
with open(model_params['model_folder'] + model_params['model_architecture_file'] + '.json', 'w') as f:
	json.dump(model_architecture, f)
	

# Store model params
with open(model_params['model_folder'] + 'model_params.json', 'w') as f:
	json.dump(model_params, f)


# Rename model_folder with monitor and its value, and tensorboard folder
val = min(hist.history[model_params['monitor']]) if 'loss' in model_params['monitor'] else max(hist.history[model_params['monitor']])
new_model_folder = model_params['model_folder'][:-1] + \
				"_{}_{:.4f}".format(model_params['monitor'], val)
new_model_folder = new_model_folder + '_{}_{}_{}_d{}/'.format(
						model_params['loss'],
						'd' if model_params['dropout'] else 'nd',
						'bn' if model_params['batchnorm'] else 'nbn',
						''.join([ str(v) for v in model_params['include_depth'] ]) if len(model_params['include_depth'])>0 else 'nd'
		)
os.rename(model_params['model_folder'], new_model_folder)
os.rename(model_params['model_folder'].replace('./models/', './logs/'), new_model_folder.replace('./models/', './logs/'))
model_params['model_folder'] = new_model_folder

model.load_weights(model_params['model_folder'] + model_params['model_weights_file'] + '.h5')


# %%

train_preds = model.predict(train_data)
train_preds = nn_utils.process_image(train_preds, (len(train_preds),)+(101,101))


# %%

original_masks, _ = nn_utils.load_folder_images("./data/train/masks/", (101,101))


# %%

print('Tuning base score')
best_bs, best_score = nn_models.tune_base_score(train_preds.ravel(), original_masks.ravel().astype(int))
model_params['base_score'] = best_bs


# %%

#import tensorflow as tf
#
#mean_precision = nn_models.iou_precision(original_masks, train_preds)
##sess = tf.Session()
##iou = sess.run(mean_precision)
#
#with tf.Session() as sess:
#	iou = sess.run(mean_precision)

	
# %%

print(' * Calculating and storing train predictions')
#train_preds = nn_utils.get_prediction_result(model, val_data, model_params['target_size'], model_params['base_score'])
train_preds = model.predict(train_data)
train_preds = [ nn_utils.get_result(tp, model_params['base_score']) for tp in train_preds ]
pickle.dump((train_preds, full_train_image_names), open(model_params['model_folder']+'preds_train.pckl', 'wb'))


# %%

# =============================================================================
# Predict data
# =============================================================================

test_dir = './data/test/'
test_data, test_image_names = nn_utils.load_folder_images(test_dir, model_params['target_size'])
if len(model_params['include_depth']) > 0:
	test_data = [test_data, depths.loc[[ n[:-4] for n in test_image_names ]].values]

	
# %%	

print(' * Calculating and storing test predictions')
#test_preds = nn_utils.get_prediction_result(model, test_data, 
#							model_params['target_size'], model_params['base_score'])
test_preds = model.predict(test_data)
test_preds = [ nn_utils.get_result(tp, model_params['base_score']) for tp in test_preds ]

pickle.dump((test_preds, test_image_names), open(model_params['model_folder']+'preds_test.pckl', 'wb'))

csv_df = pd.DataFrame.from_dict({'id': [ n.split('.')[0] for n in test_image_names ],
									'rle_mask': test_preds })

csv_path = model_params['model_folder']+model_params['model_folder'].split('/')[-2]+'.csv.gz'
csv_df.to_csv(csv_path, compression='gzip', index=False)


print(' ** Submit prediction?')
input()
print(' * Submitting predictions')
#comment = '\n'.join([ '{}: {}'.format(k,v) for k,v in model_params.items() ])
#comment = str(model_params)

comment = 'l_{}_vs{}'.format(model_params['loss'], model_params['validation_split'])
command = 'kaggle competitions submit -c tgs-salt-identification-challenge -f {} -m "{}"'.format(csv_path.replace('|', '\|'), comment)
print(command)
#os.system(command)
#print(' * Predictions submitted')


# %%

model.load_weights('./models/1004_0915_model_24_val_loss_0.1887_bce_d_bn_d1/model_weights.h5')


# %%

# Plot train results
if False:
# %%
#	inds = np.random.permutation(range(len(val_images)))
	while True:
		ind = np.random.choice(list(range(len(val_images))))
		
		img = val_images[ind].reshape((1,)+model_params['target_size']+(1,))
		mask = val_masks[ind].reshape((1,)+model_params['target_size']+(1,))

		if len(model_params['include_depth']) > 0:
			depth = val_depths[ind]
			pred = model.predict([img, depth])[0,:,:,0]
		else:
			pred = model.predict(img)[0,:,:,0]

		
		fig = plt.figure()
		plt.subplot(221)
#		plt.imshow(i[0][0,:,:,0])
		plt.imshow(img[0,:,:,0])
		plt.subplot(222)
#		plt.imshow(i[1][0,:,:,0])
		plt.imshow(mask[0,:,:,0])
		plt.subplot(223)
		plt.imshow(pred)
		plt.subplot(224)
		plt.imshow(np.where(pred > model_params['base_score'], 1, 0))
		
		break
	
	# %%
	
# Plot test_results
if False:
	
	# %%
#	test_dir = 'data/test/'
#	test_image_names = os.listdir(test_dir)
#	img = nn_utils.load_and_process_image(np.random.choice(test_image_names, 1)[0], test_dir, model_params['target_size'])
	
#	pred = model.predict(img.reshape((1,)+model_params['target_size']+(1,)))[0,:,:,0]
	ind = np.random.choice(list(range(len(val_images))))
	img = test_data[0][ind].reshape((1,)+model_params['target_size']+(1,))
	
	if len(model_params['include_depth']) > 0:
		depth = test_data[1][ind]
		pred = model.predict([img, depth])[0,:,:,0]
	else:
		pred = model.predict(img.reshape((1,)+model_params['target_size']+(1,)))[0,:,:,0]
	
	fig = plt.figure()
	plt.subplot(131)
	plt.imshow(img[0,:,:,0])
	plt.subplot(132)
	plt.imshow(pred)
	plt.subplot(133)
	plt.imshow(np.where(pred > model_params['base_score'], 1, 0))
	

# %%
	
for r in train_generator:
	pass
	break
	
	