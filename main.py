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
from keras.models import load_model, model_from_json
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import segmentation_models
from segmentation_models.utils import set_trainable
#from resnet_pretrained import UResNet34


# TODO: loss based on iou AND iou + bce
# TODO: modelos pretrained resnet34, resnet50
	# Freeze enconder at the beginning
	# https://www.kaggle.com/meaninglesslives/using-resnet50-pretrained-model-in-keras
	# https://github.com/killthekitten/kaggle-carvana-2017/blob/master/models.py
	# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65387
# TODO: modificar layers del decoder para que el output sea 101
# TODO: train durante ~40 epochs y después con early stopping on val_iou_tf

TENSORBOARD_DIR = './logs/'

'''
IOU metric added on fit
Predicciones con TTA
ReduceLROnPlateau
Added list of fold models. Predictions -> mean of single model predictions
Pretrained ResNet34 and ResNet18 u-net added with segmentation_models library
'''

# TEST
# TODO: bcedice d[] 	| 0.1949 - 0.779 | 0.7432 - 0.745
# TODO: bcedice d[1] 	| 0.2262 - 0.756
# TODO: bcedice d[0] 	| 0.2287 - 0.762
# TODO: bcedice d[0,1] 	| 0.2275 - 0.752
#
# TODO: bce d[] 		| 0.1480 - 0.695 | 0.7436 - 0.738
# TODO: bce d[1] 	 	| 0.1441 - 0.724
# TODO: bce d[0] 	 	| 0.1140 - 0.764
# TODO: bce d[0,1] 	 	|

# TODO: rrunet, 	batch_size=32, nn_size_base=32
# TODO: resnet34, 	batch_size=32 target_size (256,256)


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

if model_params['model_type'] in ['resnet34', 'resnet18']:
	full_train_images = np.array([ np.dstack([img]*3) for img in full_train_images ])


# %%

# Create train and validation dataset
if model_params['validation_split'] > 0:
	train_images, val_images, train_masks, val_masks= train_test_split(
			full_train_images, full_train_masks, test_size=model_params['validation_split'], random_state=123)
	if model_params['stratify'] and model_params['validation_split']>0.0:
		train_image_names, val_image_names = train_test_split(full_train_image_names, 
					test_size=model_params['validation_split'], stratify=coverage_class, random_state=123)
	else: 
		train_image_names, val_image_names = train_test_split(full_train_image_names, test_size=model_params['validation_split'], random_state=123)
else:
	train_images = val_images = full_train_images.copy()
	train_masks = val_masks = full_train_masks.copy()
	train_depths = val_depths = depths.copy()
	train_image_names = val_image_names = full_train_image_names.copy()


if model_params['model_type'] not in ['resnet34', 'resnet18']:
	train_images = train_images.reshape((len(train_images),)+model_params['target_size']+(1,))
	val_images = val_images.reshape((len(val_images),)+model_params['target_size']+(1,))
	
model_params['num_train'] = len(train_images)
model_params['num_val'] = len(val_images)

# Augmented train dataset
#train_generator, num_samples_train = get_train_generator(batch_size, target_size)
if len(model_params['include_depth']) > 0:
	
	print(' * Including depth')
	train_depths = depths.loc[[ n[:-4] for n in train_image_names ]].values
	val_depths = depths.loc[[ n[:-4] for n in val_image_names ]].values
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
	train_generator = nn_utils.get_image_generator_on_memory_v2(
			train_images, train_masks,
			model_params['batch_size'], model_params['data_gen_args'])
	model_params['num_samples_train'] = train_images.shape[0]
	train_data = train_images
	val_data = val_images

print(' *  Data generator ready')


# %%
	
# =============================================================================
# Create and train Model
# =============================================================================

model_params['model_folder'] = nn_utils.create_new_model_folder()
if not os.path.exists(model_params['model_folder']): os.makedirs(model_params['model_folder'])
model_params['model_weights_file']= 'model_weights'

print(model_params['model_folder'])


models = []
for i in range(model_params['n_models']):
	
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
			
		
#	print(model.summary())
	print('Layers unfrozen: {}/{}'.format(sum([l.trainable for l in model.layers]), len(model.layers)))
	
	
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
					   monitor=model_params['monitor'], verbose=1, save_best_only=True, mode=model_params['monitor_mode']),
					   
				EarlyStopping(monitor=model_params['monitor'], min_delta=0.00001, 
					 verbose=1, mode=model_params['monitor_mode'], patience=model_params['es_patience']),
				
				TensorBoard(log_dir='{}{}'.format(TENSORBOARD_DIR, model_params['model_folder'].split('/')[-2]), 
							  histogram_freq=0, write_graph=True, 
							  write_grads=1, batch_size=model_params['batch_size'], write_images=True),
				   
				CSVLogger(model_params['model_folder']+'log.csv', separator=',', append=True),
				
				ReduceLROnPlateau(monitor=model_params['monitor'], mode=model_params['monitor_mode'], 
								  factor=0.5, patience=model_params['rlr_patience'], min_lr=0.00001)
			]
	
	print(' *  Model {}/{} ready'.format(i+1, model_params['n_models']))



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
#		model = nn_models.set_model_trainable(model)
#		model.compile(loss=nn_loss, optimizer=model_params['optimizer'], metrics=model_params['metrics'])
		set_trainable(model)
		print('Layers unfrozen: {}/{}'.format(sum([l.trainable for l in model.layers]), len(model.layers)))
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
	
	print(' *  Model {}/{} trained'.format(i+1, model_params['n_models']))
	models.append(model)


# %%



# %%


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

model.load_weights(model_params['model_folder'] + model_params['model_weights_file'] + '.h5')


# %%

original_masks, _ = nn_utils.load_folder_images("./data/train/masks/", (101,101))
if model_params['validation_split'] > 0:
	train_original_masks, val_origial_masks = train_test_split(original_masks, test_size=model_params['validation_split'], random_state=123)
else:
	train_original_masks = val_origial_masks = original_masks
	
	
# %%

val_preds = nn_models.predict_models(models, val_data, tta=model_params['tta'])
#val_preds = model.predict(val_data)
val_preds = nn_utils.process_image(val_preds, (len(val_preds),)+(101,101))

# %%

print('Tuning base score', datetime.datetime.now().strftime('%H:%M:%S'))
t = time.time()
#best_bs, best_score = nn_models.tune_base_score(val_preds.ravel(), val_origial_masks.ravel().astype(int))
best_bs, best_score = nn_models.tune_base_score(val_preds, val_origial_masks.astype(int))
model_params['base_score'] = best_bs
print('Base score tuned. {}. {:.2f}m. {:.4f} | {}'.format(datetime.datetime.now().strftime('%H:%M:%S'), 
	  (time.time()-t)/60, best_bs, best_score))


# %%

print(' * Calculating and storing train predictions')
#train_preds = nn_utils.get_prediction_result(model, val_data, model_params['target_size'], model_params['base_score'])
#train_preds = model.predict(train_data)
train_preds = nn_models.predict_models(models, train_data, tta=model_params['tta'])
train_preds = [ nn_utils.get_result(tp, model_params['base_score']) for tp in train_preds ]
pickle.dump((train_preds, full_train_image_names), open(model_params['model_folder']+'preds_train.pckl', 'wb'))


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


print(' ** Submit prediction?')
#input()
print(' * Submitting predictions')
#comment = '\n'.join([ '{}: {}'.format(k,v) for k,v in model_params.items() ])
#comment = str(model_params)

val_loss = max(hist.history['val_loss'])
comment = 'l{}_vl{:.4f}_vs{}'.format(model_params['loss'], val_loss, model_params['validation_split'])
command = 'kaggle competitions submit -c tgs-salt-identification-challenge -f {} -m "{}"'.format(csv_path.replace('|', '\|'), comment)
print(command)
#os.system(command)
#print(' * Predictions submitted')





# %%

#model.load_weights('./models/1004_0915_model_24_val_loss_0.1887_bce_d_bn_d1/model_weights.h5')


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
		ax = plt.subplot(221)
#		plt.imshow(i[0][0,:,:,0])
		plt.imshow(img[0,:,:,0])
		ax.set_title("Orig")
		ax = plt.subplot(222)
#		plt.imshow(i[1][0,:,:,0])
		plt.imshow(mask[0,:,:,0])
		ax.set_title("Real mask")
		ax = plt.subplot(223)
		plt.imshow(pred)
		ax.set_title("Pred")
		ax = plt.subplot(224)
		plt.imshow(np.where(pred > model_params['base_score'], 1, 0))
		ax.set_title("Pred mask")
		
		pred_res = nn_utils.resize(pred, (101, 101), mode='constant', preserve_range=True, anti_aliasing=True)
		rle = nn_utils.rle_encoding(pred_res, model_params['base_score'])
		pred_fin = nn_utils.rle_to_mask(rle.split(), (101,101))
		pred_send = nn_utils.get_result(pred, model_params['base_score'])
		pred_send = nn_utils.rle_to_mask(pred_send.split(), (101,101))
		plt.figure()
		plt.subplot(131)
		plt.imshow(pred_res)
		plt.subplot(132)
		plt.imshow(pred_fin)
		plt.subplot(133)
		plt.imshow(pred_send)
		
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
	print(test_image_names[ind])
	
	
	# %%
	
# Plot test_results
if False:
	
	# %%
#	test_dir = 'data/test/'
#	test_image_names = os.listdir(test_dir)
#	img = nn_utils.load_and_process_image(np.random.choice(test_image_names, 1)[0], test_dir, model_params['target_size'])
	
#	pred = model.predict(img.reshape((1,)+model_params['target_size']+(1,)))[0,:,:,0]
	ind = np.random.choice(list(range(len(csv_df))))
	img_name = csv_df.iloc[ind].id
	img_rle = csv_df.iloc[ind].rle_mask
	img = nn_utils.rle_to_mask(img_rle.split(), (101,101))
	img_orig = test_data[0][test_image_names.index(img_name+'.png')]
	
#	img = test_data[0][ind].reshape((1,)+model_params['target_size']+(1,))
#	
#	if len(model_params['include_depth']) > 0:
#		depth = test_data[1][ind]
#		pred = model.predict([img, depth])[0,:,:,0]
#	else:
#		pred = model.predict(img.reshape((1,)+model_params['target_size']+(1,)))[0,:,:,0]
	
	fig = plt.figure()
	plt.subplot(131)
	plt.imshow(img_orig[:,:,0])
	plt.subplot(132)
	plt.imshow(img)
#	plt.subplot(133)
#	plt.imshow(np.where(pred > model_params['base_score'], 1, 0))
	print(img_name)
	

# %%
	
for r in train_generator:
	pass
	break
	
	