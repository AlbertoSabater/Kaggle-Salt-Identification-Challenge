# CHANGELOG

### 2018-10-18 | LB 1604/3243 | Score 0.793

Test Time Augmentation (TTA) added to model predictions  
Train an array of models. Final prediction obtained with the average of single predictions  
IoU metric added to epoch metrics  
Added ReduceLROnPlateau to training callbacks  
Added pretrained u-net models based on ResNet-18 and ResNet-34. Unable to train them due to the lack of GPU memory  


### 2018-10-10 | LB 1781/3187 | Score 0.779

Ubuntu updated to Cuda 9.0  
Depth added to U-net concatenating it to first and/or mid layer  
Model store name updated  
Prediction threshold tunned with IoU metric  
Added more augmentation parameters  
U-net layer sizes configurable  
NEW MODEL, U-net with residual blocks created + configurable layer sizes. Not need image resampling  
Stratified train-test split on salt coverage  


### 2018-09-08 | LB 1870/2152 | Score 0.622

Dice Loss added  
Dice + BCE Loss added  
Dropout and BatchNormalization added to the baseline model  
Prediction threshold tunned  
Loading best weights after training  
Predictions compressed before submission  


### 2018-08-25 | LB 1321/1497 | Score 0.552

RLE prediction fixed  
Training without splitting train dataset into train and validation. Using train dataset without augmentation as validation dataset  
Added Windows support and its environment yml  
Parallel processing not working in Windows  
Preprocessed data is load or stored on disk for a quicker execution  
Train and test predicted dataset is stored on disk  
Submitting RLE predictions to Kaggle  
Using model_params as a comment


### 2018-08-23 | LB 1344/1432 | Score 0.347

Basic learning workflow implementation
* Data Loading and preprocessing in parallel
* Simple Data Augmentation in memory or on the fly
* Split train and validation data
* Load and store NN and architecture
* Fit NN
* Metric: binary_crossentropy
* Callbacks: ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
	
Prediction + RLE + CSV submission file. Processing in parallel
