# CHANGELOG

### 2018-08-25 | LB 1321/1497 | Score 0.552

RLE prediction fixed

Training without splitting train dataset into train and validation. Using train dataset without augmentation as validation dataset

Added Windows support and its environment yml. Parallel processing not working in Windows

Preprocessed data is load or stored on disk for a quicker execution

Train and test predicted dataset is stored on disk

Submitting RLE predictions to Kaggle. Using model_params as a comment


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
