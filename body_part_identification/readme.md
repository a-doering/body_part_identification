# Bodypart Identification
This is written and intended to be run on the server of my institute. The paths, and requirements were therefore specific to the setup of the server at the time of my thesis.
## Usage
Connect to server, replace server_name with your servers name.
```shell
$ ssh server_name #no -X unless the connection to ssh will not be terminated
```
On server, move into body_part_identification directory.
```shell
$ export PYTHONPATH="$PWD" # once before training
```
Trains the  network, saves the output in the file_out and file_err instead of printing it to command. The training results will be saved in save_dir (specified in /training/train.py).
```shell
$ nohup python3 -u training/train.py > file_out 2> file_err &
```

## Components
### medio
get_paths.py gets all the paths to labels and images in all given directories.
read_image.py reads out image or labels from .mat files.

### training
train.py main program, calls everything.
data_generator.py generates data for the keras model.fit_generator, serves data during training.
prepare_data.py prepares data for the data generator.
model_hybrid.py model for combined classification and regression.
model_classification.py model for pure classification.
custom_metrics.py contains several custom metric functions.
saver.py saves results of training.

predictor.py uses the trained weights to predict the position of the landmarks, visualizes them.
eval_hyperparameter_tuning.py evaluates the results of hyperparameter optimization, visualizes them.

### util
convert_label.py converts the labels to appropriate class or regression labels.
patches.py can extract patches, create list of all possible patches.