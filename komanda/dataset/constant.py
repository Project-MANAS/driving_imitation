import os
import subprocess

# define some constants

# RNNs are typically trained using (truncated) backprop through time. SEQ_LEN here is the length of BPTT.
# Batch size specifies the number of sequence fragments used in a sigle optimization step.
# (Actually we can use variable SEQ_LEN and BATCH_SIZE, they are set to constants only for simplicity).
# LEFT_CONTEXT is the number of extra frames from the past that we append to the left of our input sequence.
# We need to do it because 3D convolution with "VALID" padding "eats" frames from the left, decreasing the sequence length.
# One should be careful here to maintain the model's causality.

SEQ_LEN = 10
LEFT_CONTEXT = 5
BATCH_SIZE = 4
BUFFER_SIZE = 10
N_CAMS = 3
N_THREADS = 8

N_AUG = 10

# These are the input image parameters.
HEIGHT = 480
WIDTH = 640
CHANNELS = 3  # RGB

# The parameters of the LSTM that keeps the model state.
RNN_SIZE = 32
RNN_PROJ = 32

DATASET_MAP_PARALLEL = 10
NUM_EPOCHS = 100
# Our training data follows the "interpolated.csv" format from Ross Wightman's scripts.
CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-6:-3]  # angle,torque,speed
OUTPUT_DIM = len(OUTPUTS)  # predict all features: steering angle, torque and vehicle speed

if os.environ.get('CHEKPOINTS', False) and os.environ.get('DATASETS', False):
	CHECKPOINT_DIR = os.environ['CHECKPOINTS'] + "/udacity_steering/challenge_2/v4"
	DATASET_DIR = os.environ['DATASETS'] + "/udacity_steering/challenge_2"
else:
	print('Datasets and Checkpoints directory not in os.environ. Temporarily hard coding path')
	username = subprocess.check_output(['bash', '-c', 'whoami'])
	username = str(username.strip())[2:-1]
	CHECKPOINT_DIR = '/media/' + username + '/7B3A33754CA4427C/checkpoints/udacity_steering/challenge_2/v4'
	DATASET_DIR = '/media/' + username + '/7B3A33754CA4427C/datasets/udacity_steering/challenge_2'
	print('CHECKPOINT_DIR :' + CHECKPOINT_DIR)
	print('DATASET_DIR :' + DATASET_DIR)
validation_fraction = 0.001
test_fraction = 0.001
