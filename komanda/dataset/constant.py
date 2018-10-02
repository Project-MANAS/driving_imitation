# RNNs are typically trained using (truncated) backprop through time. SEQ_LEN here is the length of BPTT.
# Batch size specifies the number of sequence fragments used in a sigle optimization step.
# (Actually we can use variable SEQ_LEN and BATCH_SIZE, they are set to constants only for simplicity).
# LEFT_CONTEXT is the number of extra frames from the past that we append to the left of our input sequence.
# We need to do it because 3D convolution with "VALID" padding "eats" frames from the left, decreasing the seq length.
# One should be careful here to maintain the model's causality.

SEQ_LEN = 10
LEFT_CONTEXT = 5
BATCH_SIZE = 1
BUFFER_SIZE = 10
N_THREADS = 8

HEIGHT = 480
WIDTH = 640
CHANNELS = 3

RNN_SIZE = 32
RNN_LAYERS = 1

NUM_EPOCHS = 100

CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-6:-3]  # angle,torque,speed
OUTPUT_DIM = len(OUTPUTS)  # predict all features: steering angle, torque and vehicle speed

CHECKPOINT_DIR = '/tmp'
DATASET_DIR = '/home/lastjedi/Workspace/dataset/'
print('CHECKPOINT_DIR :', CHECKPOINT_DIR)
print('DATASET_DIR :', DATASET_DIR)

validation_fraction = 0.01
test_fraction = 0.01
