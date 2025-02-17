import os
import numpy as np

import warnings
import numpy as np

# Suppress specific NumPy warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core")
# Or more specifically:
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

# problem with tf.keras on WSL ubuntu, have to choose gpu
# TODO: not used in all files? currently im setting this env in multiple files
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
LOG_DIR  = os.path.join(BASE_DIR, "logs")
RES_DIR  = os.path.join(BASE_DIR, "res")

TETRIS_COMMAND = os.path.join(SRC_DIR, "cpp", "tetris")
FIFO_STATES = "fifo_states"
FIFO_CONTROLS = "fifo_controls"


LOGGING = False

EPSILON_DECAY = 0.995
EPSILON = 1.0
DISCOUNT = 0.95
EPSILON_COUNTER_EPOCH = 50
MIN_EPSILON = 0.01

LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-4
WARMUP_STEPS  = 500      # for LR scheduling
MAX_STEPS     = 5000     # for lr scheduling
BATCH_SIZE    = 64
COUNTER       = 2000     #when to perform batch training
EPOCHS        = 2   # how often to iterate over samples
NUM_BATCHES   = 70  # when counter is reached, how many random batches are chosen from memory


ACTIONS = list(range(-20, 24))   # represents left and rotate, left, nothing, right, right and rotate; 

PLOT_COUNTER = 50      # after 100 epochs save the plot 
MOVING_AVG_WINDOW_SIZE = 50        # for plots, what is moving avg?


COUNTER_TETRIS_EXPERT = 2
NUMBER_OF_PIECES      = 4       # how many pieces, default is 7 different (I, O, L, J, ...) 
                                # must be the same as  AMOUNT_OF_PIECES in `tetris.hpp``

# how long to wait in receive_from_pipe.
COMMUNICATION_TIME_OUT = 45.0


# from main file: 
# default value should be (350/5000), how much to sleep between the communication with c++
SLEEPTIME = 0.000001    
# sleeptime between multiple epochs    
INTER_ROUND_SLEEP_TIME = 0.2
ITERATIONS = 100000   # temp
POSSIBLE_NUMBER_STEPS = 4
                                
LOAD_MODEL = False          # load model?


# files ideosyncratic to the neural network
# currently this is a CNN, maybe architecture is changed in the future
FC_HIDDEN_UNIT_SIZE = 150
BOARD_HEIGHT = 28
BOARD_WIDTH  = 10
SIMPLE_CNN = True       # want to use the simple cnn => True
                        # more sophisticated with pooling etc => False


#q agent stuff
# ---------------------------------
ONLY_TRAINING = False           # only training, no pretraining with expert
IMITATION_COLLECTOR = False

USE_LR_SCHEDULER =True

# memory objs
# max length for the memory objects
MEMORY_MAXLEN        = 170000
MEMORY_EXPERT_MAXLEN = 60000
# biases for sampling from memory   
USE_REWARD_BIAS  = False    # favor best reward-samples in memory
USE_RECENCY_BIAS = False    # favor recently collected samplses (partially unifromly)
REWARD_TEMPERATURE = 1.0    # if 0 - uniform, if 1 strong bias

# pretraining / imitation learning at the start of learning to nudge model in right direction
IMITATION_LEARNING_LR         = 0.002       # learning rate only used in pretraining
IMITATIO_LEARNING_BATCHES     = 130     # currently not used
IMITATION_LEARNING_BATCH_SIZE = 64
IMITATION_LEARNING_EPOCHS = 0

MODEL_NAME = "../models/model"  # where are models saved? (for e.g. checkpointing )
MEMORY_PATH = "../res/precollected-memory/memory.pkl"   # where to collect mem

# placeholder for the pretraining. currently not used, as it would require real examles. 
PLACEHOLDER_GAME_BOARD = np.array([
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0.],
    [1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1.]
])
