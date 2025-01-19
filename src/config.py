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

#TODO: complete this
# what should be logged
LOGGING = False

EPSILON_DECAY = 0.997
EPSILON = 1.0
DISCOUNT = 0.95
EPSILON_COUNTER_EPOCH = 50
MIN_EPSILON = 0.01

LEARNING_RATE = 0.0004
BATCH_SIZE    = 64
COUNTER       = 20      #when to perform batch training
EPOCHS        = 1       # how often to iterate over samples
NUM_BATCHES   = 10       # when counter is reached, how many random batches are chosen from memory

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


ACTIONS = list(range(-20, 24))   # represents left and rotate, left, nothing, right, right and rotate; 

PLOT_COUNTER = 50      # after 100 epochs save the plot 
MOVING_AVG_WINDOW_SIZE = 50        # for plots, what is moving avg?


COUNTER_TETRIS_EXPERT = 2
NUMBER_OF_PIECES      = 7       # how many pieces, default is 7 different (I, O, L, J, ...) 
                                # must be the same as  AMOUNT_OF_PIECES in `tetris.hpp``

# how long to wait in receive_from_pipe.
COMMUNICATION_TIME_OUT = 15.0


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
FC_HIDDEN_UNIT_SIZE = 128
BOARD_HEIGHT = 28
BOARD_WIDTH  = 10