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

EPSILON_DECAY = 0.996
EPSILON = 0.99
DISCOUNT = 0.99

LEARNING_RATE = 0.001
BATCH_SIZE    = 512
COUNTER       = 512     #when to perform batch training
EPOCHS        = 4       # how often to iterate over samples
NUM_BATCHES   = 20      # when counter is reached, how many random batches are chosen from memory

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


ACTIONS = list(range(-32, 24))   # represents left and rotate, left, nothing, right, right and rotate; 

PLOT_COUNTER = 50      # after 100 epochs save the plot 
MOVING_AVG_WINDOW_SIZE = 100        # for plots, what is moving avg?


COUNTER_TETRIS_EXPERT = 2