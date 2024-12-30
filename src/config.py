import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
LOG_DIR  = os.path.join(BASE_DIR, "logs")
RES_DIR  = os.path.join(BASE_DIR, "res")

TETRIS_COMMAND = os.path.join(SRC_DIR, "cpp", "tetris")
# FIFO_CONTROLS = os.path.join(SRC_DIR, "fifo_controls")
# FIFO_STATES    = os.path.join(SRC_DIR, "fifo_states")
FIFO_STATES = "fifo_states"
FIFO_CONTROLS = "fifo_controls"

#TODO: complete this
# what should be logged
LOGGING = True

EPSILON_DECAY = 0.996
EPSILON = 0.99
DISCOUNT = 0.95

LEARNING_RATE = 0.001
BATCH_SIZE    = 32
COUNTER       = 64     #when to perform batch training
EPOCHS        = 2       # how often to iterate over samples
NUM_BATCHES   = 5      # when counter is reached, how many random batches are chosen from memory

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
    [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0.]
])


ACTIONS = list(range(-32, 20))   # represents left and rotate, left, nothing, right, right and rotate; 

PLOT_COUNTER = 50      # after 100 epochs save the plot 

MOVING_AVG_WINDOW_SIZE = 100