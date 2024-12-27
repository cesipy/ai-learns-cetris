import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
LOG_DIR  = os.path.join(BASE_DIR, "logs")
RES_DIR  = os.path.join(BASE_DIR, "res")

TETRIS_COMMAND = os.path.join(SRC_DIR, "cpp", "tetris")
FIFO_CONTROLS = os.path.join(SRC_DIR, "fifo_controls")
FIFO_STATES    = os.path.join(SRC_DIR, "fifo_states")


#TODO: complete this
# what should be logged
LOGGING = False
