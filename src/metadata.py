import numpy as np
from typing import List


COUNTER_THRESH = 20

from simpleLogger import SimpleLogger

logger = SimpleLogger()

class Metadata:
    def __init__(self, logger, fifo_states_name, fifo_controls_name, fd_states:int, fd_controls:int):
        self.logger = logger
        self.fifo_states_name   = fifo_states_name
        self.fifo_controls_name = fifo_controls_name
        self.fd_states = fd_states
        self.fd_controls = fd_controls
        

    def debug(self):
        return "fd_states: " + str(self.fd_states) + " fd_controls: " + str(self.fd_controls)
    



