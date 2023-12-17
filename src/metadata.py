import numpy as np
import pickle

COUNTER_THRESH = 20

class Metadata:
    def __init__(self, logger, fifo_states_name, fifo_controls_name, fd_states:int, fd_controls:int):
        self.logger = logger
        self.fifo_states_name   = fifo_states_name
        self.fifo_controls_name = fifo_controls_name
        self.fd_states = fd_states
        self.fd_controls = fd_controls
        

    def debug(self):
        return "fd_states: " + str(self.fd_states) + " fd_controls: " + str(self.fd_controls)
    


class State:
    def __init__(self, lines_cleared, height, holes, bumpiness, piece_type):
        self.lines_cleared = lines_cleared
        self.height        = height
        self.holes         = holes
        self.bumpiness     = bumpiness
        self.piece_type    = piece_type


    def __repr__(self):
        message: str = f"""
    lines cleared: {self.lines_cleared}
    holes: {self.holes}
    height: {self.height}
    bumpiness {self.bumpiness}
    piece type {self.piece_type}
                        """
        return message
    

    def convert_to_array(self):
        return np.array([self.lines_cleared, self.height, self.holes, self.bumpiness, self.piece_type])
    
    
    def get_values(self):
        lines_cleared = self.lines_cleared
        height = self.height
        holes = self.holes
        bumpiness = self.bumpiness
        piece_type = self.piece_type

        return lines_cleared, height, holes, bumpiness, piece_type
    


class Game:
    def __init__(self):
        self.lines_cleared = 0
        self.epoch         = 0
        self.counter       = 0
        self.epsilon       = 0

    def increase_epoch(self):
        self.epoch += 1

    def set_lines_cleared(self, lines_cleared):
        self.lines_cleared = lines_cleared

    def increase_counter(self):
        """
        returns true, if counter reaches threshold, defined in `COUNTER_THRESH`
        """
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def load_model(self):
        file_path = "../res/saved_game.pkl"
        with open(file_path, 'rb') as f:
            obj:Game = pickle.load(f)
        
        self.set_lines_cleared(obj.lines_cleared)
        self.set_epoch(obj.epoch)
        

    def save_model(self):
        file_path = "../res/saved_game.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def __repr__(self):
        string:str = f"""
    ----------------------------------------------------------------------

    ----------------------------------------------------------------------
    current epoch        ={self.epoch}
    current lines cleared={self.lines_cleared}
    current epsilon      ={self.epsilon}
    ----------------------------------------------------------------------

    ----------------------------------------------------------------------

    """
        return string