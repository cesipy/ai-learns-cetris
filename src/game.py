import time
import numpy as np

import config

import pickle

class Game:
    def __init__(self):
        self.total_lines_cleared = 0
        self.epoch         = 0
        self.counter       = 0
        self.epsilon       = 0
        self.lines_cleared_current_epoch = 0
        self.start_time    = None
        self.end_game      = None
        self.lines_cleared_array = []
        self.mean_rewards        = []           # store mean rewards per epoch
        self.current_rewards     = []           # store rewards for current epoch 
        self.current_piece_count = 0
        
    def start_time_measurement(self):
        self.start_time = time.time()
        
    def end_time_measurement(self) -> float:
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        
        self.start_time = 0
        self.end_time   = 0
        
        return elapsed_time

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


    def set_lines_cleared_current_epoch(self, lines_cleared_current_epoch):
        self.lines_cleared_current_epoch = lines_cleared_current_epoch


    def update_after_epoch(self):
        # updates for plots
        self.lines_cleared_array.append(self.lines_cleared_current_epoch)
        self.mean_rewards.append(np.mean(self.current_rewards))

        self.total_lines_cleared += self.lines_cleared_current_epoch
        self.lines_cleared_current_epoch = 0
        self.current_rewards = []
        self.current_piece_count = 0


    def load_model(self):
        file_path = config.RES_DIR + "/saved_game.pkl"
        with open(file_path, 'rb') as f:
            obj:Game = pickle.load(f)
        
        self.set_lines_cleared(obj.lines_cleared)
        self.set_epoch(obj.epoch)
        

    def save_model(self):
        file_path = config.RES_DIR + "/saved_game.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def __repr__(self):
        string:str = f"""
    -------------------------------------------------------------------------

    -------------------------------------------------------------------------
    current epoch        ={self.epoch}
    current lines cleared={self.lines_cleared_current_epoch}
    current epsilon      ={self.epsilon}
    total lines cleared  ={self.total_lines_cleared}
    -------------------------------------------------------------------------

    -------------------------------------------------------------------------

    """
        return string
    
    def print_with_stats(
        self, 
        current_lines_cleared: int, 
        elapsed_time:float, 
        avg_reward: float,
        ) -> str: 
        string:str = f"""
    -------------------------------------------------------------------------

    -------------------------------------------------------------------------
    current epoch        ={self.epoch}
    current lines cleared={current_lines_cleared}
    current epsilon      ={self.epsilon}
    total lines cleared  ={self.total_lines_cleared}
    elapsed time         ={elapsed_time:.3f}
    current avg reward   ={avg_reward:.3f}
    -------------------------------------------------------------------------

    -------------------------------------------------------------------------

    """
        return string