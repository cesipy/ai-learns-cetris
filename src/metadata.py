import numpy as np
import pickle
from typing import List
import config

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
    def __init__(
        self, 
        game_board: List [List [int]], 
        lines_cleared: int
    ):
        self.game_board = np.array(game_board, dtype=np.float32)
        self.game_board_copy = self._copy_game_board()
        self.height     = self._calculate_height()
        self.holes      = self._calculate_holes()
        self.bumpiness  = self._calculate_bumpiness()
        self.lines_cleared = lines_cleared
        
    def _copy_game_board(self):
        game_board_copy = []
        for row in self.game_board:
            new_row = [0 if cell == 2 else cell for cell in row]
            game_board_copy.append(new_row)
        return np.array(game_board_copy, dtype=np.float32)
            
    def _calculate_height(self): 
        for i, row in enumerate(self.game_board_copy):
            if sum(row) != 0: 
                return len(self.game_board_copy) - i
        return 0

    def _calculate_holes(self):
        holes = 0
        for i, row in enumerate(self.game_board_copy):
            if sum(row) > len(self.game_board_copy[0])  * 0.75: # 75% are filled
                full_row = len(self.game_board_copy[0])
                holes += full_row - sum(row)
            
        return holes
    
    def _calculate_bumpiness(self):
        bumpiness = 0
        
        highest_in_column = []
        
        # get highest point in each column
        for i in range(len(self.game_board_copy[0])):
             
             for j in range(len(self.game_board_copy)):
                if self.game_board_copy[j][i] == 1:
                    highest_in_column.append(j)
                    break
            
        for i in range(len(highest_in_column) - 1):
            delta = abs(highest_in_column[i] - highest_in_column[i + 1])
            bumpiness += delta
        
        return bumpiness
            


    def __repr__(self):
        message: str = f"""
    game_board:    {self.game_board}  
    lines cleared: {self.lines_cleared}
    holes:         {self.holes}
    height:        {self.height}
    bumpiness      {self.bumpiness}
                        """
        return message
    

    def convert_to_array(self):
        return self.game_board.flatten()
    
    
    def get_values(self):
        lines_cleared = self.lines_cleared
        height = self.height
        holes = self.holes
        bumpiness = self.bumpiness

        return lines_cleared, height, holes, bumpiness
    


class Game:
    def __init__(self):
        self.lines_cleared = 0
        self.epoch         = 0
        self.counter       = 0
        self.epsilon       = 0
        self.lines_cleared_current_epoch = 0

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
        self.lines_cleared += self.lines_cleared_current_epoch
        self.lines_cleared_current_epoch = 0


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
    current lines cleared={self.lines_cleared}
    current epsilon      ={self.epsilon}
    -------------------------------------------------------------------------

    -------------------------------------------------------------------------

    """
        return string