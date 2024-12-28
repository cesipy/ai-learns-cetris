import numpy as np
import pickle
from typing import List
import config
import time

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
    


class State:
    def __init__(
        self, 
        game_board: List [List [int]], 
        lines_cleared: int,
        piece_type:int,
    ):
        self.game_board = np.array(game_board, dtype=np.float32)
        self.game_board_copy = self._copy_game_board()
        #self.height     = self._calculate_height()
        self.height     = self._calculate_aggregate_height()
        self.holes      = self._calculate_holes()
        self.bumpiness  = self._calculate_bumpiness()
        self.lines_cleared = lines_cleared
        self.piece_type = piece_type
        
        #advanced 
        self.column_heights = self._calculate_column_heights()
        self.wells = self._calculate_wells()
        self.row_transitions = self._calculate_row_transitions()
        self.column_transitions = self._calculate_column_transitions()
        self.landing_height = self._calculate_landing_height()
        
        
    # def get_piece_type(self):
    #     all_2s = []
    #     for row in range(len(self.game_board)):
    #         for col in range(len(self.game_board[0])):
    #             if self.game_board[row][col] == 2:
    #                 all_2s.append((row, col))

        
        
    def _copy_game_board(self):
        game_board_copy = []
        for row in self.game_board:
            new_row = [0 if cell == 2 else cell for cell in row]
            game_board_copy.append(new_row)
        return np.array(game_board_copy, dtype=np.float32)
    
    def _calculate_aggregate_height(self):
        return sum(self._calculate_column_heights())

    def _calculate_height(self):
        for i, row in enumerate(self.game_board_copy):
            if sum(row) != 0:
                return len(self.game_board_copy) - i
        return 0

    def _calculate_holes(self):
        holes = 0
        for col in range(len(self.game_board_copy[0])):
            block_found = False
            for row in range(len(self.game_board_copy)):
                if self.game_board_copy[row][col] == 1:
                    block_found = True
                elif block_found and self.game_board_copy[row][col] == 0:
                    holes += 1
        return holes

    def _calculate_bumpiness(self):
        bumpiness = 0
        highest_in_column = []
        # get highest point in each column
        for i in range(len(self.game_board_copy[0])):
            for j in range(len(self.game_board_copy)):
                if self.game_board_copy[j][i] == 1:
                    highest_in_column.append(len(self.game_board_copy) - j)
                    break
            else:
                highest_in_column.append(0)
        
        for i in range(len(highest_in_column) - 1):
            delta = abs(highest_in_column[i] - highest_in_column[i + 1])
            bumpiness += delta
        return bumpiness
    
    
    # advanced features
    def _calculate_column_heights(self):
        heights = []
        for col in range(len(self.game_board_copy[0])):
            for row in range(len(self.game_board_copy)):
                if self.game_board_copy[row][col] == 1:
                    heights.append(len(self.game_board_copy) - row)
                    break
            else:
                heights.append(0)
        return heights

    def _calculate_wells(self):
        wells = 0
        heights = self.column_heights
        for i in range(len(heights)):
            if i == 0:
                if heights[i] < heights[i+1] - 1:
                    wells += heights[i+1] - heights[i] - 1
            elif i == len(heights) - 1:
                if heights[i] < heights[i-1] - 1:
                    wells += heights[i-1] - heights[i] - 1
            else:
                min_neighbor = min(heights[i-1], heights[i+1])
                if heights[i] < min_neighbor - 1:
                    wells += min_neighbor - heights[i] - 1
        return wells

    def _calculate_row_transitions(self):
        transitions = 0
        for row in self.game_board_copy:
            for i in range(len(row)-1):
                if row[i] != row[i+1]:
                    transitions += 1
        return transitions

    def _calculate_column_transitions(self):
        transitions = 0
        for col in range(len(self.game_board_copy[0])):
            for row in range(len(self.game_board_copy)-1):
                if self.game_board_copy[row][col] != self.game_board_copy[row+1][col]:
                    transitions += 1
        return transitions

    def _calculate_landing_height(self):
        # Height where the last piece landed
        for row in range(len(self.game_board)):
            for col in range(len(self.game_board[0])):
                if self.game_board[row][col] == 2:  # falling piece
                    return len(self.game_board) - row
        return 0
            

    def __repr__(self):
        message: str = f"""
    game_board:    {self.game_board}  
    lines cleared: {self.lines_cleared}
    holes:         {self.holes}
    height:        {self.height}
    bumpiness      {self.bumpiness}
    wells          {self.wells}
    column_heights {self.column_heights}
    row_transitions {self.row_transitions}
    column_transitions {self.column_transitions}
    landing_height {self.landing_height}
                        """
        return message
    

    def convert_to_array(self):
        #flattened_array = np.array(self.game_board).flatten()  # wrong, we dont want to have the board, way too many possible configurations
        #return np.concatenate((flattened_array, [self.lines_cleared, self.height, self.holes, self.bumpiness]))
        return np.array([
            self.lines_cleared,         # TODO: does this even make sense?
            self.height, 
            self.holes, 
            self.bumpiness, 
            # self.wells,
            # self.row_transitions,
            # self.column_transitions,
            # self.landing_height
            
        ])
    
    
    def get_values(self):
        lines_cleared = self.lines_cleared
        height = self.height
        holes = self.holes
        bumpiness = self.bumpiness

        return lines_cleared, height, holes, bumpiness
    
    def set_values(
        self, 
        lines_cleared: int, 
        height: int, 
        holes: int, 
        bumpiness: int, 
        wells: int, 
        row_transitions: int, 
        column_transitions: int, 
        landing_height: int, 
    ): 
        self.lines_cleared = lines_cleared
        self.height = height
        self.holes = holes
        self.bumpiness = bumpiness
        self.wells = wells
        self.row_transitions = row_transitions
        self.column_transitions = column_transitions
        self.landing_height = landing_height
    


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