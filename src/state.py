from typing import List, Tuple
import numpy as np

from simpleLogger import SimpleLogger
from config import *


logger = SimpleLogger()


class State:
    def __init__(
        self, 
        game_board: List [List [int]], 
        lines_cleared: int,
        piece_type:int,
        piece_count:int,
        middle_point: Tuple[int, int], 
    ):
        self.game_board = np.array(game_board, dtype=np.float32)
        self.game_board_copy = self._copy_game_board()
        #self.height     = self._calculate_height()
        self.height     = self._calculate_aggregate_height()
        self.holes      = self._calculate_holes()
        self.max_height = self._calculate_height()
        self.bumpiness  = self._calculate_bumpiness()
        self.lines_cleared = lines_cleared
        self.piece_type = piece_type
        self.piece_count= piece_count
        self.middle_point = middle_point
        self.immedeate_lines_cleared = None
        
        #advanced 
        self.column_heights = self._calculate_column_heights()
        self.wells = self._calculate_wells()
        self.row_transitions = self._calculate_row_transitions()
        self.column_transitions = self._calculate_column_transitions()
        self.landing_height = self._calculate_landing_height()
        
        self.max_height_diff = self.get_max_height_diff()
        self.height_diff = self.get_max_height_diff()
        
        
    # def get_piece_type(self):
    #     all_2s = []
    #     for row in range(len(self.game_board)):
    #         for col in range(len(self.game_board[0])):
    #             if self.game_board[row][col] == 2:
    #                 all_2s.append((row, col))
    
    
    def get_holes_column_wise(self):
        holes_per_column = np.zeros(len(self.game_board_copy[0]), dtype=np.float32)
        
        for col in range(len(self.game_board_copy[0])):
            block_found = False
            holes = 0
            for row in range(len(self.game_board_copy)):
                if self.game_board_copy[row][col] == 1:
                    block_found = True
                elif block_found and self.game_board_copy[row][col] == 0:
                    holes += 1
            holes_per_column[col] = holes
            
        return holes_per_column

    def get_heights_column_wise(self):
        heights = np.zeros(len(self.game_board_copy[0]), dtype=np.float32)
        
        for col in range(len(self.game_board_copy[0])):
            for row in range(len(self.game_board_copy)):
                if self.game_board_copy[row][col] == 1:
                    heights[col] = len(self.game_board_copy) - row
                    break
                    
        return heights
    
    def get_column_features(self): 
        column_holes = self.get_holes_column_wise()
        column_height = self.get_heights_column_wise()

        return np.stack([column_holes, column_height])   # (2, board_width)


    def is_state_game_over(self)-> bool:
        for cell in self.game_board_copy[2]:
            if cell == 1:
                return True
        
        return False
        
        
    def get_height_variance(self):
        heights = np.array(self.column_heights)
        return np.var(heights)  # High variance = uneven stacking

    def get_max_height_diff(self):
        heights = self.column_heights
        return max(heights) - min(heights) if heights else 0
        
        
    def _copy_game_board(self):
        game_board_copy = []
        for row in self.game_board:
            #remove new flying piece, is not relevant for status
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
            # Initialize height as 0 for empty columns
            height = 0
            for row in range(len(self.game_board_copy)):
                if self.game_board_copy[row][col] == 1:
                    height = len(self.game_board_copy) - row
                    break
            heights.append(height)
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
        board_array =np.array(self.game_board_copy, dtype=np.float32)
        
        # one-hot representation for current piece type
        piece_type_array = np.zeros(NUMBER_OF_PIECES, dtype=np.float32)
        piece_type_array[self.piece_type] = np.float32(1.0)
        
        
        return board_array, piece_type_array
        
        board_reshaped = board_array.reshape(1, 28, 10)
        #logger.log(f"board_reshaped: {board_reshaped}")
        return board_reshaped
        #flattened_array = np.array(self.game_board).flatten()  # wrong, we dont want to have the board, way too many possible configurations
        #return np.concatenate((flattened_array, [self.lines_cleared, self.height, self.holes, self.bumpiness]))
        # return np.array([
        #     self.lines_cleared,         # TODO: does this even make sense?
        #     self.height, 
        #     self.holes, 
        #     self.bumpiness, 
        #     self.piece_type,
        #     #self.wells,
        #     #self.row_transitions,
        #     #self.column_transitions,
        #     #self.landing_height
        # ])
    
    @staticmethod
    def normalize_action(action):
        # TODO: make this more sofisticated 
        return action +20
    
    def denormalize_action(action): 
        return action - 20
    
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
    

