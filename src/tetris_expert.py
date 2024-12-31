from config import *
from reward import calculate_reward, calculate_reward_tetris_expert
from state import State
from simpleLogger import SimpleLogger

from typing import List, Tuple, Optional

import numpy as np
from config import *



logger = SimpleLogger()


class TetrisExpert:
    
    def __init__(self, actions):
        self.board_dimentsion = PLACEHOLDER_GAME_BOARD.shape
        self.actions     = actions     # is this the correct action space?
        
    def _get_action_mapping(self, action: int)-> Tuple[int, int]: 
        """returns relative position, rotation"""
        action_mapping = {
            # Right movements (negative indices)
            -40: (-10,0),  # -10right-rotate0
            -39: (-10,1),  # -10right-rotate1
            -38: (-10,2),  # -10right-rotate2
            -37: (-10,3),  # -10right-rotate3
            -36: (-9,0),  # -9right-rotate0
            -35: (-9,1),  # -9right-rotate1
            -34: (-9,2),  # -9right-rotate2
            -33: (-9,3),  # -9right-rotate3
            -32: (-8,0),  # -8right-rotate0
            -31: (-8,1),  # -8right-rotate1
            -30: (-8,2),  # -8right-rotate2
            -29: (-8,3),  # -8right-rotate3
            -28: (-7,0),  # -7right-rotate0
            -27: (-7,1),  # -7right-rotate1
            -26: (-7,2),  # -7right-rotate2
            -25: (-7,3),  # -7right-rotate3
            -24: (-6,0),  # -6right-rotate0
            -23: (-6,1),  # -6right-rotate1
            -22: (-6,2),  # -6right-rotate2
            -21: (-6,3),  # -6right-rotate3
            -20: (-5,0),  # -5right-rotate0
            -19: (-5,1),  # -5right-rotate1
            -18: (-5,2),  # -5right-rotate2
            -17: (-5,3),  # -5right-rotate3
            -16: (-4,0),  # -4right-rotate0
            -15: (-4,1),  # -4right-rotate1 
            -14: (-4,2),  # -4right-rotate2
            -13: (-4,3),  # -4right-rotate3
            -12: (-3,0),  # -3right-rotate0
            -11: (-3,1),  # -3right-rotate1
            -10: (-3,2),  # -3right-rotate2
            -9:  (-3,3),  # -3right-rotate3
            -8:  (-2,0),  # -2right-rotate0
            -7:  (-2,1),  # -2right-rotate1
            -6:  (-2,2),  # -2right-rotate2
            -5:  (-2,3),  # -2right-rotate3
            -4:  (-1,0),  # -1right-rotate0
            -3:  (-1,1),  # -1right-rotate1
            -2:  (-1,2),  # -1right-rotate2
            -1:  (-1,3),  # -1right-rotate3
            # Left movements (non-negative indices)
            0:   (0,0),   # 0left-rotate0
            1:   (0,1),   # 0left-rotate1
            2:   (0,2),   # 0left-rotate2
            3:   (0,3),   # 0left-rotate3
            4:   (1,0),   # 1left-rotate0
            5:   (1,1),   # 1left-rotate1
            6:   (1,2),   # 1left-rotate2
            7:   (1,3),   # 1left-rotate3
            8:   (2,0),   # 2left-rotate0
            9:   (2,1),   # 2left-rotate1
            10:  (2,2),   # 2left-rotate2
            11:  (2,3),   # 2left-rotate3
            12:  (3,0),   # 3left-rotate0
            13:  (3,1),   # 3left-rotate1
            14:  (3,2),   # 3left-rotate2
            15:  (3,3),   # 3left-rotate3
            16:  (4,0),   # 4left-rotate0
            17:  (4,1),   # 4left-rotate1
            18:  (4,2),   # 4left-rotate2
            19:  (4,3),   # 4left-rotate3
            20:  (5,0),   # 5left-rotate0
            21:  (5,1),   # 5left-rotate1
            22:  (5,2),   # 5left-rotate2
            23:  (5,3),   # 5left-rotate3
            24:  (6,0),   # 6left-rotate0
            25:  (6,1),   # 6left-rotate1
            26:  (6,2),   # 6left-rotate2
            27:  (6,3),   # 6left-rotate3
        }
        return action_mapping[action]
        
        
    def get_best_move(self, state: State):
        # if first 6 rows have any 1(= one block), stop, dont calculate actions.
        # currently takes too much
        current_board = state.game_board
        for i in range(len(current_board)): 
            for j in range(len(current_board[0])): 
                if i <= 5:  # TODO: mke this configurabel in config file
                    if current_board[i][j] == 1: 
                        if LOGGING:
                            logger.log("early quit")
                            return None     # TODO: adapt, what is no action?
                    
        rewards = []
        for action in self.actions:
            next_state = self._simulate_action(action=action, state=state)
            if not next_state:  # no good state found
                continue
            #currently not available
            
            reward = calculate_reward_tetris_expert(next_state=next_state)
            # if LOGGING:
            #     logger.log(f"action: {action}, reward: {reward}, bumpiness: {next_state.bumpiness}")
            rewards.append((action, reward))
            #logger.log(f"action: {action}, reward: {reward}, bumpiness: {next_state.bumpiness}")
        
        #logger.log(rewards)
        # consists of action:int, reward:float
        ret= self._get_max_reward_action(rewards=rewards)
        if ret is None: 
            return None
        
        max_action, max_reward = ret
        next_state = self._simulate_action(state=state, action=max_action)
        #logger.log(f"final simulation for max_action {max_action}:\n{next_state.game_board}")
        #logger.log(f"\n-----\nall rewards: {rewards}")
        #logger.log(f"after tetris export, this is the best action: action: {max_action}, reward: {max_reward}---\n\n")
        
        return max_action
            
    
    def _simulate_action(self, state: State, action: int) -> State: 
        """based on current state, the action is performed and the resulting state is returned"""
        current_board = np.copy(state.game_board)
        
        piece_positions = []
        for i in range(len(current_board)):
            for j in range(len(current_board[0])):
                if current_board[i][j]== 2:     # falling piece
                    piece_positions.append((i,j))
                    
        if not piece_positions:
            # no falling positions??
            return None

        horizontal_move, rotation = self._get_action_mapping(action=action)
        
        rotated_board = self._rotate_piece(state=state, rotation=rotation)
        if rotated_board is None:
            return None
        
        translated_board = self._horizontal_move_piece(rotated_board, move=horizontal_move)
        if translated_board is None:
            return None
        #TODO: there are still redundancies left, when +4 and cant rotate
        
        final_board = self._gravity(translated_board)
        if final_board is None:
            return None
        
        new_lines_cleared = self._get_new_lines_cleared(board=final_board)
        #if LOGGING:
            #logger.log(f"final board for action {action}: \n{final_board}")
            
        #print(f"final board for action {action}: \n{final_board}")
        
        new_state = State(
            game_board=final_board,
            lines_cleared=state.lines_cleared+new_lines_cleared,      #TODO: update
            piece_count=state.piece_count,
            piece_type=state.piece_type, 
            middle_point=state.middle_point
        )
        return new_state
    
    def _get_new_lines_cleared(self, board: np.ndarray) -> int:
        """returns the number of lines cleared in the new state"""
        lines_cleared = 0
        for i in range(len(board)):
            if np.all(board[i] == 1):
                lines_cleared += 1
        return lines_cleared
        
        
    def _horizontal_move_piece(self, board: np.ndarray, move: int) -> np.ndarray:
        """
        Move the piece horizontally by the specified amount
        
        Args:
            board (np.ndarray): Current game board
            move (int): Number of positions to move (positive = right, negative = left)
            
        Returns:
            np.ndarray: Board with moved piece or None if move is invalid
        """
        new_board = np.copy(board)
        
        # Find all current falling piece positions
        piece_positions = [(i, j) for i in range(len(board)) 
                          for j in range(len(board[0])) 
                          if board[i][j] == 2]
        
        if not piece_positions:
            return new_board
            
        for i, j in piece_positions:
            new_board[i][j] = 0
            
        # new positions after move
        new_positions = [(i, j + move) for i, j in piece_positions]
        
        for i, j in new_positions:
            # Check board boundaries
            if not (0 <= j < board.shape[1]):
                return None
            # Check collision with existing pieces
            if 0 <= i < board.shape[0] and board[i][j] == 1:
                return None
                
        for i, j in new_positions:
            if 0 <= i < board.shape[0] and 0 <= j < board.shape[1]:
                new_board[i][j] = 2
                
        return new_board
        
    
    
    def _rotate_piece(self, state: State, rotation: int) -> np.ndarray:
        """
        Rotate the piece around its middle point
        
        Args:
            state (State): Current game state
            rotation (int): Number of 90-degree rotations (0-3)
        
        Returns:
            np.ndarray: Board with rotated piece
        """
        board = np.copy(state.game_board)
        
        mid_row, mid_col = state.middle_point
        
        #current falling piece positions
        piece_positions = [(i, j) for i in range(len(board)) 
                        for j in range(len(board[i])) 
                        if board[i][j] == 2]
        
        if not piece_positions:
            return board
        
        for i, j in piece_positions:
            board[i][j] = 0
        
        # Calculate relative positions to the middle point
        relative_positions = [(i - mid_row, j - mid_col) for (i, j) in piece_positions]
        
        # Rotate relative positions
        def rotate_point(x, y):
            """Rotate a point 90 degrees clockwise around origin"""
            return y, -x 
            #return -y, x
        
        # Perform rotation based on the rotation count
        rotated_relative_positions = relative_positions
        for _ in range(rotation):
            rotated_relative_positions = [
                rotate_point(x, y) for (x, y) in rotated_relative_positions
            ]
        
        # Reposition rotated pieces back to the board
        for (dx, dy) in rotated_relative_positions:
            new_row = mid_row + dx
            new_col = mid_col + dy
            
            # Check board boundaries
            if 0 <= new_row < board.shape[0] and 0 <= new_col < board.shape[1]:
                board[new_row, new_col] = 2
        
        return board
                
    def _gravity(self, board: np.ndarray) -> np.ndarray:
        """Apply gravity to the falling piece until it lands"""
        current_board = np.copy(board)

        while True:
            piece_positions = []
            for i in range(len(current_board)):
                for j in range(len(current_board[0])):
                    if current_board[i][j] == 2:
                        piece_positions.append((i,j))
                        
            if not piece_positions:
                return current_board
                
            can_move = True
            for i, j in piece_positions:
                if i == len(current_board) - 1:  # Reached bottom
                    can_move = False
                    break
                if current_board[i + 1][j] == 1:  # Would hit placed piece
                    can_move = False
                    break
                    
            if not can_move:
                # convert falliung to normal cells
                for i, j in piece_positions:
                    current_board[i][j] = 1
                return current_board
                
            # Move all pieces down one step
            new_board = np.copy(current_board)
   
            for i, j in piece_positions:
                new_board[i][j] = 0

            for i, j in piece_positions:
                new_board[i + 1][j] = 2
                
            current_board = new_board
            #if LOGGING:
                #logger.log(f"gravity board: \n{current_board}")


            
        
        
        
        
        
                
        
    
    def _get_max_reward_action(self, rewards: List[Tuple[int, float]]) -> Tuple[int, float]:
        if not rewards:  
            return None
            
        current_max_reward = float("-inf")
        current_max_elem = rewards[0]  # Initialize with first element
        
        multiple_max_rewards = []
        
        for reward_tuple in rewards:
            if reward_tuple[1] > current_max_reward:
                current_max_reward = reward_tuple[1]
                current_max_elem = reward_tuple
                # remove the multiple, are not valid anymore
                multiple_max_rewards = []
                
            if reward_tuple[1] == current_max_reward:
                multiple_max_rewards.append(reward_tuple)
                
        if len(multiple_max_rewards)>=1:
            if current_max_elem[1] > multiple_max_rewards[0][1]:
                return current_max_elem
            
            else:
                random_idx = np.random.randint(0, len(multiple_max_rewards))
                return multiple_max_rewards[random_idx] 
                    
                
        
        return current_max_elem
        






def main(): 
    
    test_state = State(game_board=PLACEHOLDER_GAME_BOARD, lines_cleared= 0, 
                       piece_type=1, piece_count=4, middle_point=(2,9))
    expert = TetrisExpert(actions=ACTIONS)
    
    new_state = expert._simulate_action(state=test_state, action=1)
    
    print("\n\nfinal stats:")
    print(new_state.game_board)
    print(f"new lines cleared: {new_state.lines_cleared}")
    
    
    new_board = expert._rotate_piece(state=test_state, rotation=1)
    print("\n\nrotated board:")
    print(new_board)
    
    print(expert.get_best_move(test_state))
    
    
    
    

if __name__ == '__main__':
    main()
    