from config import *
from reward import calculate_reward
from state import State
from simpleLogger import SimpleLogger

from typing import List, Tuple, Optional

import numpy as np



logger = SimpleLogger()


class TetrisExpert:
    
    def __init__(self, actions):
        self.board_dimentsion = PLACEHOLDER_GAME_BOARD.shape
        self.actions     = actions     # is this the correct action space?
        
    def _get_action_mapping(self, action: int)-> Tuple[int, int]: 
        """returns relative position, rotation"""
        action_mapping = {
            # Right movements (negative indices)
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
            19:  (4,3)    # 4left-rotate3
        }
        return action_mapping[action]
        
        
    def get_best_move(self, state: State):
        rewards = []
        for action in self.actions:
            next_state = self._simulate_action(action=action, state=state)
            
            #currently not available
            continue
            reward = calculate_reward(next_state=next_state)
            rewards.append((action, reward))
        
        # consists of action:int, reward:float
        #max_action, max_reward = self._get_max_reward_action(rewards=rewards)
        
        #return max_action
        
    
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
        
        rotated_board = self._rotate_piece(board=state.game_board, rotation=rotation)
        print(f"rotated board for action {action}: \n{rotated_board}")
        
    
    def _get_midde_point(self, state:State):
        if state.piece_type == 0: 
            pass
        
        
    
    
    def _rotate_piece(self, state:State, rotation: int) -> np.ndarray:
        """
        Rotate the falling piece.
        
        Args:
            board (np.ndarray): Current game board
            rotation (int): Number of 90-degree rotations to perform
        
        Returns:
            np.ndarray: Board with rotated piece
        """
        board = state.game_board
        # Create a copy of the board to avoid modifying the original
        rotated_board = np.copy(board)
        result_board  = np.copy(board)
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                condition: bool = board[i][j] == 2
            
                if condition:
                    pass
                    


        
        
        
        
        
        
                
        
    
    def _get_max_reward_action(self, rewards: List[Tuple[int, float]]) -> Tuple[int, float]:
        current_max_reward = float("-inf")
        current_max_elem: Optional[Tuple[int, float]]
        for i in rewards:
            if i[1] > current_max_reward: 
                current_max_reward = i[1]
                current_max_elem = i
        
        return current_max_elem
            

        






def main(): 
    
    test_state = State(game_board=PLACEHOLDER_GAME_BOARD, lines_cleared= 0, 
                       piece_type=1, piece_count=4)
    expert = TetrisExpert(actions=ACTIONS)
    
    expert.get_best_move(test_state)
    
    

if __name__ == '__main__':
    main()
    