from config import *
from reward import calculate_reward
from state import State
from simpleLogger import SimpleLogger

from typing import List, Tuple, Optional

import numpy as np




class TetrisExpert:
    
    def __init__(self, actions):
        self.board_dimentsion = PLACEHOLDER_GAME_BOARD.shape
        self.actions     = actions     # is this the correct action space?
        
    def get_best_move(self, state: State):
        rewards = []
        for action in self.actions:
            next_state = self._simulate_action(action=action, state=state)
            reward = calculate_reward(next_state=next_state)
            rewards.append((action, reward))
        
        # consists of action:int, reward:float
        max_action, max_reward = self._get_max_reward_action(rewards=rewards)
        
        return max_action
        

        
    
    
    def _simulate_action(self, state: State, action) -> State: 
        """based on current state, the action is performed and the resulting state is returned"""
        current_board = np.copy(state.game_board)
        
        piece_positions = []
        for i in range(len(current_board)):
            for j in range(len(current_board)):
                if current_board[i][j]== 2:     # falling piece
                    piece_positions.append((i,j))
                    
        if not piece_positions:
            # no falling positions??
            return None
        
        
                
        
    
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
    