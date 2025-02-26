from state import State
from simpleLogger import SimpleLogger
logger = SimpleLogger()



# # reward from the paper, super simple
# def calculate_reward(next_state: State): 
#     reward = 0

#     reward += (
#         -0.51* next_state.height + 
#         -0.36* next_state.holes + 
#         -0.18* next_state.bumpiness + 
#         0.76 * (next_state.immedeate_lines_cleared ** 2) * 200 +
#         next_state.piece_count
#     )

#     return reward/ 200.0

def calculate_reward(next_state: State):
    reward = 0
    
    if next_state.immedeate_lines_cleared > 0:
        line_cleared_reward = (next_state.immedeate_lines_cleared ** 2) * 200

        reward += line_cleared_reward
        
    survival_bonus = min(1.0* next_state.piece_count, 80)
    reward += survival_bonus
    

    reward -= 0.3 * next_state.get_height_variance() ** 1.5
    reward -= 1.5 * next_state.max_height

    reward -= 0.2 * next_state.bumpiness
    reward -= 0.95 * next_state.holes
    
    if next_state.is_state_game_over():
        game_over_penalty = 500 
        reward -= game_over_penalty
        
    return reward/500.0
