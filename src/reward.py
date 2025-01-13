from state import State
from simpleLogger import SimpleLogger
logger = SimpleLogger()

# super simple reward
# def calculate_reward(next_state: State):

#     reward = 0
    
#     #add basic reward for surviving: 
#     reward += 0.2 * next_state.piece_count
    
#     if next_state.immedeate_lines_cleared > 0:
#         reward += (next_state.immedeate_lines_cleared ** 2) * 100
        
#     # Heavy punishment for game over (when game terminates)  
#     if next_state.is_state_game_over():
#         reward -= 600
        
#     return reward


def calculate_reward(next_state: State):
    reward = 0
    
    if next_state.immedeate_lines_cleared > 0:
        reward += (next_state.immedeate_lines_cleared ** 2) * 150
    
    # Survival reward - scale with height to encourage lower stacks
    survival_bonus = max(0, 20 - next_state.max_height) * 0.1
    reward += survival_bonus
    
    if next_state.max_height > 10:
        # Make height penalty exponential after certain threshold
        height_penalty = ((next_state.max_height - 10) ** 1.5) * 0.5
        reward -= height_penalty
    
    # Hole penalties - make them more punishing as they accumulate
    hole_penalty = (next_state.holes ** 1.5) * 0.8
    reward -= hole_penalty
    
    # Structure penalties - scaled down to not overshadow main objectives
    reward -= 0.5 * next_state.get_height_variance()
    reward -= 0.3 * next_state.get_max_height_diff()
    

    reward -= 0.2 * next_state.bumpiness
    
    if next_state.is_state_game_over():
        game_over_penalty = 300 + next_state.max_height * 2
        reward -= game_over_penalty
        
    return reward


# super simple reward for only expert
def calculate_reward_tetris_expert(next_state: State):
#     score = (-0.1* next_state.holes+
#             (4*next_state.lines_cleared **2) * 14 #14 is width
#     )
    
#     score -= 0.2*next_state.max_height
#     return score
    
    lines_cleared, height, holes, bumpiness = next_state.get_values()
    
    
    
    # # Exponential reward for lines cleared
    lines_reward = {
        1: 100,    # Single
        2: 300,   # Double
        3: 600,   # Triple
        4: 1200   # Tetris
    }.get(lines_cleared, 0)
    
    if lines_cleared >=4:
        lines_reward = 1200 + 400*lines_cleared
    
    reward = (
        0.766*lines_reward +
        -0.91*height + 
        -0.35*holes + 
        -0.18*bumpiness + 
        0.200*next_state.piece_count
    )
    
    # #piece_count_reward = min(20 * game.current_piece_count, 500)
    # height_penalty = -0.910 * height #* (1 + game.current_piece_count / 100)
    
    # tidiness_bonus = 0
    # if height < 10 and holes == 0:
    #     tidiness_bonus = 10
    
    reward = (
        lines_reward +
        #-0.54  * height +
        -0.6   * next_state.max_height +
        -0.20 * holes        # Quadratic holes penalty
        # -0.184 * bumpiness
    )
    
    # score = (next_state.piece_count *1 + 
    #         (next_state.lines_cleared **2) * 14 #14 is width
    # )
    
    # score -= 0.05*next_state.bumpiness
    # score -= 0.1*next_state.max_height
    # return score
    
    return reward