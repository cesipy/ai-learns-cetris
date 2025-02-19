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
        fraction = next_state.piece_count / 70
        if fraction > 1: 
            #logger.log("70 pieces reached!")
            line_cleared_reward = (next_state.immedeate_lines_cleared ** 2) * 200
        else:
            line_cleared_reward = (next_state.immedeate_lines_cleared ** 2) * 200 * fraction*2
        #logger.log(f"line cleared reward: {line_cleared_reward}")
        reward += line_cleared_reward
        
    survival_bonus = 1.0* next_state.piece_count 
    reward += survival_bonus
    

    reward -= 0.3 * next_state.get_height_variance() ** 1.5
    reward -= 1.5 * next_state.max_height

    reward -= 0.2 * next_state.bumpiness
    reward -= 0.95 * next_state.holes
    
    if next_state.is_state_game_over():
        game_over_penalty = 500 
        reward -= game_over_penalty
        #logger.log(f"game over reward: {reward}")
        
    return reward/500.0



# def calculate_reward(next_state: State):
#     lines_cleared = 0
#     if next_state.immedeate_lines_cleared: 
#         lines_cleared = next_state.immedeate_lines_cleared
#     reward = (
#         next_state.piece_count  + 
#         (lines_cleared ** 2.5) * 100
#     )

#     reward -= next_state.holes * 0.7

#     if next_state.is_state_game_over(): 
#         reward -= 500
#     return reward/200.0

# def calculate_reward(next_state: State):

#     reward = 0
    
#     #add basic reward for surviving: 
#     reward += 1.4 * next_state.piece_count ** 1.25
    
#     if next_state.immedeate_lines_cleared > 0:
#         line_weights = {1: 100, 2: 300, 3: 600, 4: 1200}
#         reward += line_weights.get(next_state.immedeate_lines_cleared, 0)
        
        
#     reward += -1.5 * next_state.get_height_variance()**1.8
#     reward -= 5.5 * next_state.max_height                    # Penalize tall stacks
#     reward -= .6 * next_state.holes ** 1.54                        # Strong hole penalty
#     reward -= .7 * next_state.bumpiness 
        
#     # Heavy punishment for game over (when game terminates)  
#     if next_state.is_state_game_over():
#         reward -= 600
        
#     return reward/1000.0




# def calculate_reward(next_state: State):
#     reward = 0
    
#     if next_state.immedeate_lines_cleared > 0:
#         fraction = next_state.piece_count / 70
#         if fraction > 1: 
#             #logger.log("70 pieces reached!")
#             line_cleared_reward = (next_state.immedeate_lines_cleared ** 2) * 200
#         else:
#             line_cleared_reward = (next_state.immedeate_lines_cleared ** 2) * 200 * fraction*2
#         #logger.log(f"line cleared reward: {line_cleared_reward}")
#         reward += line_cleared_reward
        
#     survival_bonus = 0.5* next_state.piece_count 
#     reward += survival_bonus
    

#     reward -= 0.3 * next_state.get_height_variance() ** 1.5
#     reward -= 1.5 * next_state.max_height

#     reward -= 0.2 * next_state.bumpiness
#     reward -= 0.95 * next_state.holes
    
#     if next_state.is_state_game_over():
#         game_over_penalty = 1000 
#         reward -= game_over_penalty
#         #logger.log(f"game over reward: {reward}")
        

#     return reward/10


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