from state import State

# this is advanced reward function. maybe simpler is better?
# def calculate_reward(next_state: State):
#     lines_cleared, height, holes, bumpiness = next_state.get_values()
    
#     # rewards for lines cleared -> more lines => better
#     lines_reward = {
#         0: 0,
#         1: 1000,
#         2: 3000,
#         3: 5000,
#         4: 8000
#     }
#     line_clear_reward = lines_reward.get(lines_cleared, 0)
    
#     # penalties on holes, bumpiness and height
#     hole_penalty = -20 * holes                      
#     height_penalty = max(0, -10 * (height - 10))    # only penalty for too high, normal height is ok.
#     bumpiness_penalty = -2 * bumpiness
    
#     # Column height distribution
#     col_heights = next_state.column_heights
#     middle_cols_avg = sum(col_heights[3:7]) / 4                             # Average height of middle columns
#     side_cols_avg = (sum(col_heights[:3]) + sum(col_heights[7:])) / 6       # Average height of side columns
#     balance_bonus = 20 if middle_cols_avg < side_cols_avg else 0            # Reward for keeping middle lower
    

#     #death_penalty = -500 if height >= 20 else 0
    
#     reward = (
#         line_clear_reward +
#         hole_penalty +
#         height_penalty +
#         bumpiness_penalty +
#         balance_bonus 
#        # + death_penalty
#     )
    
#     return reward

# def calculate_reward(next_state: State): 
    
#     lines_cleared, height, holes, bumpiness = next_state.get_values()
    
    
    
#     # # Exponential reward for lines cleared
#     lines_reward = {
#         1: 100,    # Single
#         2: 300,   # Double
#         3: 600,   # Triple
#         4: 1200   # Tetris
#     }.get(lines_cleared, 0)
    
#     if lines_cleared >=4:
#         lines_reward = 1200 + 400*lines_cleared
    
#     reward = (
#         0.766*lines_reward +
#         -0.91*height + 
#         -0.35*holes + 
#         -0.18*bumpiness + 
#         0.200*next_state.piece_count
#     )
    
#     # #piece_count_reward = min(20 * game.current_piece_count, 500)
#     # height_penalty = -0.910 * height #* (1 + game.current_piece_count / 100)
    
#     # tidiness_bonus = 0
#     # if height < 10 and holes == 0:
#     #     tidiness_bonus = 10
    
#     # reward = (
#     #     lines_cleared +
#     #     height_penalty +
#     #     1*next_state.piece_count +
#     #     #piece_c p0ount_reward +
#     #     -0.760 * holes  +      # Quadratic holes penalty
#     #     -0.184 * bumpiness
#     # )
    
#     return reward


# super simple reward
def calculate_reward(next_state: State):
    score = (next_state.piece_count *1 + 
            (next_state.lines_cleared **2) * 14 #14 is width
    )
    return score