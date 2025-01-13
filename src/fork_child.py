
import os



import subprocess as sub
import time
import numpy as np
import communication
import re
import config
from config import *

import traceback
import multiprocessing as mp

from typing import List

from simpleLogger import SimpleLogger
from metadata import Metadata
from state import State
from game import Game

from communication import Communicator
from reward import calculate_reward

os.chdir(SRC_DIR)

logger = SimpleLogger()

def child_function():
    
    #currently all of this code is a bit messy with all the sub functions
    # only copied all the code directly from the main, to adapt it to the 
    # transition from forking to mp.Process
    # consider refactoring
    game = Game()
    def parse_state(state_string:str, piece_count):
        #logger.log(f"state string: {state_string}")
        if LOGGING:
            logger.log(f"state_string: {state_string}")
        game_board = []
        row = []
        number_buffer = ""  # to collect digits for multi-digit numbers
        
        first_numbers = []
        comma_count = 0
        i = 0
        while comma_count < 4:  # first few meta numbers
            if state_string[i] == ',':
                first_numbers.append(int(number_buffer))
                number_buffer = ""
                comma_count += 1
            else:
                number_buffer += state_string[i]
            i += 1
        
        # process the game board
        for char in state_string[i:]:
            if char == ",":
                game_board.append(row)
                row = []
            else:
                row.append(int(char))
                
        lines_cleared = first_numbers[0]  # Now this will be the full number
        piece_type = first_numbers[1]
        middle_p_x = first_numbers[2]
        middle_p_y = first_numbers[3]
        
        game_board = game_board[:]     # already have metadata handled
        
        #logger.log(f"in parse_state:lines_cleared: {lines_cleared}")
        
        if LOGGING:
            logger.log(f"game board: {game_board}")
            logger.log(f"lines cleared: {lines_cleared}")
        
        state = State(
            game_board=game_board, 
            lines_cleared=lines_cleared, 
            piece_type=piece_type, 
            piece_count=piece_count, 
            middle_point=(middle_p_x, middle_p_y))
        
        if game.lines_cleared_current_epoch < lines_cleared:
            game.lines_cleared_current_epoch = lines_cleared
        
        return state


    def parse_control(control) -> str:
        #logger.log(f"control before parsing: {control}")
        action_mapping = {
                # Right movements (negative indices)

                # -32: (-8,0),  # -8right-rotate0
                # -31: (-8,1),  # -8right-rotate1
                # -30: (-8,2),  # -8right-rotate2
                # -29: (-8,3),  # -8right-rotate3
                # -28: (-7,0),  # -7right-rotate0
                # -27: (-7,1),  # -7right-rotate1
                # -26: (-7,2),  # -7right-rotate2
                # -25: (-7,3),  # -7right-rotate3
                # -24: (-6,0),  # -6right-rotate0
                # -23: (-6,1),  # -6right-rotate1
                # -22: (-6,2),  # -6right-rotate2
                # -21: (-6,3),  # -6right-rotate3
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
            }

        action = action_mapping[control]
        #logger.log(f"control in parse_control: {control}, action string: {action}")
        
        new_rel_position =action[0]
        rotation         =action[1]
        
        control = f"{new_rel_position},{rotation}"
        #logger.log(f"control after parsing in parse_control: {control}")
        return control


    def calculate_current_control(game_state: State) -> str:
        control = generate_random_control()
        return control

    def generate_random_normal_number(mu, sigma):
        random_number = np.random.normal(mu, sigma)
        number = int(random_number)
        return number


    def generate_random_control() -> str:
        mu, sigma = 0, 3.2
        random_number = generate_random_normal_number(mu, sigma)
        mu, sigma = 0, 2
        random_rotate = abs(generate_random_normal_number(mu, sigma))
        should_rotate = 1 if random_rotate else 0
        control = f"{random_number},{should_rotate}"    
        return control


    def init() -> Metadata:
        try:
            fd_controls = os.open(FIFO_CONTROLS, os.O_WRONLY)
            
            # named pipes are created in c++. 
            # they are created with mkfifo and then have to be opend by bothends. 
            # therefore we need to wait here some time for c code to mk the pipe
            # c: mkfifo(fd_controls)
            # python: open(fd_controls)
            # ------- (sleeping, that c has time)
            # c mkfifo(fd_states)
            # python: open(fd_states)
            time.sleep(1)
            
            fd_states = os.open(FIFO_STATES, os.O_RDONLY)

            metadata = Metadata(logger, FIFO_STATES, FIFO_CONTROLS, fd_states, fd_controls)
            logger.log(metadata.debug())
            return metadata
        except OSError as e:
            logger.log(f"Error opening FIFOs: {e}")
            raise


    def clean_up(metadata: Metadata) -> None:
        try: 
            os.close(metadata.fd_controls)
            os.close(metadata.fd_states)
            os.unlink(FIFO_CONTROLS)
            logger.log("successfully closed pipes!")
        except Exception as e: 
            logger.log(f"problem cleaning up: {e}")
            


    def step(communicator, agent) -> int:
        
        if LOGGING:
            return step_verbose(communicator, agent)
        else:
            return step_minimal(communicator, agent)

    def step_verbose(communicator: Communicator, agent) -> int:
        received_game_state = communicator.receive_from_pipe()
        logger.log(f"received_game_state1: {received_game_state}")
        status = parse_ending_message(received_game_state)
        logger.log(f"status after parse ending message: {status}")
        if status: return status
        
        state = parse_state(received_game_state, game.current_piece_count)
        logger.log(f"parsed state: {state}")
        time.sleep(SLEEPTIME)
        action = agent.epsilon_greedy_policy(state)
        perform_action(action, communicator)
        
        current_lines_cleared = state.lines_cleared
        
        
        # new state
        received_game_state = communicator.receive_from_pipe()
        logger.log(f"received_game_state2: {received_game_state}")
        status = parse_ending_message(received_game_state)
        logger.log(f"status after parse ending message2: {status}")
        if status: return status
        
        game.current_piece_count +=1        # piece count increases only here
        
        next_state = parse_state(received_game_state, game.current_piece_count)
        logger.log(f"parsed next state: {next_state}")
        communicator.send_placeholder_action()
        
        if next_state.lines_cleared - current_lines_cleared > 0:
            next_state.immedeate_lines_cleared = next_state.lines_cleared - current_lines_cleared
        else: 
            next_state.immedeate_lines_cleared = 0
        
        reward = calculate_reward(next_state)
        game.current_rewards.append(reward)         # add reward for mean reward calculation
        
        logger.log(f"reward: {reward}\n")
        agent.train(state, action, next_state, reward)
        return 0

    def step_minimal(communicator: Communicator, agent) -> int:
        # First state
        received_game_state = communicator.receive_from_pipe()
        status = parse_ending_message(received_game_state)
        if status: return status
        
        state = parse_state(received_game_state, game.current_piece_count)
        #logger.log(f"state::\n{state.game_board}")
        time.sleep(SLEEPTIME)
        action = agent.epsilon_greedy_policy(state)
        perform_action(action, communicator)

        
        current_lines_cleared = state.lines_cleared
        
        # Next state
        received_game_state = communicator.receive_from_pipe()
        status = parse_ending_message(received_game_state)
        if status: return status
        
        game.current_piece_count +=1        # piece count increases only here
        
        next_state = parse_state(received_game_state, game.current_piece_count)
        #logger.log(f"next_state::\n{next_state.game_board}")
        communicator.send_placeholder_action()
    
        if next_state.lines_cleared - current_lines_cleared > 0:
            next_state.immedeate_lines_cleared = next_state.lines_cleared - current_lines_cleared

        else: 
            next_state.immedeate_lines_cleared = 0
        
        reward = calculate_reward(next_state)
        game.current_rewards.append(reward)         # add reward for mean reward calculation
        
        agent.train(state, action, next_state, reward)
        #logger.log("-----------------------\n")
        return 0


    def parse_ending_message(game_state: str) -> int:
        if game_state.startswith("end") or game_state.startswith("game_endend"):
        #if game_state in ["end", "game_endend"]:
            return 1
        elif game_state.startswith("game_end"):
            return 2
        else:
            return 0


    def play_one_round(communicator: communication.Communicator, agent) -> int:
        game.start_time_measurement()
        logger.log("started new round")
        if LOGGING:
            logger.log("entering play_one_round")
            
        return_value = 0
        while True:
            val = step(communicator, agent=agent)
            if val == 1:
                return_value = 1
                break
            elif val == 2: 
                if LOGGING:
                    logger.log("one round is finished")
                    
                return_value = 2
                break
            
            #game.current_piece_count += 1
        
        current_lines_cleared = game.lines_cleared_current_epoch
        game.update_after_epoch()
        game.set_epsilon(agent.get_epsilon())
        game.increase_epoch()
        
        elapsed_time = game.end_time_measurement()
        current_avg_reward = game.mean_rewards[-1]
        
        logger.log(game.print_with_stats(
            current_lines_cleared=current_lines_cleared, 
            elapsed_time=elapsed_time, 
            avg_reward=current_avg_reward,
            ))
        if LOGGING:
            logger.log(f"return_value in play one round: {return_value}")

        time.sleep(INTER_ROUND_SLEEP_TIME)
        communicator.send_to_pipe("ready")
        return return_value


    def perform_action(control, communicator: communication.Communicator):
        action: str = parse_control(control)
        #logger.log(f"action string in perform_action, before sending: {action}")
        communicator.send_to_pipe(action)


    def construct_action_space(n):
        action_space = []
        for i in range(-n, n+1):
            direction = "right" if i < 0 else "left"
            for j in range(0, 4):
                action_space.append(f"{i}{direction}-rotate{j}")
        if LOGGING:
            logger.log(f"action space: {action_space}")
            logger.log(ACTIONS)
        
        logger.log(f"action space: {action_space}")
        logger.log(ACTIONS)
        return action_space

    # def plot_lines_cleared(lines_cleared_array: List[int]):
    #     #TODO: maybe do this in with moving average
    #     import plotly.graph_objects as go
        
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=list(range(len(lines_cleared_array))), 
    #                             y=lines_cleared_array))
    #     fig.update_layout(title="Lines Cleared per Epoch",
    #                      xaxis_title="Epoch",
    #                      yaxis_title="Lines Cleared")
        
    #     fig.write_html(os.path.join(RES_DIR, "lines_cleared.html"))

    def plot_lines_cleared(lines_cleared_array: List[int], mean_rewards: List[float]):
        """
        Plot both lines cleared and mean rewards per epoch
        """
        import plotly.graph_objects as go
        #TODO: maybe do this in with moving average
        
        # MAs
        def moving_average(arr, window=3):
            import pandas as pd
            series = pd.Series(arr)
            return series.rolling(window=window, min_periods=1, center=True).mean()
        
        #lines_cleared_array = np.cumsum(lines_cleared_array)
        lines_cleared_array = moving_average(lines_cleared_array, window=MOVING_AVG_WINDOW_SIZE)
        new_mean_rewards        = moving_average(mean_rewards, window=MOVING_AVG_WINDOW_SIZE)
        logger.log(f"difference of lengths: {len(new_mean_rewards)} vs {len(mean_rewards)}")
        # logger.log(f"MA lines cleared: {lines_cleared_array}")
        # logger.log(f"MA mean rewards: {mean_rewards}")
        
        fig = go.Figure()
        
        # add lines cleared trace
        fig.add_trace(
            go.Scatter(
                x=list(range(len(lines_cleared_array))),
                y=lines_cleared_array,
                name="Lines Cleared",
                line=dict(color='blue')
            )
        )
        
        # Add mean rewards trace with secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=list(range(len(new_mean_rewards))),
                y=new_mean_rewards,
                name="Mean Reward",
                line=dict(color='red'),
                yaxis="y2"
            )
        )
        
        # for mean reward
        fig.update_layout(
            title="Training Progress per Epoch",
            xaxis_title="Epoch",
            yaxis_title="Lines Cleared",
            yaxis2=dict(
                title="Mean Reward",
                overlaying="y",
                side="right"
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.write_html(os.path.join(RES_DIR, "training_progress.html"))  
        
    try: 
        num_actions = len(ACTIONS)
        board_shape = (28,14)
        from q_agent import Agent
        time.sleep(2)
        meta = init()
        # if LOAD_MODEL:
        #     game.load_model()
        action_space = construct_action_space(POSSIBLE_NUMBER_STEPS)
        communicator = communication.Communicator(meta)
        agent = Agent(
            n_neurons=200,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            q_table={},
            actions=ACTIONS, 
            action_space_string=action_space, 
            load_model=LOAD_MODEL, 
            num_actions=num_actions, 
            board_shape=board_shape
        )
        #logger.log("agent initialized")
        time.sleep(1)
        handshake = communicator.receive_from_pipe()
        #logger.log(f"handshake: {handshake}")
        communicator.send_handshake(str(ITERATIONS))
        logger.log("sent handshake back")
        game_state = 0
        current_iteration = ITERATIONS
        while True:
            game_state = play_one_round(communicator, agent)
            
            # save visualisations
            if current_iteration % PLOT_COUNTER == 0:
                plot_lines_cleared(game.lines_cleared_array, game.mean_rewards)
                
            
            if game_state == 1: 
                break
            elif game_state == 2:
                current_iteration -= 1
        game.save_model()
        clean_up(meta)
        meta.logger.log("successfully reached end!")
        exit(0)
    except Exception as e: 
        error_trace = traceback.format_exc()
        
        logger.log(f"Error occurred: {str(e)}\n{error_trace}")
        raise e
