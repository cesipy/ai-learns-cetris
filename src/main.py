import subprocess as sub
import os
import time
import numpy as np
import communication
import re
import config
from config import *

from typing import List

from simpleLogger import SimpleLogger
from metadata import Metadata, State, Game
from q_agent import Agent
from communication import Communicator

os.chdir(SRC_DIR)

SLEEPTIME = 0.00001        # default value should be (350/5000)

ITERATIONS = 100000   # temp
logger = SimpleLogger()
POSSIBLE_NUMBER_STEPS = 4
ACTIONS = list(range(-16, 20))   # represents left and rotate, left, nothing, right, right and rotate; 
                                 # TODO:  make dependend on POSSIBLE_NUMBER_STEPS
game = Game()
LOAD_MODEL = False          # load model?



def parse_state(state_string:str):
    if LOGGING:
        logger.log(f"state_string: {state_string}")
    game_board = []
    row = []
    for char in state_string: 
        if char == ",":
            game_board.append(row)
            row = []
        else:
            row.append(int(char))
    
    lines_cleared = game_board[0][0]
    
    game_board = game_board[1:]     # remove lines cleared parameter
    if LOGGING:
        logger.log(f"game board: {game_board}")
        logger.log(f"lines cleared: {lines_cleared}")
    state = State(game_board, lines_cleared)
    
    if game.lines_cleared_current_epoch < lines_cleared:
        game.lines_cleared_current_epoch = lines_cleared
    
    return state


def parse_control(control) -> str:
    action = control // 2
    should_rotate = 1 if control % 2 == 1 else 0
    control = f"{action},{should_rotate}"
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
    # create FIFOs if they don't exist
    try:
        if not os.path.exists(FIFO_CONTROLS):
            os.mkfifo(FIFO_CONTROLS)
            logger.log(f"Created FIFO_CONTROLS at {FIFO_CONTROLS}")
        if not os.path.exists(FIFO_STATES):
            os.mkfifo(FIFO_STATES)
            logger.log(f"Created FIFO_STATES at {FIFO_STATES}")
    except OSError as e:
        logger.log(f"Error creating FIFOs: {e}")
        raise

    try:
        fd_controls = os.open(FIFO_CONTROLS, os.O_WRONLY)
        fd_states = os.open(FIFO_STATES, os.O_RDONLY)
        metadata = Metadata(logger, FIFO_STATES, FIFO_CONTROLS, fd_states, fd_controls)
        logger.log(metadata.debug())
        return metadata
    except OSError as e:
        logger.log(f"Error opening FIFOs: {e}")
        raise


def clean_up(metadata: Metadata) -> None:
    os.close(metadata.fd_controls)
    os.close(metadata.fd_states)
    os.unlink(FIFO_CONTROLS)
    logger.log("successfully closed pipes!")


def step(communicator, agent: Agent) -> int:
    if LOGGING:
        return step_verbose(communicator, agent)
    else:
        return step_minimal(communicator, agent)

def step_verbose(communicator: Communicator, agent: Agent) -> int:
    received_game_state = communicator.receive_from_pipe()
    logger.log(f"received_game_state1: {received_game_state}")
    status = parse_ending_message(received_game_state)
    logger.log(f"status after parse ending message: {status}")
    if status: return status
    
    state = parse_state(received_game_state)
    logger.log(f"parsed state: {state}")
    time.sleep(SLEEPTIME)
    action = agent.epsilon_greedy_policy(state)
    perform_action(action, communicator)
    received_game_state = communicator.receive_from_pipe()
    logger.log(f"received_game_state2: {received_game_state}")
    status = parse_ending_message(received_game_state)
    logger.log(f"status after parse ending message2: {status}")
    if status: return status
    
    next_state = parse_state(received_game_state)
    logger.log(f"parsed next state: {next_state}")
    communicator.send_placeholder_action()
    
    reward = calculate_reward(next_state)
    game.current_rewards.append(reward)         # add reward for mean reward calculation
    
    logger.log(f"reward: {reward}\n")
    agent.train(state, action, next_state, reward)
    return 0

def step_minimal(communicator, agent: Agent) -> int:
    # First state
    received_game_state = communicator.receive_from_pipe()
    status = parse_ending_message(received_game_state)
    if status: return status
    
    state = parse_state(received_game_state)
    time.sleep(SLEEPTIME)
    action = agent.epsilon_greedy_policy(state)
    perform_action(action, communicator)
    
    # Next state
    received_game_state = communicator.receive_from_pipe()
    status = parse_ending_message(received_game_state)
    if status: return status
    
    next_state = parse_state(received_game_state)
    communicator.send_placeholder_action()
    reward = calculate_reward(next_state)
    game.current_rewards.append(reward)         # add reward for mean reward calculation
    
    agent.train(state, action, next_state, reward)
    return 0


def parse_ending_message(game_state: str) -> int:
    if game_state.startswith("end") or game_state.startswith("game_endend"):
    #if game_state in ["end", "game_endend"]:
        return 1
    elif game_state.startswith("game_end"):
        return 2
    else:
        return 0

# this is advanced reward function. maybe simpler is better?
def calculate_reward(next_state: State):
    lines_cleared, height, holes, bumpiness = next_state.get_values()
    
    # rewards for lines cleared -> more lines => better
    lines_reward = {
        0: 0,
        1: 1000,
        2: 3000,
        3: 5000,
        4: 8000
    }
    line_clear_reward = lines_reward.get(lines_cleared, 0)
    
    # penalties on holes, bumpiness and height
    hole_penalty = -20 * holes                      
    height_penalty = max(0, -10 * (height - 10))    # only penalty for too high, normal height is ok.
    bumpiness_penalty = -2 * bumpiness
    
    # Column height distribution
    col_heights = next_state.column_heights
    middle_cols_avg = sum(col_heights[3:7]) / 4                             # Average height of middle columns
    side_cols_avg = (sum(col_heights[:3]) + sum(col_heights[7:])) / 6       # Average height of side columns
    balance_bonus = 20 if middle_cols_avg < side_cols_avg else 0            # Reward for keeping middle lower
    

    #death_penalty = -500 if height >= 20 else 0
    
    reward = (
        line_clear_reward +
        hole_penalty +
        height_penalty +
        bumpiness_penalty +
        balance_bonus 
       # + death_penalty
    )
    
    return reward


def play_one_round(communicator: communication.Communicator, agent: Agent) -> int:
    game.start_time_measurement()
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
    current_lines_cleared = game.lines_cleared_current_epoch
    game.update_after_epoch()
    game.set_epsilon(agent.get_epsilon())
    game.increase_epoch()
    
    elapsed_time = game.end_time_measurement()
    
    logger.log(game.print_with_stats(current_lines_cleared=current_lines_cleared, elapsed_time=elapsed_time))
    if LOGGING:
        logger.log(f"return_value in play one round: {return_value}")
    return return_value


def perform_action(control, communicator: communication.Communicator):
    action: str = parse_control(control)
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
            x=list(range(len(mean_rewards))),
            y=mean_rewards,
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
    
     
def main():
    num_actions = len(ACTIONS)
    board_shape = (28,14)
    pid = os.fork()
    if pid == 0:
        time.sleep(1)
        meta = init()
        if LOAD_MODEL:
            game.load_model()
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
        handshake = communicator.receive_from_pipe()
        logger.log(f"handshake: {handshake}")
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
    else:
        tetris_command = config.TETRIS_COMMAND
        status = sub.call(tetris_command)
        logger.log(f"parent process(tetris) exited with code: {status}")
        exit(0)

main()
