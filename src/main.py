import subprocess as sub
import os
import time
import numpy as np
import communication
import re

from simpleLogger import SimpleLogger
from metadata import Metadata, State, Game
from q_agent import Agent

SLEEPTIME = 0.001        # default value should be (350/5000)
FIFO_STATES = "fifo_states"
FIFO_CONTROLS = "fifo_controls"
ITERATIONS = 100000   # temp
logger = SimpleLogger()
POSSIBLE_NUMBER_STEPS = 4
ACTIONS = list(range(-16, 20))   # represents left and rotate, left, nothing, right, right and rotate; 
                                 # TODO:  make dependend on POSSIBLE_NUMBER_STEPS
game = Game()
LOAD_MODEL = False          # load model?
EPSILON = 0.99


def parse_state(state_string:str):
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
    
    game_board = game_board[1:]     # remove lines cleared
    logger.log(f"game board: {game_board}")
    logger.log(f"lines cleared: {lines_cleared}")
    state = State(game_board, lines_cleared)
    
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
    fd_controls = os.open(FIFO_CONTROLS, os.O_WRONLY)
    fd_states = os.open(FIFO_STATES, os.O_RDONLY)
    metadata = Metadata(logger, FIFO_STATES, FIFO_CONTROLS, fd_states, fd_controls)
    logger.log(metadata.debug())
    return metadata

def clean_up(metadata: Metadata) -> None:
    os.close(metadata.fd_controls)
    os.close(metadata.fd_states)
    os.unlink(FIFO_CONTROLS)
    logger.log("successfully closed pipes!")

def step(communicator, agent: Agent) -> int:
    received_game_state = communicator.receive_from_pipe()
    logger.log(f"received_game_state: {received_game_state}")
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
    logger.log(f"reward: {reward}\n")
    agent.train(state, action, next_state, reward)

def parse_ending_message(game_state: str) -> int:
    if game_state.startswith("end") or game_state.startswith("game_endend"):
    #if game_state in ["end", "game_endend"]:
        return 1
    elif game_state.startswith("game_end"):
        return 2
    else:
        return 0

def calculate_reward(state: State):
    lines_cleared, height, holes, bumpiness= state.get_values()
    logger.log(f"lines_cleared: {lines_cleared}, height: {height}, holes: {holes}, bumpiness: {bumpiness}")
    weight_lines_cleared = 3
    weight_height = -1.5
    weight_holes = -0.35
    weight_bumpiness = -1.44
    reward = (
        weight_lines_cleared * lines_cleared +
        weight_height * height + 
        weight_holes * holes + 
        weight_bumpiness * bumpiness 
    )
    if lines_cleared > game.lines_cleared_current_epoch:
        logger.log("increase lines_cleared_current_epoch")
        game.set_lines_cleared_current_epoch(lines_cleared)
    return reward

def play_one_round(communicator: communication.Communicator, agent: Agent) -> int:
    return_value = 0
    while True:
        val = step(communicator, agent=agent)
        if val == 1:
            return_value = 1
            break
        elif val == 2: 
            logger.log("one round is finished")
            return_value = 2
            break
    game.update_after_epoch()
    game.set_epsilon(agent.get_epsilon())
    game.increase_epoch()
    logger.log(game)
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
    logger.log(action_space)
    logger.log(ACTIONS)
    return action_space

def main():
    board_shape = (14, 28)
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
            q_table={},
            actions=ACTIONS, 
            action_space_string=action_space, 
            load_model=LOAD_MODEL, 
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
            if game_state == 1: break
            elif game_state == 2:
                current_iteration -= 1
        game.save_model()
        clean_up(meta)
        meta.logger.log("successfully reached end!")
        exit(0)
    else:
        tetris_command = './cpp/tetris'
        status = sub.call(tetris_command)
        logger.log(f"parent process(tetris) exited with code: {status}")
        exit(0)

main()
