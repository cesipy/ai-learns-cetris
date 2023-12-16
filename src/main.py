import subprocess as sub
import os
import time
import numpy as np
import communication
import re

from simpleLogger import SimpleLogger
from metadata import Metadata
from metadata import State
from q_agent import Agent

SLEEPTIME = 0.001        # default value should be (350/5000)
FIFO_STATES = "fifo_states"
FIFO_CONTROLS = "fifo_controls"
ITERATIONS    = 100   # temp
logger = SimpleLogger()
ACTIONS = list(range(-16, 20))   # represents left and rotate, left, nothing, right, right and rotate


def parse_state(state_string: str) -> State:
    matches = re.findall(r'\b\d+\b', state_string)
    logger.log(f"matches: {matches}")

    lines_cleared, height, holes, bumpiness, piece_type = map(int, matches)
    
    state = State(lines_cleared, height, holes, bumpiness, piece_type)
    # logger.log(state)

    return state


def parse_control(control) -> str:
    action = 0
    should_rotate = 0
    
    action = control // 2
    
    if control % 2 == 1:
        should_rotate = 1

    control = str(action) + "," + str(should_rotate)

    return control


def calculate_current_control(game_state: State) -> str:
    # temporarily only generates random numbers
    # todo: based on received game state calculate a new control string

    control = generate_random_control()
    
    return control


def generate_random_normal_number(mu, sigma):

    # randomly generated number is normally distributed
    random_number = np.random.normal(mu, sigma)
    # round to integers
    number = int(random_number)
   
    return number


def generate_random_control() -> str:
    # get normally distributed number: 
    # generates new relative position
    mu            = 0
    sigma         = 3.2
    random_number = generate_random_normal_number(mu, sigma)

    #  should piece rotate?
    mu            = 0
    sigma         = 2
    random_rotate =  abs (generate_random_normal_number(mu, sigma))

    should_rotate = 1 if random_rotate else 0

    control       = str(random_number) + "," +  str(should_rotate)    
    
    return control



def init() -> Metadata: 
    """
    opens file descriptors for named pipes.
    for one unit of the program, we need to open it only
    once.
    """
    fd_controls = os.open(FIFO_CONTROLS, os.O_WRONLY)
    fd_states   = os.open(FIFO_STATES, os.O_RDONLY)
    metadata    = Metadata(logger, FIFO_STATES, FIFO_CONTROLS, fd_states, fd_controls)


    logger.log(metadata.debug())
    return metadata


def clean_up(metadata: Metadata) -> None:
    """
    cleans up named fifos. 
    """
    # close the named pipes
    os.close(metadata.fd_controls)
    os.close(metadata.fd_states)
    os.unlink(FIFO_CONTROLS)

    logger.log("successfully closed pipes!")


def step(communicator, agent:Agent) -> int:
    """
    agent steps one step further in environment.
    """
    received_game_state = communicator.receive_from_pipe()
    if received_game_state == "end": 
        return 1
    elif received_game_state == "game_end": 
        return 2
    elif received_game_state == "game_endend":
        return 1
    
    state = parse_state(received_game_state)
    time.sleep(SLEEPTIME)
    action = agent.epsilon_greedy_policy(state)
    perform_action(action, communicator)

    # get next state
    received_game_state = communicator.receive_from_pipe()
    if received_game_state == "end": 
        return 1
    elif received_game_state == "game_end": 
        return 2
    elif received_game_state == "game_endend":
        return 1
    
    next_state = parse_state(received_game_state)
    communicator.send_placeholder_action()
    logger.log("sending fake controls")

    reward = calculate_reward(next_state)
    logger.log(f"reward: {reward}\n")

    agent.train(state, action, next_state, reward)


def calculate_reward(state: State):
    lines_cleared, height, holes, bumpiness, piece_type = state.get_values()

    # only temp values: magic numbers
    weight_lines_cleared = 1.0
    weight_height = -0.1
    weight_holes = -1.0
    weight_bumpiness = -0.5
    weight_piece_type = 0.1

    reward = (
        weight_lines_cleared * lines_cleared +
        weight_height * height + 
        weight_holes * holes + 
        weight_bumpiness * bumpiness + 
        weight_piece_type * piece_type 
    )
    
    return reward


def play_one_round(communicator: communication.Communicator, agent: Agent) -> int:
    """
    finishes one episode.

    @param communicator - communicator object used to communicate via named pipe.
    @param agent 
    """
    
    while True:

        val = step(communicator, agent=agent)
        if val == 1: return 1
        elif val == 2: 
            logger.log("one round is finished")
            return 2
    

def perform_action(control, communicator: communication.Communicator):
    action: str = parse_control(control)
    logger.log(f"action performed: {action}")
    communicator.send_to_pipe(action)


def construct_action_space():
    action_space = []

    for i in range (-4, 5):
        if i < 0:
            direction = "right"
        else:
            direction = "left"

        for j in range (0, 4):

            action_space.append(str(i)+direction+"-rotate"+str(j))
    logger.log(action_space)
    logger.log(ACTIONS)
    return action_space



def main():
    pid = os.fork()

    if pid == 0:
        # child 
        # process to handle the tetris game
        time.sleep(1)
        meta = init()

        action_space = construct_action_space()
        communicator = communication.Communicator(meta)
        agent = Agent(n_neurons=30,
                      epsilon=0.3,
                      q_table={},
                      actions=ACTIONS, 
                      action_space_string=action_space)

        handshake: str = ""
        # handle handshake
        handshake = communicator.receive_from_pipe()
        logger.log("handshake: "+ handshake)
        #print(handshake)

        communicator.send_handshake(str(ITERATIONS))
        logger.log("sent handshake back")

        game_state: int = 0
        current_iteration = ITERATIONS

        while True:
            
            # each one episode played
            game_state = play_one_round(communicator, agent)
            #game_state = step(communicator)
            if game_state == 1: break
            elif game_state == 2:
                current_iteration -= 1

        clean_up(meta)                  # close named pipes
        meta.logger.log("successfully reached end!")

        exit(0)
    else:
        # parent
        # executes the tetris binary
        tetris_command = './cpp/tetris'

        status = sub.call(tetris_command)
        logger.log(f"parent process(tetris) exited with code: {status}")
        exit(0)


main()
