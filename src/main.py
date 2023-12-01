import subprocess as sub
import os
import time
import numpy as np
import communication

from simpleLogger import SimpleLogger
from metadata import Metadata

SLEEPTIME = 3          # default value should be (350/5000)
FIFO_STATES = "fifo_states"
FIFO_CONTROLS = "fifo_controls"
iterations  = 10   # temp
logger = SimpleLogger()


def parse_control(relative_position_change: int, should_rotate: bool, ):
    pass


def calculate_current_control(data):
    # temporary only generates random number.
    # get normal distributed number: 
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


def generate_random_normal_number(mu, sigma):

    # random number normal distributed
    random_number = np.random.normal(mu, sigma)
    # rount to integers
    number = int(random_number)
   
    return number


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


def step(communicator: communication.Communicator) -> int: 
    received_game_state = communicator.receive_from_pipe()
    if received_game_state == "end": 
        return 1
    
    time.sleep(SLEEPTIME)
    
    # based on current state calculate next control
    control = calculate_current_control(received_game_state)

    communicator.send_to_pipe(control)
    return 0


def main():
    pid = os.fork()

    if pid == 0:
        # child 
        # process to handle the tetris game
        time.sleep(1)
        meta = init()

        communicator = communication.Communicator(meta)

        handshake: str = ""
        # handle handshake
        handshake = communicator.receive_from_pipe()
        logger.log("handshake: "+ handshake)
        #print(handshake)

        communicator.send_handshake(str(iterations))
        logger.log("sent handshake back")

        game_state: str = ""
        while True:

            game_state = step(communicator)
            if game_state: break

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
