import subprocess as sub
import os
import time
import numpy as np
import communication

from simpleLogger import SimpleLogger

FIFO_STATES = "fifo_states"
FIFO_CONTROLS = "fifo_controls"
iterations = 100
logger = SimpleLogger()

# TODO: fifo should be opened only once, not every time 
# `receive_from_pipe()` is created.


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


def main():
    pid = os.fork()

    if pid == 0:
        # child process to handle the tetris game
        time.sleep(1)
        communicator = communication.Communicator(logger, FIFO_STATES, FIFO_CONTROLS)
        data: str = ""
        while True:

            data = communicator.receive_from_pipe()
            if data == "end": break

            time.sleep(350/1000)

            communicator.send_to_pipe(data)

        logger.log("successfully reached end!")
        os.unlink(FIFO_CONTROLS)
        exit(0)
    else:
        # parent
        # executes the tetris binary
        tetris_command = './tetris'

        status = sub.call(tetris_command)# shell=True)
        logger.log("parent process(tetris) exited successfully!")
        exit(0)


main()
