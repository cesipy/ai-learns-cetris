import signal
import subprocess as sub
import os
import time
import random

FIFO_STATES = "fifo_states"
FIFO_CONTROLS = "fifo_controls"
iterations = 100


def parse_control(relative_position_change: int, should_rotate: bool, ):
    pass


def receive_from_pipe():
    try:
        # Open the FIFO for reading
        fifo_fd = os.open(FIFO_STATES, os.O_RDONLY)

        # Read data from the FIFO
        data = os.read(fifo_fd, 100)  # Adjust the buffer size as needed

        # Close the FIFO
        os.close(fifo_fd)

        return data.decode('utf-8')  # Assuming data is in UTF-8 encoding
    except FileNotFoundError:
        print(f"Error: {FIFO_STATES} does not exist.")
        return ""
    except Exception as e:
        print(f"Error while reading from {FIFO_STATES}: {e}")
        return ""


def sent_to_pipe():
    with open(FIFO_CONTROLS, "w") as f:
        data = random.randint(-6, 6)
        f.write(str(data))


def main():
    pid = os.fork()

    if pid == 0:
        # child process to handle the tetris game
        time.sleep(1)
        data: str = ""
        while not data == "end":

            receive_from_pipe()

            time.sleep(350/1000)

            sent_to_pipe()

        print("reached!")
        exit(0)
    else:
        # parent
        tetris_command = './tetris'

        sub.run(tetris_command, shell=True)

        exit(0)


main()
