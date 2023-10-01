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


def child():
    # with open(FIFO_STATES) as f:
    #    while True:
    #        data = f.read()
    #        if len(data) != 0:
    #            print("read: {}".format(data))
    #        else:
    #            break
    with open(FIFO_CONTROLS, "w") as f:
        data = random.randint(-6, 6)
        f.write(str(data))


def main():
    pid = os.fork()

    if pid == 0:
        # child process to handle the tetris game
        time.sleep(1)
        for _ in range(100):
            child()
            time.sleep(350 / 1000)

        print("done through iterations")
        exit(0)
    else:
        # parent
        tetris_command = './tetris'

        sub.run(tetris_command, shell=True)

        exit(0)


main()
