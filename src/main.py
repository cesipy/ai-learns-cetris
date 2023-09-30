import subprocess as sub
import os
import time
import random

FIFO_STATES   = "fifo_states"
FIFO_CONTROLS = "fifo_controls"
iterations = 100


def child():
    #with open(FIFO_STATES) as f:
    #    while True:
    #        data = f.read()
    #        if len(data) != 0:
    #            print("read: {}".format(data))
    #        else:
    #            break

    with open(FIFO_CONTROLS, "w") as f:
        data = random.randint(0, 100)
        f.write(str(data))



def main():
    pid = os.fork()

    if pid == 0:
        # child process to handle the tetris game
        time.sleep(1)
        while True:
            child()
            time.sleep(350/1000)

    else:
        # parent
        tetris_command = './tetris'

        sub.run(tetris_command, shell=True)


main()
