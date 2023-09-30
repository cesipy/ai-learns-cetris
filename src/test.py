import subprocess as sub
import os
import time

FIFO = "named_pipe"


def child():
    with open(FIFO) as f:
        while True:
            data = f.read()
            if len(data) != 0:
                print("read: {}".format(data))
            else: break


def main():
    pid = os.fork()

    if pid == 0:
        # child process to handle the tetris game
        time.sleep(1)
        child()

    else:
        # parent
        tetris_command = './tetris_implementation/tetris'

        sub.run(tetris_command, shell=True)


main()
