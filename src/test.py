import subprocess as sub
import os


def child():
    print("hello from child")


def main():
    pid = os.fork()

    if pid == 0:
        # child process to handle the tetris game
        child()

    else:
        # parent
        tetris_command = './tetris_implementation/tetris'

        sub.run(tetris_command, shell=True)

main()