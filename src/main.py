import subprocess as sub
import os
import time
import numpy as np

FIFO_STATES = "fifo_states"
FIFO_CONTROLS = "fifo_controls"
iterations = 100

# TODO: fifo should be opened only once, not every time 
# `receive_from_pipe()` is created.


def parse_control(relative_position_change: int, should_rotate: bool, ):
    pass


def receive_from_pipe() -> str:
    try:
        # Open the FIFO for reading
        fifo_fd = os.open(FIFO_STATES, os.O_RDONLY)

        # Read data from the FIFO
        data = os.read(fifo_fd, 1024)  # Adjust the buffer size as needed

        # Close the FIFO
        os.close(fifo_fd)

        return data.decode('utf-8')  # Assuming data is in UTF-8 encoding
    except FileNotFoundError:
        print(f"Error: {FIFO_STATES} does not exist.")
        return ""
    except Exception as e:
        print(f"Error while reading from {FIFO_STATES}: {e}")
        return ""


def send_to_pipe(data):
    try:
        # Open the FIFO for writing
        fifo_fd = os.open(FIFO_CONTROLS, os.O_WRONLY)

        control = calculate_current_control(data)

        # Write data to the FIFO
        os.write(fifo_fd, control.encode('utf-8'))  # Encode data if not in bytes

        # Close the FIFO
        os.close(fifo_fd)

    except FileNotFoundError:
        print(f"Error: {FIFO_CONTROLS} does not exist.")
        
    except Exception as e:
        print(f"Error while writing to {FIFO_CONTROLS}: {e}")


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
        data: str = ""
        while True:

            data = receive_from_pipe()
            if data == "end": break

            time.sleep(350/1000)

            send_to_pipe(data)

        print("reached!")
        os.unlink("fifo_controls")
        exit(0)
    else:
        # parent
        # executes the tetris binary
        tetris_command = './tetris'

        status = sub.call(tetris_command)# shell=True)
        print("status: ", status)
        exit(0)


main()
