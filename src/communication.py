from simpleLogger import SimpleLogger
import os

class Communicator:
    def __init__(self, metadata ):
        self.logger             = metadata.logger
        self.fifo_states_name   = metadata.fifo_states_name
        self.fifo_controls_name = metadata.fifo_controls_name
        self.fd_states          = metadata.fd_states
        self.fd_controls        = metadata.fd_controls

    def receive_from_pipe(self) -> str:
        try:
            # Open the FIFO for reading
           # fifo_fd = os.open(self.fifo_states_name, os.O_RDONLY)

            # Read data from the FIFO
            data = os.read(self.fd_states, 1024)  # Adjust the buffer size as needed

            # log data
            self.logger.log("data read from fifo_states: " + data.decode('utf-8'))


            # Close the FIFO
            #os.close(fifo_fd)

            return data.decode('utf-8')  # Assuming data is in UTF-8 encoding
        except FileNotFoundError:
            print(f"Error: {self.fifo_states_name} does not exist.")
            return ""
        except Exception as e:
            print(f"Error while reading from {self.fifo_states_name}: {e}")
            return ""


    def send_to_pipe(self, control) -> None:
        try:
            # Open the FIFO for writing
            #fifo_fd = os.open(self.fifo_controls_name, os.O_WRONLY)

            # control = control.encode('utf-8')
            #logger.log("send via fifo_controls: " + str(control))

            # Write data to the FIFO
            os.write(self.fd_controls, control.encode('utf-8'))  # Encode data if not in bytes

            # Close the FIFO
            #os.close(fifo_fd)

        except FileNotFoundError:
            print(f"Error: {self.fifo_controls_name} does not exist.")
            
        except Exception as e:
            print(f"Error while writing to {self.fifo_controls_name}: {e}")

