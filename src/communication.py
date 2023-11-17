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
        """
        receive data (=states) from named pipe.
        """
        try:
          
            # read data from the FIFO
            data = os.read(self.fd_states, 1024)  # Adjust the buffer size as needed

            # log data
            self.logger.log("data read from fifo_states: " + data.decode('utf-8'))


           

            return data.decode('utf-8')  # Assuming data is in UTF-8 encoding
        except FileNotFoundError:
            print(f"Error: {self.fifo_states_name} does not exist.")
            return ""
        except Exception as e:
            print(f"Error while reading from {self.fifo_states_name}: {e}")
            return ""


    def send_to_pipe(self, control) -> None:
        """
        sends control message via named pipe defined in object. 
        """
        """
        try:
            # write data to the FIFO
            os.write(self.fd_controls, control.encode('utf-8'))  # Encode data if not in bytes

        except FileNotFoundError:
            print(f"Error: {self.fifo_controls_name} does not exist.")
            
        except Exception as e:
            print(f"Error while writing to {self.fifo_controls_name}: {e}")

