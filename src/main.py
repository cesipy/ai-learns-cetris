
import os

import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from config import *

import traceback

from simpleLogger import SimpleLogger
from fork_child import child_function
import config

os.chdir(SRC_DIR)

logger = SimpleLogger()

                                



     
def main():
    
    try: 
        training_process = mp.Process(
            target=child_function,      # this is the agent/ML code
            args=()
        )
        training_process.start()
    
        #import AFTER init
        import subprocess as sub
        tetris_command = config.TETRIS_COMMAND
        status = sub.call(tetris_command)
        logger.log(f"parent process(tetris) exited with code: {status}")
        
        training_process.join()
        
    except Exception as e: 
        logger.log("error")
        raise e

if __name__ == '__main__':
    try: 
        # there was an exception telling me that i need this line for freezuing
        # until bootstrapping is finished
        mp.freeze_support()
        
        main()
    except Exception as e: 
        error_trace = traceback.format_exc()
        
        logger.log(f"Error occurred: {str(e)}\n{error_trace}")
        raise e
