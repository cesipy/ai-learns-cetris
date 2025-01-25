import numpy as np
import os
from config import *

def log_config_variables():
    """
    Prints all configuration variables defined in the config file.
    Organizes them by category and formats them for readability.
    
    NOTE: This function was AI generated. 
    """
    import inspect
    import sys
    current_module = sys.modules[__name__]
    
    categories = {
        'Training': ['EPSILON', 'EPSILON_DECAY', 'DISCOUNT', 'LEARNING_RATE', 'BATCH_SIZE', 
                    'COUNTER', 'EPOCHS', 'NUM_BATCHES', 'MIN_EPSILON', 'EPSILON_COUNTER_EPOCH'],
        'Model': ['BOARD_HEIGHT', 'BOARD_WIDTH', 'FC_HIDDEN_UNIT_SIZE', 'NUMBER_OF_PIECES'],
        'Environment': ['BASE_DIR', 'SRC_DIR', 'LOG_DIR', 'RES_DIR', 'TETRIS_COMMAND'],
        'Communication': ['FIFO_STATES', 'FIFO_CONTROLS', 'COMMUNICATION_TIME_OUT', 
                         'SLEEPTIME', 'INTER_ROUND_SLEEP_TIME'],
        'Experiment': ['LOGGING', 'LOAD_MODEL', 'ITERATIONS', 'PLOT_COUNTER', 
                      'MOVING_AVG_WINDOW_SIZE', 'COUNTER_TETRIS_EXPERT']
    }
    
    # Get all variables in the module
    all_vars = {name: value for name, value in inspect.getmembers(current_module)
                if not name.startswith('__') and not callable(value)}
    
    # Create timestamp for the log
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare the log output
    log_output = [f"\n{'='*50}",
                  f"Configuration Variables - {timestamp}",
                  f"{'='*50}\n"]
    
    # Log variables by category
    for category, var_names in categories.items():
        log_output.append(f"\n{category}:")
        log_output.append("-" * (len(category) + 1))
        for var_name in var_names:
            if var_name in all_vars:
                value = all_vars[var_name]
                # Format arrays/lists differently
                if isinstance(value, (list, np.ndarray)):
                    if isinstance(value, np.ndarray) and value.size > 100:
                        value_str = f"Array of shape {value.shape}"
                    else:
                        value_str = str(value)
                else:
                    value_str = str(value)
                log_output.append(f"{var_name:.<30} {value_str}")
                all_vars.pop(var_name)
    
    # Log any remaining variables that weren't in categories
    if all_vars:
        log_output.append("\nOther Variables:")
        log_output.append("-" * 14)
        for name, value in sorted(all_vars.items()):
            if not callable(value):
                log_output.append(f"{name:.<30} {str(value)}")
    
    # Print and log the output
    log_text = '\n'.join(log_output)
    
    print(log_text)
    
    return log_text

