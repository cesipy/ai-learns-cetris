class Metadata:
    def __init__(self, logger, fifo_states_name, fifo_controls_name, fd_states:int, fd_controls:int):
        self.logger = logger
        self.fifo_states_name   = fifo_states_name
        self.fifo_controls_name = fifo_controls_name
        self.fd_states = fd_states
        self.fd_controls = fd_controls
        
    def debug(self):
        return "fd_states: " + str(self.fd_states) + " fd_controls: " + str(self.fd_controls)
    


    
class State:
    def __init__(self, lines_cleared, height, holes, bumpiness, piece_type):
        self.lines_cleared = lines_cleared
        self.height        = height
        self.holes         = holes
        self.bumpiness     = bumpiness
        self.piece_type    = piece_type

    def __repr__(self):
        message: str = f"""
    lines cleared: {self.lines_cleared}
    holes: {self.holes}
    height: {self.height}
    bumpiness {self.bumpiness}
    piece type {self.piece_type}
                        """
        return message