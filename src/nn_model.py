from torch import nn
from simpleLogger import SimpleLogger

logger = SimpleLogger()

board_height = 28
board_width  = 10

class CNN(nn.Module):
    def __init__(self, num_actions: int): 
        super().__init__()
        self.conv1 = nn.Conv2d(1,32, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.layers = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
        )
        
        cov_output_size = 64* board_height * board_width       # current board dimensions, maybe change that. 
        
        self.fc1 = nn.Linear(cov_output_size, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_actions)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            self.fc1,
            self.relu2,
            self.fc2
        
        )
        
    def forward(self, x): 
        #logger.log(f"in forward shape: {x.shape}")
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if not present
            #logger.log(f"x after unsqueezing: {x.shape}")
        x = self.layers(x)
        x = self.fc(x)
        return x
        
    