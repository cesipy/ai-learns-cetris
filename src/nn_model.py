from torch import nn
from simpleLogger import SimpleLogger
from config import *

import torch

logger = SimpleLogger()

# class CNN(nn.Module):
#     def __init__(self, num_actions: int): 
#         super().__init__()
    #     self.conv1 = nn.Conv2d(1,32, 3, padding=1)
    #     self.relu = nn.ReLU()
    #     self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
    #     self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
    #     self.first_cnn_layers = nn.Sequential(
    #         self.conv1,
    #         nn.BatchNorm2d(32),
    #         self.relu,
    #         self.conv2,
    #         nn.BatchNorm2d(32),
    #         self.relu,
    #         self.conv3,
    #         nn.BatchNorm2d(64),
    #         self.relu,
    #     )
        
    #     self.column_collapse = nn.AdaptiveAvgPool2d((BOARD_WIDTH, 1))
        
    #     self.post_collapse_layers = nn.Sequential(
    #         nn.Conv2d(64, 128, 3, padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #         nn.Conv2d(128, 128, 1),  # 1x1 convolution
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #         nn.Conv2d(128, 128, 3, padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #     )
        
    #     cov_output_size = 128 * BOARD_WIDTH # *  BOARD_HEIGHT       # current board dimensions, maybe change that.
    #     piece_type_size = NUMBER_OF_PIECES
        
    #     self.fc1 = nn.Linear(cov_output_size + piece_type_size, 128 )
    #     self.relu2 = nn.ReLU()
    #     self.fc2 = nn.Linear(128, 100)
    #     self.fc3 = nn.Linear(100, num_actions)
        
        
    #     self.fc_layers = nn.Sequential(
    #         #nn.Dropout(p=0.25),
    #         self.fc1, 
    #         self.relu2, 
    #         #nn.Dropout(p=0.25), 
    #         self.fc2,
    #         self.relu2,
    #         #nn.Dropout(p=0.25),
    #         self.fc3
    #     )

        
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.conv1.weight)
        # torch.nn.init.xavier_uniform_(self.conv2.weight)
        # torch.nn.init.xavier_uniform_(self.conv3.weight)
        
        
        
        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     self.fc1,
        #     self.relu2,
        #     self.fc2

        # )
        
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
        
        cov_output_size = 64* BOARD_HEIGHT * BOARD_WIDTH       # current board dimensions, maybe change that.
        piece_type_size = NUMBER_OF_PIECES
        
        self.fc1 = nn.Linear(cov_output_size + piece_type_size, FC_HIDDEN_UNIT_SIZE )
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(FC_HIDDEN_UNIT_SIZE, num_actions)
        
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        
    # def forward(self, x, piece_type): 
    #     #logger.log(f"in forward shape: {x.shape}")
    #     if len(x.shape) == 3:
    #         x = x.unsqueeze(1)  # Add channel dimension if not present
    #         #logger.log(f"x after unsqueezing: {x.shape}")
            
            
    #     x = self.layers(x)
    #     x = nn.Flatten()(x)
        
    #     if len(piece_type.shape) == 1:
    #         piece_type = piece_type.unsqueeze(0)
    #     #logger.log(f"dimensions in forward: \nx: {x.shape}, piece_t: {piece_type}")
        
    #     combined = torch.cat([x, piece_type], dim=1)
    #     #logger.log(f"combined shape {combined.shape}")
        
    #     x = self.fc1(combined)
    #     x = self.relu2(x)
    #     x = self.fc2(x)
    #     return x
    
    # def forward(self, x, piece_type):
    #     if len(x.shape) == 3:
    #         x = x.unsqueeze(1)
            
    #     x = self.first_cnn_layers(x)
        
    #     x = self.column_collapse(x)
        
    #     x = self.post_collapse_layers(x)
        
    #     x = x.flatten(1)
        
    #     if len(piece_type.shape) == 1:
    #         piece_type = piece_type.unsqueeze(0)
            
    #     combined = torch.cat([x, piece_type], dim=1)
            
    #     x = self.fc_layers(combined)
        
    #     return x

    # def forward(self, x, piece_type):
    #     if len(x.shape) == 3:
    #         x = x.unsqueeze(1)
            
    #     x = self.conv_layers(x)
    #     x = x.flatten(1)  # Keep all spatial information
        
    #     if len(piece_type.shape) == 1:
    #         piece_type = piece_type.unsqueeze(0)
            
    #     combined = torch.cat([x, piece_type], dim=1)
    #     return self.fc_layers(combined)
        
    def forward(self, x, piece_type): 
        #logger.log(f"in forward shape: {x.shape}")
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if not present
            #logger.log(f"x after unsqueezing: {x.shape}")
            
            
        x = self.layers(x)
        x = nn.Flatten()(x)
        
        if len(piece_type.shape) == 1:
            piece_type = piece_type.unsqueeze(0)
        #logger.log(f"dimensions in forward: \nx: {x.shape}, piece_t: {piece_type}")
        
        combined = torch.cat([x, piece_type], dim=1)
        #logger.log(f"combined shape {combined.shape}")
        
        x = self.fc1(combined)
        x = self.relu2(x)
        x = self.fc2(x)
        return x