from torch import nn
from simpleLogger import SimpleLogger
from config import *

import torch

logger = SimpleLogger()


class DB_CNN(nn.Module):
    def __init__(self,num_actions: int, simple_cnn=False):      # sinmple cnn is currently only placeholder 
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5, padding=2),
            #nn.BatchNorm2d(64), 
            nn.ReLU()
        )
        self.piece_embed = nn.Embedding(NUMBER_OF_PIECES, 16)
        self.piece_fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU()
        )
        # Feature processing
        self.fc = nn.Sequential(
            nn.Linear(64*(BOARD_HEIGHT//2)*(BOARD_WIDTH//2)*2 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x, piece_type, column_features): 

        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # ensure x has shape [batch, 1, H, W]
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_conv = torch.cat([x1.flatten(1), x2.flatten(1)], dim=1)

        piece_type_indices = torch.argmax(piece_type, dim=1).long()

        embedded_piece = self.piece_embed(piece_type_indices)
        embedded_piece = self.piece_fc(embedded_piece)

        combined = torch.cat([x_conv, embedded_piece], dim=1)

        result = self.fc(combined)
        return result

class CNN(nn.Module):
    def __init__(self, num_actions: int, simple_cnn: bool): 
        super().__init__()
        self.simple_cnn = simple_cnn
        if simple_cnn:
            
            self.conv1 = nn.Conv2d(1,32, 3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

            self.pool = nn.MaxPool2d(2)
            
            self.layers = nn.Sequential(
                self.conv1,
                self.relu,
                self.conv2,
                self.relu,
                self.conv3,
                self.relu,
                #self.pool
            )
            
            # using pooling use this.
            # conv_output_height = BOARD_HEIGHT // 2
            # conv_output_width = BOARD_WIDTH // 2
            
            conv_output_height = BOARD_HEIGHT
            conv_output_width = BOARD_WIDTH 
            cov_output_size = 64 * conv_output_height * conv_output_width 
            column_feature_size = BOARD_WIDTH * 2      # as we have two column features: height, holes
            #cov_output_size = 64* BOARD_HEIGHT * BOARD_WIDTH       # current board dimensions, maybe change that.
            piece_type_size = NUMBER_OF_PIECES
            
            #process the piece type (a one-hot vector of length NUMBER_OF_PIECES)
            # with its own fc
            self.piece_embed = nn.Embedding(NUMBER_OF_PIECES, 16)
            self.piece_fc = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU()
            )

            self.column_fc = nn.Sequential(
                nn.Linear(BOARD_WIDTH*2, 32), #2 featuires per column
                nn.ReLU()
            )
        #     self.column_net = nn.Sequential(
        #     nn.Conv1d(1, 16, kernel_size=3, padding=1),  # Process column neighborhoods
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(16*BOARD_WIDTH, 32)
        # )
            
            self.fc1 = nn.Linear(cov_output_size + 32 + 32, FC_HIDDEN_UNIT_SIZE )
            self.relu2 = nn.ReLU()
            self.fc2 = nn.Linear(FC_HIDDEN_UNIT_SIZE, num_actions)
            
            for layer in [self.conv1, self.conv2, self.conv3]:
                nn.init.xavier_uniform_(layer.weight)
            for layer in [self.fc1, self.fc2]:
                nn.init.xavier_uniform_(layer.weight)
        
        else:  
            self.conv1 = nn.Conv2d(1,32, 3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            
            self.first_cnn_layers = nn.Sequential(
                self.conv1,
                #nn.BatchNorm2d(32),
                self.relu,
                self.conv2,
                #nn.BatchNorm2d(32),
                self.relu,
                self.conv3,
                #nn.BatchNorm2d(64),
                self.relu,
            )
            
            self.column_collapse = nn.AdaptiveAvgPool2d((BOARD_WIDTH, 1))
            
            self.post_collapse_layers = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                #nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 1),  # 1x1 convolution
                #nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                #nn.BatchNorm2d(128),
                nn.ReLU(),
            )

            self.piece_fc = nn.Sequential(
                nn.Linear(NUMBER_OF_PIECES, 32),
                nn.ReLU()
            )
            
            cov_output_size = 128 * BOARD_WIDTH # *  BOARD_HEIGHT       # current board dimensions, maybe change that.
            piece_type_size = NUMBER_OF_PIECES
            
            self.fc1 = nn.Linear(cov_output_size + 32, 128 )
            self.relu2 = nn.ReLU()
            self.fc2 = nn.Linear(128, 100)
            self.fc3 = nn.Linear(100, num_actions)
            
            
            self.fc_layers = nn.Sequential(
                #nn.Dropout(p=0.25),
                self.fc1, 
                self.relu2, 
                #nn.Dropout(p=0.25), 
                self.fc2,
                self.relu2,
                #nn.Dropout(p=0.25),
                self.fc3
            )

            
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.xavier_uniform_(self.conv1.weight)
            torch.nn.init.xavier_uniform_(self.conv2.weight)
            torch.nn.init.xavier_uniform_(self.conv3.weight)
            
            
            
            self.fc = nn.Sequential(
                nn.Flatten(),
                self.fc1,
                self.relu2,
                self.fc2

            )
        

    def forward(self, x, piece_type, column_features): 
        if self.simple_cnn:
            if len(x.shape) == 3:
                x = x.unsqueeze(1)  # ensure x has shape [batch, 1, H, W]

                
                
            x = self.layers(x)
            x = torch.flatten(x, start_dim=1)
            
            if len(piece_type.shape) == 1:
                piece_type = piece_type.unsqueeze(0)
            
            piece_type_indices = torch.argmax(piece_type, dim=1).long()
        
            pieces = self.piece_embed(piece_type_indices)  # Now 2D: [batch, 16]
            pieces = self.piece_fc(pieces)  # [batch, 32]

            column_features = column_features.view(-1, BOARD_WIDTH*2)
            column_features = self.column_fc(column_features)

        
            #column_features = torch.flatten(column_features, start_dim=1)
            
            
            combined = torch.cat([x, pieces, column_features], dim=1)
            #logger.log(f"combined shape {combined.shape}")
            
            x = self.fc1(combined)
            x = self.relu2(x)
            x = self.fc2(x)
            return x
        
        else: 
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
                
            x = self.first_cnn_layers(x)
            
            x = self.column_collapse(x)
            
            x = self.post_collapse_layers(x)
            
            x = x.flatten(1)
            
            if len(piece_type.shape) == 1:
                piece_type = piece_type.unsqueeze(0)

            pieces_features = self.piece_fc(piece_type)
                
            combined = torch.cat([x, pieces_features], dim=1)
                
            x = self.fc_layers(combined)
            
            return x
