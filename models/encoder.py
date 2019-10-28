import torch.nn as nn
import torch.nn.functional as F

import models.model_utils as model_utils


class ConvEncoder(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        super(ConvEncoder, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim

        self.conv_1 = model_utils.ConvBlock(1, 16, 5, 2)
        self.conv_2 = model_utils.ConvBlock(16, 16, 5, 2)
        self.conv_3 = model_utils.ConvBlock(16, 32, 5, 4)
        self.conv_4 = model_utils.ConvBlock(32, 32, 3, 2)

        # Additional convolutions to handle sequences of 16 and 32 seconds
        if input_dim in [4096, 8192]:
            self.conv_4_2 = model_utils.ConvBlock(32, 32, 3, 2)
        if input_dim == 8192:
            self.conv_4_3 = model_utils.ConvBlock(32, 32, 3, 2)

        self.conv_5 = model_utils.ConvBlock(32, 32, 3, 2)
        self.conv_6 = nn.Conv1d(32, 16, 3, 1, padding=1)
        self.pool_6 = nn.MaxPool1d(4, 4)

    def forward(self, x):  # 1 x 2048
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = F.dropout(x, p=self.dropout) if self.dropout > 0 else x

        x = self.conv_3(x)
        x = self.conv_4(x)

        x = F.dropout(x, p=self.dropout) if self.dropout > 0 else x

        x = self.conv_4_2(x) if self.input_dim in [4096, 8192] else x
        x = self.conv_4_3(x) if self.input_dim == 8192 else x

        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.pool_6(x)

        x = F.dropout(x, p=self.dropout) if self.dropout > 0 else x

        return x
