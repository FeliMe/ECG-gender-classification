import torch.nn as nn

import models.model_utils as model_utils


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.input_dim = input_dim

        self.upconv_1 = model_utils.UpConvBlock(16, 32, 3, 4)
        self.upconv_2 = model_utils.UpConvBlock(32, 32, 3, 2)
        self.upconv_3 = model_utils.UpConvBlock(32, 32, 3, 2)

        # Additional convolutions to handle sequences of 16 and 32 seconds
        if input_dim in [4096, 8192]:
            self.upconv_3_2 = model_utils.UpConvBlock(32, 32, 3, 2)
        if input_dim == 8192:
            self.upconv_3_3 = model_utils.UpConvBlock(32, 32, 3, 2)

        self.upconv_4 = model_utils.UpConvBlock(32, 16, 5, 4)
        self.upconv_5 = model_utils.UpConvBlock(16, 16, 5, 2)
        self.upsample_6 = nn.Upsample(scale_factor=2)
        self.conv_6 = nn.Conv1d(16, 1, 5, 1, padding=2)

    def forward(self, x):

        x = self.upconv_1(x)
        x = self.upconv_2(x)
        x = self.upconv_3(x)

        x = self.conv_3_2(x) if self.input_dim in [4096, 8192] else x
        x = self.conv_3_3(x) if self.input_dim == 8192 else x

        x = self.upconv_4(x)
        x = self.upconv_5(x)
        x = self.upsample_6(x)
        x = self.conv_6(x)

        return x
