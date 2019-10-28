import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(x.shape[0], *self.size)


class ConvBlock(nn.Module):
    """
    1D Convolution followed by Batch Normalization, ReLU Activation and
    1D max pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, mp_factor):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 1,
                              padding=kernel_size // 2, bias=False)
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=mp_factor, stride=mp_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class UpConvBlock(nn.Module):
    """
    Nearest Neighbor Upsampling followed by a 1D Convolution,
    Batch Normalization and ReLU Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, mp_factor):
        super(UpConvBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=mp_factor, mode='nearest')
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 1,
                              padding=kernel_size // 2, bias=False)
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class FCBlock(nn.Module):
    """
    Fully Connected layer, followed by Batch Normalization, ReLU Activation
    and Dropout
    """
    def __init__(self, in_features, num_hiddens, p):
        super(FCBlock, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=num_hiddens,
                                bias=False)
        self.batchnorm = nn.BatchNorm1d(num_features=num_hiddens)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mp_factor=2):
        super(ResBlock, self).__init__()

        # Main path
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size // 2, stride=1, bias=False)
        self.relu1 = nn.ReLU()

        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2, stride=mp_factor,
                               bias=False)
        self.relu2 = nn.ReLU()

        # Residual path
        self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=mp_factor, stride=mp_factor)

    def forward(self, x):
        residual = x
        residual = self.conv_res(residual)
        residual = self.pool(residual)

        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)

        out = out + residual

        return out
