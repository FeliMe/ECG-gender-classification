import abc

import torch
import torch.nn as nn

import models.encoder as encoder
import models.decoder as decoder
import models.model_utils as model_utils


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".
        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.cpu(), path)


class GenderClassifier(BaseModel):
    def __init__(self, input_dim):
        super(GenderClassifier, self).__init__()
        self.is_autoencoder = False
        self.predict_gender = True

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(False),
            nn.Dropout(0.4),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        gender_pred = self.classifier(x)

        return -1, gender_pred


class MayoResNet(BaseModel):
    """
    Adapted from "An artificial intelligence-enabled ECG algorithm for the
    identification of patients with atrial fibrillation during sinus
    rhythm: a retrospective analysis of outcome prediction"
    """
    def __init__(self):
        super(MayoResNet, self).__init__()
        self.is_autoencoder = False
        self.predict_gender = False

        # input size: 1 x 2048

        self.encoder = nn.Sequential(
            model_utils.ResBlock(1, 16, 5, 2),   # 16 x 1024
            model_utils.ResBlock(16, 16, 3, 2),  # 16 x 512
            nn.Dropout(p=0.2),
            model_utils.ResBlock(16, 32, 3, 4),  # 32 x 128
            model_utils.ResBlock(32, 32, 3, 2),  # 32 x 64
            nn.Dropout(p=0.2),
            model_utils.ResBlock(32, 64, 3, 2),  # 64 x 32
            model_utils.ResBlock(64, 64, 3, 4),  # 64 x 8
            nn.Dropout(p=0.2),
        )

        self.out = nn.Linear(64 * 8, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)

        gender_pred = self.out(x)

        return -1, gender_pred


class MayoNet(BaseModel):
    """
    Adapted from: "Screening for cardiac contractile dysfunction using an
    artificial intelligence–enabled electrocardiogram"
    """
    def __init__(self):
        super(MayoNet, self).__init__()
        self.is_autoencoder = False
        self.predict_gender = True

        # input size: 1 x 2048

        self.encoder = nn.Sequential(
            model_utils.ConvBlock(1, 16, 5, 2),   # 16 x 1024
            model_utils.ConvBlock(16, 16, 5, 2),  # 16 x 512
            model_utils.ConvBlock(16, 32, 5, 4),  # 32 x 128
            model_utils.ConvBlock(32, 32, 3, 2),  # 32 x 64
            model_utils.ConvBlock(32, 64, 3, 2),  # 64 x 32
            model_utils.ConvBlock(64, 64, 3, 4),  # 64 x 8
        )

        self.linear = nn.Sequential(
            model_utils.FCBlock(64 * 8, 64, p=0.4),
            model_utils.FCBlock(64, 32, p=0.4)
        )

        self.out = nn.Linear(32, 2)

    def forward(self, x):
        z = self.encoder(x)

        x = z.view(z.shape[0], -1)
        x = self.linear(x)

        gender_pred = self.out(x)

        return -1, gender_pred


class ConvModel(BaseModel):
    """
    This model can be used as a classification model or to pre-train the encoder
    in an autoencoder style
    """
    def __init__(self, prediction, input_dim, dropout=0.0):
        super(ConvModel, self).__init__()
        self.is_autoencoder = False
        self.predict_gender = False

        self.encoder = encoder.ConvEncoder(input_dim, dropout)

        if 'gender' in prediction:
            self.predict_gender = True
            self.linear = model_utils.FCBlock(16 * 8, 32, p=0.4)
            self.gender_out = nn.Linear(32, 2)
        if 'autoencoder' in prediction:
            self.is_autoencoder = True
            self.decoder = decoder.ConvDecoder(input_dim)

    def forward(self, x):
        gender_pred = -1
        y = -1

        z = self.encoder(x)

        if self.predict_gender:
            x = z.view(z.shape[0], -1)
            x = self.linear(x)
            gender_pred = self.gender_out(x)
        if self.is_autoencoder:
            y = self.decoder(z)

        return y, gender_pred
