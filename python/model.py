import torch
import torch.nn as nn
from torch.nn import functional as TF
from efficientnet_pytorch import EfficientNet



class Classifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(Classifier, self).__init__()
        self.model = EfficientNet.from_name("efficientnet-b0")
        self.drop_rate = dropout
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_layer = nn.Linear(1280, 9)

    def forward(self, input, dropout=False):

        output = self.model.extract_features(input)
        output = self.pool(output)
        output = output.view(output.shape[0], -1)

        if dropout:
            output = TF.dropout(output, self.drop_rate)

        output = self.output_layer(output)
        return TF.log_softmax(output, dim=1)
