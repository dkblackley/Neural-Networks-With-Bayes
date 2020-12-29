"""
File responsible for holding the model, uses efficientnet
"""
import torch.nn as nn
from torch.nn import functional as TF
from efficientnet_pytorch import EfficientNet


class Classifier(nn.Module):
    """
    Class that holds and runs the efficientnet CNN
    """
    def __init__(self, dropout=0.5):
        """
        init function sets the type of efficientnet and any extra layers
        :param dropout: rate for dropout
        """
        super(Classifier, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        self.drop_rate = dropout
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_layer = nn.Linear(1280, 8)

    def forward(self, input, dropout=False):
        """
        Method for handling a forward pass though the network, applies dropout using nn.functional
        :param input: input batch to be processed
        :param dropout: bool for whether or not dropout should be applied
        :return: processed output
        """
        output = self.model.extract_features(input)
        output = self.pool(output)
        output = output.view(output.shape[0], -1)

        if dropout:
            output = TF.dropout(output, self.drop_rate)

        output = self.output_layer(output)
        return output
