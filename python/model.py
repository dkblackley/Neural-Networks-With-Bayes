"""
File responsible for holding the model, uses efficientnet
"""
import torch
import torch.nn as nn
from torch.nn import functional as TF
from efficientnet_pytorch import EfficientNet
import BBB


class Classifier(nn.Module):
    """
    Class that holds and runs the efficientnet CNN
    """
    def __init__(self, image_size, output_size, dropout=0.5, BBB=False):
        """
        init function sets the type of efficientnet and any extra layers
        :param dropout: rate for dropout
        """
        super(Classifier, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        self.drop_rate = dropout
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_size = output_size

        with torch.no_grad():

            temp_input = torch.zeros(1, 3, image_size, image_size)
            encoder_size = self.model.extract_features(temp_input).shape[1]

        # Initialises the classification head for generating predictions.
        self.efficient_net_output = nn.Linear(encoder_size, 512)

        if BBB:
            self.hidden_layer = BBB.BayesianLayer(512, 512)
        else:
            self.hidden_layer = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, output_size)

    def forward(self, input, dropout=False, sample=False):
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
            output = self.efficient_net_output(output)
            output = TF.dropout(output, self.drop_rate)

            if BBB:
                output = self.hidden_layer(output, sample)
            else:
                output = self.hidden_layer(output)

            output = TF.dropout(output, self.drop_rate)
        else:
            output = self.efficient_net_output(output)

            if BBB:
                output = self.hidden_layer(output, sample)
            else:
                output = self.hidden_layer(output)

        output = self.output_layer(output)
        return output

    #Methods for BBB
    def log_prior(self):
        return self.hidden_layer.log_prior

    def log_varaitional_posterior(self):
        return self.hidden_layer.log_varational_posterior

    def sample_elbo(self, input, target, batch_size, n_classes, num_batches, samples=SAMPLES):

        outputs = torch.zeros(samples, batch_size, n_classes)
        log_priors = torch.zeros(samples)
        log_variational_posteriors = torch.zeros(samples)

        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()

        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = TF.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior) / num_batches + negative_log_likelihood
        return loss


