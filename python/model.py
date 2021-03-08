"""
File responsible for holding the model, uses efficientnet
"""
import torch
import torch.nn as nn
from torch.nn import functional as TF
from efficientnet_pytorch import EfficientNet
import BayesModel


class Classifier(nn.Module):
    """
    Class that holds and runs the efficientnet CNN
    """
    def __init__(self, image_size, output_size, class_weights, dropout=0.5, BBB=False):
        """
        init function sets the type of efficientnet and any extra layers
        :param dropout: rate for dropout
        """
        super(Classifier, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        self.drop_rate = dropout
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_size = output_size
        self.BBB = BBB
        self.class_weights = class_weights

        with torch.no_grad():

            temp_input = torch.zeros(1, 3, image_size, image_size)
            encoder_size = self.model.extract_features(temp_input).shape[1]

        # Initialises the classification head for generating predictions.
        self.efficient_net_output = nn.Linear(encoder_size, 512)

        if BBB:
            self.hidden_layer = BayesModel.BayesianLayer(512, 512)
            self.hidden_layer2 = BayesModel.BayesianLayer(512, 512)
        else:
            self.hidden_layer = nn.Linear(512, 512)
            self.hidden_layer2 = nn.Linear2(512, 512)
        self.output_layer = nn.Linear(512, output_size)

    def forward(self, input, labels=None, dropout=False):
        """
        Method for handling a forward pass though the network, applies dropout using nn.functional
        :param input: input batch to be processed
        :param dropout: bool for whether or not dropout should be applied
        :return: processed output
        """



        output = self.model.extract_features(input)
        output = self.pool(output)
        output = output.view(output.shape[0], -1)

        if self.training:
            output = TF.dropout(output, self.drop_rate)
            output = self.efficient_net_output(output)
            output = TF.dropout(output, self.drop_rate)
        else:
            output = self.efficient_net_output(output)


        if dropout:
            output = TF.relu(self.hidden_layer(output))
            output = TF.dropout(output, self.drop_rate)
            output = TF.relu(self.hidden_layer2(output))
            output = TF.dropout(output, self.drop_rate)

        elif self.BBB:

            if self.training:
                output = self.sample_elbo(output, labels)
            else:
                output = TF.relu(self.hidden_layer(output))
                output = TF.relu(self.hidden_layer2(output))
                output = self.output_layer(output)
            return output

        elif self.training:
            output = TF.relu(self.hidden_layer(output))
            output = TF.dropout(output, self.drop_rate)
            output = TF.relu(self.hidden_layer2(output))
            output = TF.dropout(output, self.drop_rate)

        else:
            output = self.hidden_layer(output)
            output = self.hidden_layer2(output)

        output = self.output_layer(output)
        return output

    def bayesian_sample(self, input):
        output = TF.relu(self.hidden_layer(input))
        output = TF.relu(self.hidden_layer2(output))
        output = self.output_layer(output)

        return output

    #Methods for BBB
    def log_prior(self):
        return self.hidden_layer.log_prior + self.hidden_layer2.log_prior

    def log_variational_posterior(self):
        return self.hidden_layer.log_variational_posterior + self.hidden_layer2.log_variational_posterior

    def sample_elbo(self, input, target, samples=10, n_classes=8):

        num_batches = 555
        batch_size = input.size()[0]

        outputs = torch.zeros(samples, batch_size, n_classes)
        log_priors = torch.zeros(samples)
        log_variational_posteriors = torch.zeros(samples)

        for i in range(samples):
            outputs[i] = self.bayesian_sample(input)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()

        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()

        negative_log_likelihood = TF.cross_entropy(outputs.mean(0), target, weight=self.class_weights, reduction='sum')

        KL_divergence = (log_variational_posterior - log_prior)
        loss = KL_divergence / num_batches + negative_log_likelihood


        #loss = loss/batch_size

        if torch.isnan(loss):
            print("nan loss detected")

        self.BBB_loss = loss

        return outputs.mean(0)


