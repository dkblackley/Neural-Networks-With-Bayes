"""
File responsible for holding the model, uses efficientnet
"""
import torch
import torch.nn as nn
from torch.nn import functional as TF
from efficientnet_pytorch import EfficientNet
import BayesModel


class OutputHook(list):
    """
    Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)
        

class Classifier(nn.Module):
    """
    Class that holds and runs the efficientnet CNN
    """
    def __init__(self, image_size, output_size, class_weights, device, hidden_size=512, dropout=0.5, BBB=False):
        """
        Initialises network parameters
        :param image_size: Input image size, used to calculate output of efficient net layer
        :param output_size: number of classes to classify
        :param class_weights: the weights assigned to each class, used for BBB
        :param device: cpu or gpu, used for BBB
        :param hidden_size: size of first hidden layer
        :param dropout: Drop rate
        :param BBB: Whether or not to make layers Bayesian
        """
        super(Classifier, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        self.drop_rate = dropout
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_size = output_size
        self.BBB = BBB
        self.class_weights = class_weights
        self.device = device
        self.relu = torch.nn.ReLU()
        
        print(f"Hidden layer size: {hidden_size}")

        with torch.no_grad():

            temp_input = torch.zeros(1, 3, image_size, image_size)
            encoder_size = self.model.extract_features(temp_input).shape[1]

        # Initialises the classification head for generating predictions.
        if BBB:
            self.hidden_layer = BayesModel.BayesianLayer(encoder_size, hidden_size, device)
        else:
            self.hidden_layer = nn.Linear(encoder_size, hidden_size)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input, samples=1, sample=False, drop_rate=None, dropout=False):
        """
        Extracts efficient Net output then passes it through our other layers
        :param input: input Image batch
        :param samples: number of forward passes to run
        :param sample: Whether or not to sample the ELBO in BBB
        :param drop_rate: drop rate for dropout
        :param dropout: whether or not to apply dropout
        :return: the output of our network
        """

        output = self.extract_efficientNet(input)
        output = self.pass_through_layers(output, sample=sample,
                                          drop_rate=drop_rate, samples=samples, dropout=dropout)
        return output

    def extract_efficientNet(self, input):
        output = self.model.extract_features(input)
        output = self.pool(output)
        output = output.view(output.shape[0], -1)

        return output

    def pass_through_layers(self, input, sample=False, drop_rate=None, samples=1, dropout=False):
        """
        Run the output of efficient net through our layers
        :param input: Input image batch
        :param sample: Whether we should calculate BBB loss
        :param drop_rate: drop rate for dropout
        :param samples: number of samples to run
        :param dropout: whether or not to apply dropout
        :return: the networks classification batch
        """

        if drop_rate is None:
            drop_rate = self.drop_rate

        if self.BBB:
            # Don't bother calculating KL Divergence if we're not training or unless we ask
            if self.training or sample:
                output = self.sample_elbo(input, samples=samples)
                return output
            else:
                output = self.relu(self.bn1(self.hidden_layer(input)))
                
                return self.output_layer(output)
        outputs = torch.zeros(samples, input.size()[0], self.output_size).to(self.device)
        for i in range(0, samples):
            if dropout:
                input = TF.dropout(input, drop_rate)

            output = self.relu(self.bn1(self.hidden_layer(input)))

            if dropout:
                output = TF.dropout(output, drop_rate)

            outputs[i] = self.output_layer(output)
        return outputs.mean(0)

    # Methods for BbB
    def bayesian_sample(self, input):
        output = self.relu(self.bn1(self.hidden_layer(input)))
        output = self.output_layer(output)

        return output
    
    def log_prior(self):
        return self.hidden_layer.log_prior

    def log_variational_posterior(self):
        return self.hidden_layer.log_variational_posterior

    def sample_elbo(self, input, samples=1, n_classes=8):
        """
        Samples from the evidence lower bound
        :param input: Input image batch
        :param samples: number of samples to run across the Bayesian Layers
        :param n_classes: number of output classes
        :return: the network's classification batch
        """
        num_batches = 555
        batch_size = input.size()[0]

        outputs = torch.zeros(samples, batch_size, n_classes).to(self.device)
        log_priors = torch.zeros(samples).to(self.device)
        log_variational_posteriors = torch.zeros(samples).to(self.device)

        for i in range(samples):
            outputs[i] = self.bayesian_sample(input)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()

        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()

        KL_divergence = (log_variational_posterior - log_prior)
        loss = KL_divergence / num_batches

        self.BBB_loss = loss
        return outputs.mean(0)


