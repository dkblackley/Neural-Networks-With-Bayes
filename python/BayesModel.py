
import math
import torch
import torch.nn as nn
from torch.nn import functional as TF

# Credit to https://www.nitarshan.com/bayes-by-backprop/, Tutorial was followed and adapted to suit my specific model
class GaussianDistribution():

    def __init__(self, mu, rho, device):
        self.mu = mu
        self.rho = rho
        self.device = device
        self.normal = torch.distributions.Normal(0, 1, validate_args=True)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample_distribution(self):
        """
        Reparameterization trick
        :return: Reparameterized Gaussian
        """
        e = self.normal.sample(self.rho.size()).to(self.device)
        return self.mu + self.sigma * e

    def log_prob(self, input):
        """
        Makes use of our scale and slab prioir
        :param input:
        :return:
        """
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class ScaleMixtureGaussian():
    def __init__(self, pi, sigma1, sigma2, device):
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.device = device
        self.gaussian1 = torch.distributions.Normal(torch.tensor(0).to(device), sigma1, validate_args=True)
        self.gaussian2 = torch.distributions.Normal(torch.tensor(0).to(device), sigma2, validate_args=True)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()

class BayesianLayer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = GaussianDistribution(self.weight_mu, self.weight_rho, device)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = GaussianDistribution(self.bias_mu, self.bias_rho, device)

        # Prior distributions
        SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(device)
        SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(device)
        PI = torch.tensor(0.5).to(device)

        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2, device)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2, device)
        self.log_prior = 0

        self.log_variational_posterior = 0

    def forward(self, input):

        weight = self.weight.sample_distribution()
        bias = self.bias.sample_distribution()

        self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
        self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)

        return TF.linear(input, weight, bias)

