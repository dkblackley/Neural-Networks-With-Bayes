
import math
import torch
import torch.nn as nn

# Credit to https://www.nitarshan.com/bayes-by-backprop/, Used to create code and follow through the steps of BBB paper
class GaussianDistribution(object):

    def __init__(self, mu, rho):
        super.__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample_distribution(self):
        e = self.normal.sample(self.rho.size())

        return self.mu + self.sigma * e

    def log_probability(self, input):
        return  (-math.log(math.sqrt(2 * math.pi))
                 - torch.log(self.sigma)
                 - ((input - self.mu ** 2) / (2 * self.sigma ** 2)).sum())

class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()

class BayesianLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = GaussianDistribution(self.weight_mu, self.weight_rho)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = GaussianDistribution(self.bias_mu, self.bias_rho)

        # Prior distributions
        SIGMA_1 = 0.8
        SIGMA_2 = 0.5
        self.weight_prior = ScaleMixtureGaussian(math.pi, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(math.pi, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calc_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample_distribution()
            bias = self.bias.sample_distribution()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calc_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_probability(weight) + self.bias.log_probability(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)
