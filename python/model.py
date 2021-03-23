"""
File responsible for holding the model, uses efficientnet
"""
import torch
import torch.nn as nn
from torch.nn import functional as TF
from efficientnet_pytorch import EfficientNet
import BayesModel


class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)
        

class Classifier(nn.Module):
    """
    Class that holds and runs the efficientnet CNN
    """
    def __init__(self, image_size, output_size, class_weights, device, hidden_size=512, hidden_size2=512, hidden_size3=512, dropout=0.5, BBB=False):
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
        self.device = device
        #self.relu = torch.nn.LeakyReLU()
        #self.relu = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        
        print(f"1: {hidden_size}, 2: {hidden_size2}, 3: {hidden_size3}")

        with torch.no_grad():

            temp_input = torch.zeros(1, 3, image_size, image_size)
            encoder_size = self.model.extract_features(temp_input).shape[1]
        #print("\nEncoder size: " + str(encoder_size))
        # Initialises the classification head for generating predictions.
        if BBB:
            self.hidden_layer = BayesModel.BayesianLayer(encoder_size, hidden_size, device)
            self.hidden_layer2 = BayesModel.BayesianLayer(hidden_size, hidden_size2, device)
            #self.hidden_layer3 = BayesModel.BayesianLayer(hidden_size2, hidden_size3, device)

        else:
            self.hidden_layer = nn.Linear(encoder_size, hidden_size)
            self.hidden_layer2 = nn.Linear(hidden_size, hidden_size2)
            #self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size2)
        #self.bn3 = nn.BatchNorm1d(num_features=hidden_size3)
        
        #self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_layer = nn.Linear(hidden_size2, output_size)
        #self.output_layer = nn.Linear(hidden_size3, output_size)
        

    def forward(self, input, train_mc=False, labels=None, drop_samples=1, sample=False, drop_rate=None, dropout=False, samples=10):
        """
        Method for handling a forward pass though the network, applies dropout using nn.functional
        :param input: input batch to be processed
        :param dropout: bool for whether or not dropout should be applied
        :return: processed output
        """
        

        output = self.extract_efficientNet(input)
        
        """if train_mc:
            batch_size = 32
            batch_size = input.size()[0]
            outputs = torch.zeros(samples, batch_size, n_classes).to(self.device)
            
            for i in range(samples):
                outputs[i] = self.pass_through_layers(output, labels=labels, sample=sample, drop_rate=drop_rate, dropout=dropout)
            
            output = outputs.mean(0)
        
        else:"""
        
        output = self.pass_through_layers(output, labels=labels, sample=sample, drop_rate=drop_rate, drop_samples=drop_samples, dropout=dropout)

        return output

    def extract_efficientNet(self, input):
        output = self.model.extract_features(input)
        output = self.pool(output)
        output = output.view(output.shape[0], -1)

        return output

    def pass_through_layers(self, input, labels=None, sample=False, drop_rate=None, drop_samples=1, dropout=False):

        if drop_rate is None:
            drop_rate = self.drop_rate
        
        
        if self.BBB:
            # Don't bother calculating KL Divergence if we're not training
            if self.training or sample:
                output = self.sample_elbo(input, labels)
                return output
            else:
                output = self.relu(self.bn1(self.hidden_layer(input)))
                output = self.relu(self.bn2(self.hidden_layer2(output)))
                #output = self.relu(self.bn3(self.hidden_layer3(output)))
                
                #output = self.relu(self.hidden_layer(input))
                #output = self.relu(self.hidden_layer2(output))
                
                return self.output_layer(output)
        outputs = torch.zeros(drop_samples, input.size()[0], self.output_size).to(self.device)
        for i in range(0, drop_samples):
            if dropout:
                input = TF.dropout(input, drop_rate)

            #print("\nInput size: " + str(input.shape[1]))
            #print("\nLinear Size: " + str(self.hidden_layer.in_features))

            output = self.relu(self.bn1(self.hidden_layer(input)))
            #output = self.relu(self.hidden_layer(input))

            if dropout:
                output = TF.dropout(output, drop_rate)

            output = self.relu(self.bn2(self.hidden_layer2(output)))
            #output = self.relu(self.hidden_layer2(output))
            if dropout:
                output = TF.dropout(output, drop_rate)

            """output = self.relu(self.bn3(self.hidden_layer3(output)))
            #output = self.relu(self.hidden_layer2(output))
            if dropout:
                output = TF.dropout(output, drop_rate)"""

            outputs[i] = self.output_layer(output)
        return outputs.mean(0)
    
    
    #Methods for BBB
    def bayesian_sample(self, input):
        output = self.relu(self.bn1(self.hidden_layer(input)))
        output = self.relu(self.bn2(self.hidden_layer2(output)))
        #output = self.relu(self.bn3(self.hidden_layer3(output)))
        
        #output = self.relu(self.hidden_layer(input))
        #output = self.relu(self.hidden_layer2(output))
        
        output = self.output_layer(output)

        return output
    
    def log_prior(self):
        return self.hidden_layer.log_prior + self.hidden_layer2.log_prior# + self.hidden_layer3.log_prior

    def log_variational_posterior(self):
        return self.hidden_layer.log_variational_posterior + self.hidden_layer2.log_variational_posterior# + self.hidden_layer3.log_variational_posterior

    def sample_elbo(self, input, target, samples=10, n_classes=8):
        
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
    
        negative_log_likelihood = TF.cross_entropy(outputs.mean(0), target, weight=self.class_weights, reduction='sum')

        KL_divergence = (log_variational_posterior - log_prior)
        loss = KL_divergence / num_batches + negative_log_likelihood


        #loss = loss/batch_size

        if torch.isnan(loss):
            print("nan loss detected")

        self.BBB_loss = loss

        return outputs.mean(0)


