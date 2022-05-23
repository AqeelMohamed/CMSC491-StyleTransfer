# Import all required modules for this file
import torch
import torch.nn as nn
import torch.nn.functional as F


# Create a gram matrix
def gram_matrix(input):
    # Get the properties of the input image
    a, b, c, d = input.size()

    # Take in the features from the input image
    features = input.view(a, b, c * d)

    # Perform a batch matrix multiplication with the features and it's transposed elements
    # and then divide it by the length and width of the input image
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * d)


# Content Loss Class
class ContentLoss(nn.Module):
    # Utilize the parent class nn.Module super()
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    # Define what the loss function will be
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# Style Loss Class
class StyleLoss(nn.Module):
    # Utilize the parent class nn.Module super()
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    # Use a gram matrix wtih MSE for this loss function
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
