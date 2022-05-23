# Import required libraries
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import copy
from PIL import Image
from torch import device

# This file remains the same as the tutorial file due to this not needing any changes since
# it is simply loading and showing an image, no improvements or changes were needed here


# Resize the photo and convert it to a  tensor
loader = transforms.Compose([
    transforms.Resize(512),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


#Load in the image and return it after
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image

# Reconvert the image from tensor to image
unloader = transforms.ToPILImage()
plt.ion()

# Show the image to the user
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
