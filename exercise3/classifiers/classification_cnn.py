"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################
        self.pool= pool
        self.stride_pool = stride_pool
        padding = int((kernel_size - 1)/2)        
        self.conv1 = nn.Conv2d(channels, 
            num_filters, 
            kernel_size, 
            stride=stride_conv,
            padding=padding)
        
        self.conv1.weight.data = weight_scale * self.conv1.weight.data
        
        # conv output size after padding
        output_size = int((height-kernel_size+2*padding)/stride_conv + 1)
        # after pooling
        pool_out_size = int((output_size-self.pool)/2 + 1)
        self.fc1 = nn.Linear(pool_out_size**2*num_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.drop = nn.Dropout(p=dropout)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################

        # cnn architecture: conv - relu - 2x2 max pool - fc - dropout - relu - fc
   
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=self.pool, stride=self.stride_pool)
        
        size = np.prod(x.size()[1:])
        x = x.view(-1,size)
        
        x = self.fc1(x)
        x = self.drop(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
    
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
