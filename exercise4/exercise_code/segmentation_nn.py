"""SegmentationNN"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
    
        
        kernel_size = 3
        stride_conv=1
        weight_scale=0.001 
        pool=2
        stride_pool=2
        dropout=0.5
        channels = 3
        self.num_classes = num_classes

        self.pool= pool
        self.stride_pool = stride_pool
        padding = int((kernel_size - 1)/2)        
        self.conv1 = nn.Conv2d(512,128, 
            kernel_size, 
            stride=stride_conv,
            padding=0)

        self.conv2 = nn.Conv2d(128,num_classes, 
            7, 
            stride=stride_conv,
            padding=0)

        self.conv3 = nn.Conv2d(128,128, 
            kernel_size, 
            stride=stride_conv,
            padding=padding)

        self.conv4 = nn.Conv2d(128,128, 
            kernel_size, 
            stride=stride_conv,
            padding=padding)

        self.conv5 = nn.Conv2d(128,256, 
            7, 
            stride=stride_conv,
            padding=0)

        self.conv6 = nn.Conv2d(256,256, 
            1, 
            stride=stride_conv,
            padding=0)

        self.conv7 = nn.Conv2d(256,num_classes, 
            1, 
            stride=stride_conv,
            padding=0)
        
        # self.conv1.weight.data = weight_scale * self.conv1.weight.data
        # self.conv2.weight.data = weight_scale * self.conv2.weight.data
        # self.conv3.weight.data = weight_scale * self.conv3.weight.data
    

        self.drop = nn.Dropout(p=dropout)


        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################




    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        x = self.features(x)      
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.drop(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.drop(x)
        
        
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, self.pool, stride=self.stride_pool)
        # x = self.drop(x)
        
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, self.pool, stride=self.stride_pool)
    

        # # fully conv
        # x = self.conv5(x)
        # x = F.relu(x)
        # x = self.drop(x)
        # x = self.conv6(x)
        # x = F.relu(x)
        # x = self.drop(x)

        m = nn.Upsample(size=(240,240), mode='bilinear',align_corners=False)
        x = m(x)


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
