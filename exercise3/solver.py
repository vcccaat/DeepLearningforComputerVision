from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        train_correct = 0
        val_correct = 0
        val_total = 0
        train_total = 0

        for epoch in range(num_epochs):
   
            for i, data in enumerate (train_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels) 
                y_predict = model(inputs)
                loss = self.loss_func (y_predict,labels)
                self.train_loss_history.append(loss)

                optim.zero_grad()
                loss.backward()
                optim.step()
                
                _, train_pred = torch.max(y_predict.data, 1)
                train_correct += (train_pred == labels).sum().item()
                train_total += labels.size(0)
            

                if i % log_nth == 0 :
                    print('[Iteration {}/{}] TRAIN loss: {:.3f}'.format(
                        i, i*epoch, loss.item()))    
 
            self.train_acc_history.append(train_correct/train_total)
                
            for i, data in enumerate (val_loader):
                val_input, val_label = data
                val_input, val_label = Variable(val_input), Variable(val_label)
                val_y_predict = model(val_input)
                val_loss = self.loss_func (val_y_predict, val_label)
                _, val_pred = torch.max(val_y_predict.data, 1)
                val_correct += (val_pred == val_label).sum().item()
                val_total += val_label.size(0)
                
            
            
            self.val_acc_history.append(val_correct/val_total)

            print('[Epoch {}/{}] TRAIN acc/loss: {:.3f}/{:.3f}'.format(
                epoch, num_epochs, train_correct/train_total,loss))
            print('[Epoch {}/{}] VAL acc/loss: {:.3f}/{:.3f}'.format(
                epoch, num_epochs, val_correct/val_total,val_loss))
            
                

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
