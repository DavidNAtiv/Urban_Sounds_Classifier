import os, json, time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import *


# -----------------------------------------------------------
# # RNN/GRU Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers, nb_classes, regularized, dropout): #, device):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.device = device
        self.nb_classes = nb_classes
        self.regularized = regularized
        self.dropout = dropout

        #=====================================================================================
        #for simple RNN remplace GRU by RNN (!!!!!)
        #for LSTM we need to creat e a cell state c_0 (using exactly the same func as for h_0)
        #=====================================================================================

        # input (N    , L  , H_in      )
        # x  -> (batch, seq, input_size)
        self.rnn = nn.GRU(
                input_size = input_size,   #nb of mfcc
                hidden_size = self.hidden_size, #hyper param
                num_layers = self.num_layers,    #hyper param
                dropout = self.dropout,
                batch_first=True,
        )

        self.fc = nn.Linear(
             self.hidden_size,
             self.nb_classes
        )


    def forward(self, x):
        # H_0 (num layers, N, H_out)
        h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)#.to(self.device)

        # output ( N , L , H_out)
        out, _ = self.rnn(x, h_0)

        #we only want the final output, at the end of the seq : this ll be the prediction
        # that s to say ( N , H_out)
        #               (batch x hidden_size)
        # the size of our output is the size of the hidden state

        #from the seq column, we keep only the last element
        out = out[ : , -1 , :] #this gives the last line of the output

        out = self.fc(out)

        return out

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)#.to(self.device)

    def is_regularized(self):
        return self.regularized