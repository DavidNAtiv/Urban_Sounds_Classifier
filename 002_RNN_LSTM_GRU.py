import os, json, time
import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import *
from model import *
from training import *

###############################3
from clearml import Task
task = Task.init(project_name="Urban Sound Classifier", task_name="my task")
################################33
#from torch.utils.tensorboard import SummaryWriter

#todo IMPROVEMENT: tensorboard
#writer = SummaryWriter('/home/root/python/UrbanSoundsClassif/runs/lstm')
###################33


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# # MAIN

#==============================================================================================
#===========================  MAIN FUNCTION  ==================================================
#==============================================================================================
if __name__ == "__main__":


    print(f"Welcome to the MLP Urban Sound Classifier -({round(time.time(),2)})")
    print("-------------------------------------\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',         type=float, required=False, help='Enter the learning rate',         default=5e-4)
    parser.add_argument('--epochs',     type=int,   required=False, help='Enter the number of epochs',      default=1)
    parser.add_argument('--batch',      type=int,   required=False, help='Enter the batch size',            default=32)
    parser.add_argument('--l2',         type=float, required=False, help='Enter the L2 regularisation rate',default=0.)
    parser.add_argument('--sr',         type=float, required=False, help='Enter the target sample rate',    default=22050)
    parser.add_argument('--maxlength',  type=int,   required=False, help='Enter the desired audio length',  default=4)
    parser.add_argument('--hidden',     type=int,   required=False, help='Enter the NN hidden size',        default=128)
    parser.add_argument('--layers',     type=int,   required=False, help='Enter the NN number of layers',   default=2)
    parser.add_argument('--dropout',    type=float,  required=False, help='Enter the dropout value [0;1]',  default=0.)
    #parser.add_argument('--disablecuda',type=bool,  required=False, help='Add to disable CUDA references',  default=False)
    parser.add_argument('--USdir',      type=str,   required=False, help='UrbanSound8K directory' )
    args = parser.parse_args()

    ##############
    #python main.py --dropout 0.2 --l2 1E-4 --USdir `pwd`/UrbanSound8K/
    ###############


    ### Parameters
    ################## audio parameters
    SAMPLE_RATE = args.sr
    MAX_LENGTH = args.maxlength

    ############### training parameters
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch

    LR = args.lr
    L2_LAMBDA = args.l2 #if args.l2 else 1E-4
    regularized = True if args.l2 else False

    ############### NN parameters
    hidden_size = args.hidden  # H_out (hyper param)
    num_layers = args.layers   # (hyper param)
    dropout = args.dropout

    nb_classes = 10 # output size
    ############3

    # ---------------------------------------------------------------
    # CUDA
    #Using GPU
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not args.disablecuda else 'cpu'
    #print(torch.cuda.get_device_name(0))

    # ---------------------------------------------------------------
    # Loading Data

    #setting directory
    ROOT_DIR = os.path.normpath(args.USdir) if args.USdir else os.path.normpath(os.path.join(os.getcwd(),'UrbanSound8K'))
    DATA_DIR = os.path.join(ROOT_DIR, 'audio')
    CSV_FILE = os.path.join(ROOT_DIR, 'metadata/UrbanSound8K.csv')

    # open and read csv file
    df = pd.read_csv(CSV_FILE)

    # create the labels list
    labels = pd.DataFrame(df.loc[:, ['classID', 'class']]).set_index('classID')
    labels = labels.drop_duplicates().sort_values('classID')
    print(labels)

    # ---------------------------------------------------------------
    # Prepare Dataset & DataLoader

    # create the data generator
    ds_train = myDataset(SAMPLE_RATE, MAX_LENGTH, DATA_DIR, CSV_FILE, True) #, device)
    dl_train = create_data_loader(ds_train, BATCH_SIZE)
    ds_test = myDataset(SAMPLE_RATE, MAX_LENGTH, DATA_DIR, CSV_FILE, False) #, device)
    dl_test = create_data_loader(ds_test, BATCH_SIZE, shuffle = False)

    # ---------------------------------------------------------------
    # Create Model

    # process input size
    seq_size = ds_train[0][0].size()[0] # L
    nb_features = ds_train[0][0].size()[1] # H_in
    input_size = nb_features

    # create the model
    model = RNN(input_size, hidden_size, BATCH_SIZE, num_layers, nb_classes, regularized, dropout) #, device).to(device)
    print(model)

    #for x,y in dl_train:
        #input = ds_train[0][0].reshape(-1,seq_size,nb_features)
        #break

    ##################
    # todo
    # writer.add_graph(model, x.float())
    # writer.close()
    ##################

    # ---------------------------------------------------------------
    # Training
    optimizer = optim.Adam(model.parameters(), lr=LR)  # , weight_decay=1E-5)
    criterion = nn.CrossEntropyLoss()

    #training

    loss,acc = train_multi_epoch(
        model=model,
        train_data=dl_train,
        test_data= dl_test,
        optimizer=optimizer,
        criterion=criterion,
        epochs=EPOCHS,
        L2_LAMBDA=L2_LAMBDA,
        batch_size=BATCH_SIZE)#,
        #device=device
    #)


    # ---------------------------------------------------------------
    # Save Model

    #torch.save(model.state_dict(), f'../RNN_{EPOCHS}.pth')

    # ---------------------------------------------------------------
    # Plot

    #plt.figure()
    plt.plot(loss,acc)
    plt.show()


    #state_dict = torch.load("../MLP_8.pth")
    #model.load_state_dict(state_dict)


    # ---------------------------------------------------------------
    # Evaluate Accuracy on

    #create the data generator

    #test(model, dl_test, device)


"""
================================= BENCH =================================
8 EPOCHS / BATCH 32
------------------

TEST SET :
---------
Current accuracy: 0.6658653846153846% on 25 test samples


VALIDATION :
----------

-- Episode 240 --> Loss = 0.2200358808040619 -- 34.8s
Total Time: 826.72s for 245 rounds

Validation test : Current accuracy: 0.98 on 50 test samples
Timer : 6712.51s.

"""