import time, os, io

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa, librosa.display

import matplotlib.pyplot as plt
from PIL import Image
from clearml import Logger

# ------------------------------------------------------------
# # TRAINING
#
def train_single_epoch(model, train_data, optimizer, criterion, L2_LAMBDA, episods, log, current_epoch): #, device):
    bench_begin = time.time()
    loss_history = []
    begin_in_bench = time.time()

    for e, (X, y) in enumerate(train_data):
        #X, y = X.to(device), y.to(device)

        # input vector has size : torch.Size([N, L, H_in])
        # (batch, seq_len, feat)

        # backprop error
        optimizer.zero_grad()

        pred = model(X.float())
        loss = criterion(pred, y)

        if model.is_regularized():
            #L2 regularization
            l2_norm = 0
            for p in model.parameters() :
                l2_norm += p.pow(2.0).sum()

            loss += L2_LAMBDA * l2_norm

        loss.backward()
        optimizer.step()

        # [ N , classes]
        running_correct = 0

        running_correct += (torch.argmax(pred) == y).sum().item()
        #print(running_correct)
        #print(torch.argmax(pred))
        #exit()



        if e % log == 0 and e != 0:
            ## CLEARML : manually log accuracy
            Logger.current_logger().report_scalar("Loss", "loss", iteration=(current_epoch-1)*episods + e , value=loss)
            #############

            print(f"----> Loss = {loss.item()} -- {round(time.time() - begin_in_bench, 2)}s")
            begin_in_bench = time.time()
            loss_history.append(loss.item())


        if e >= episods:
            break
        print(f"-- Episod {e+1}/{episods} --")

    print(f"Total Time: {round(time.time() - bench_begin, 2)}s for {e} rounds\n")
    return loss_history

def evaluate_single_epoch(model, data, set, epoch, episod, labels): #, device):
    correct = 0
    c = 0
    with torch.no_grad():
        for X, y in data:
            #X, y = X.to(device), y.to(device)
            pred = model(X.float())
            for p in y:
                c += 1
                if y[p].item() == torch.argmax(pred[p]).item():
                    correct += 1
                if c >= set:
                    break
            if c >= set:
                break


    accuracy = correct / c
    print(f"Validation test : Current accuracy: {accuracy} on {c} test samples")

    ## CLEARML : manually log accuracy
    Logger.current_logger().report_scalar("Accuracy", "accuracy", iteration=epoch * episod, value=accuracy)

    #### debug info : gives some random audio to hear from the set of data used for valid
    #generate string w/current time
    d = time.strftime("%Y,%m,%d,_%H,%M,%S")
    t = d.split(',')
    today = ''.join(t)
    #nb of graphs
    n = 1
    for i, (X, y) in enumerate(data):
        if i > n:
            break
        fig, ax = plt.subplots(1, sharex=True, sharey=True)
        #random index
        index = np.random.randint(X.shape[0])
        #corresponding label
        lab = y.detach().numpy()[index]
        #set graph title
        title = f'{labels.iloc[lab]}'
        #convert back data to np
        mfcc_data = X.detach().numpy()
        mfcc_data = mfcc_data[index].T
        #plot
        img = librosa.display.specshow(mfcc_data, x_axis='time', ax=ax)
        fig.colorbar(img, ax=[ax])
        ax.set(title=title)
        #render (trying to get logged in Debug Samples
        # buf = io.BytesIO()
        # fig.savefig(buf)
        # buf.seek(0)
        # image = Image.open(buf)

        #save
        file = f"MFCC_{today}_{episod}x{epoch}_{i}"
        path = os.path.join(os.getcwd(), './IMG/')
        path = os.path.join(path, file)
        path = os.path.normpath(path)
        plt.savefig(path, format='png')
        Logger.current_logger().report_image("MFCC Sample", title, iteration=epoch * episod, image=Image.open(path))
        plt.close()
    #############


    return accuracy

def train_multi_epoch(model, labels, train_data, test_data, optimizer, criterion, epochs, episods, log, eval, L2_LAMBDA): #, device):
    loss_history = []
    accuracy_history = []
    timer = time.time()

    for e in range(epochs):
        print(f"---- Beginning epoch {e+1}/{epochs} ----")
        #training
        epoch_loss = train_single_epoch(model, train_data, optimizer, criterion, L2_LAMBDA, episods, log, current_epoch=e+1) #, device)

        #evaluation (on a single random batch)
        print("---- Evaluating ----")
        epoch_accuracy = evaluate_single_epoch(model, test_data, eval, e+1, episods, labels) #, device)

        # storing
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)

    print(f"Timer : {round(time.time() - timer, 2)}s.\n")
    return loss_history, accuracy_history

# ----------------------------------------------------------------
#
# # TESTING

def test(model, data): #, device):
    correct = 0
    c = 0
    print(f" --- Testing model on {len(data)} rounds ---")
    for i, (X, y) in enumerate(data):
        with torch.no_grad():
            #X, y = X.to(device), y.to(device)
            pred = model(X.float())
            for p in y:
                c += 1
                if y[p].item() == torch.argmax(pred[p]).item():
                    correct += 1
        # print(f"expect {labels.iloc[y.cpu()]} -- obtained: {labels.iloc[pred.cpu()]}")
    correct /= c
    print(f"Current accuracy: {correct}% on {i} test samples")
