from torch.utils.data import Dataset, DataLoader
import librosa, librosa.display
import torch
import pandas as pd
import numpy as np

# -----------------------------------------------
# # DataSet and DataLoader
#
# - implement a class to be loaded with torch dataloader
# - generate dataset for training / testing (training=True|False)
# - we keep folder number 10 for test (10%)
# -----------------------------------------------


class myDataset(Dataset):
    def __init__(self, sr,MAX_LENGTH , DATA_DIR, CSV_FILE, training): #, device):
        #self.device = device
        self.sr = sr
        self.data = pd.read_csv(CSV_FILE)
        self.training = training
        self.DATA_DIR = DATA_DIR
        self.MAX_LENGTH = MAX_LENGTH

    def __getitem__(self, i):
        if self.training:
            #training dataset
            table_data = self.data.loc[self.data.fold != 10 ]
        else:
            # test dataset
            table_data = self.data.loc[self.data.fold == 10]

        #get file info
        filename, _, _, _, _, fold, label, _ = table_data.iloc[i,:]
        label = int(label)
        #construct path and read file
        relative_path = self.DATA_DIR + '/fold' + str(fold) + '/' + filename
        #load file
        y, sr = librosa.load(relative_path)

        #down mix if necessary
        if sr != self.sr:
            y = librosa.resample(y, sr, self.sr)
        #mono
        if y.shape[0] > 1:
            y = librosa.to_mono(y)

        #check that the sample has the desired length
        #if not, padding
        length = len(y)
        if length < self.MAX_LENGTH * self.sr:
            blank_pad = np.zeros(self.MAX_LENGTH * self.sr - length)
            y = np.concatenate((y , blank_pad ))
        elif length > self.MAX_LENGTH * self.sr:
            y = y[:-(length - self.MAX_LENGTH * self.sr)]

        #mfcc
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=64, n_fft=1024, hop_length=512)
        mfcc = np.array(mfcc.T)

        #to tensor
        y = torch.as_tensor(mfcc,dtype=float) #.to(self.device)
        label = torch.as_tensor(label) #.to(self.device)
        return y, label

    def __len__(self):
        # training dataset
        if self.training :
            d = self.data.loc[self.data.fold != 10]
        else:
            d = self.data.loc[self.data.fold == 10]
        return len(d)

#the simpliest data loader, so it doesnt need a class
#todo IMPROVEMENTS: K fold
def create_data_loader(train_data, batch_size, shuffle=True):
    train_dataloader = DataLoader(train_data, batch_size, shuffle, drop_last=True)
    return train_dataloader
