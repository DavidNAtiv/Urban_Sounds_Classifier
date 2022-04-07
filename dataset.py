from torch.utils.data import Dataset, DataLoader
import librosa, librosa.display
import torch
import pandas as pd
import numpy as np

from clearml import Logger
import scipy.io.wavfile as wav
import io, time, os

# -----------------------------------------------
# # DataSet and DataLoader
#
# - implement a class to be loaded with torch dataloader
# - generate dataset for training / testing (training=True|False)
# - we keep folder number 10 for test (10%)
# -----------------------------------------------


class myDataset(Dataset):
    def __init__(self, sr,MAX_LENGTH , DATA_DIR, CSV_FILE, training, n_mfcc): #, device):
        #self.device = device
        self.sr = sr
        self.data = pd.read_csv(CSV_FILE)
        self.training = training
        self.DATA_DIR = DATA_DIR
        self.MAX_LENGTH = MAX_LENGTH
        self.n_mfcc = n_mfcc

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
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=1024, hop_length=512)
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

    #### returns random audio samples
    def samples(self):
        if self.training:
            #training dataset
            table_data = self.data.loc[self.data.fold != 10 ]
        else:
            # test dataset
            table_data = self.data.loc[self.data.fold == 10]

        #get file info
        filename, _, _, _, _, fold, label, _ = table_data.iloc[0,:]
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

        #y contains our audio data
        d = time.strftime("%Y,%m,%d,_%H,%M,%S")
        t = d.split(',')
        today = ''.join(t)

        rd = np.random.randint(0,1000)
        file = f"AudioSample_{rd}"
        path = os.path.join(os.getcwd(), './SND/')
        path = os.path.join(path, file)
        path = os.path.normpath(path)
        wav.write(path, sr, y)

        Logger.current_logger().report_media("Audio Sample", "audiosample", iteration=1, local_path=path)



#the simpliest data loader, so it doesnt need a class
#todo IMPROVEMENTS: K fold
def create_data_loader(train_data, batch_size, shuffle=True):
    train_dataloader = DataLoader(train_data, batch_size, shuffle, drop_last=True)
    return train_dataloader
