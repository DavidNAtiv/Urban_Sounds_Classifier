# Urban_Sounds_Classifier
Basic sound classifier using pytorch RNN with GRU or LSTM

This a basic demo of a sound classifier. The idea is mainly to train using Reccurent NN with pytorch, and to benchmark the accuracy and training time 
differences between GRU and LSTM.

I use a Docker image to be able to use my AMD GPU with ROC and pytorch. The latter lib is thus customized, and may not work with torchaudio. Therefore 
I used librosa lib here to preprocess the audio (normalization, MFCC extraction)

Improvements :
--------------
  - better logs, eventually using Tensorboard or any similar tool (Clear ML ?)
  - try to tweak hyper parameters to improve scores
  - add a K fold option for training
  - try the same algo w/ lightning (for training purpose)
