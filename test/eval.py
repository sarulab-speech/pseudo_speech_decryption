#!/usr/bin/env python3
import numpy as np
import copy
import os

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import soundfile as sf
from scipy.io import wavfile

from joblib import Parallel, delayed

from utils.Wav2MFCC import Wav2MFCC, Wav2Spec, Spec2MFCC
from utils.UNet_model import UNet

from utils.Normalization import *


gen_dir = "vctk_test_gen/"
anon_dir = "vctk_test_anon/"

paths = {
    "load_model_path" : "../dev/save_models/",
    "load_scaler_path" : "../dev/scaler/",
    "save_mfcc_path" : "mfcc/"
}

files = {
    "save_train_log" : "train.log",
    "load_model_name" : "best.pth",
    "save_mfcc_file" : "mfcc.txt"
    }

resample_tmp = "resample_tmp"
protocol = "test.protocol.csv-5"
BATCHSIZE=64

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_audio(directory, protocol):
    audio_list = []
    df = pd.read_csv(protocol)
    for row in tqdm(range(len(df))):
        row = df.iloc[row]
        _, audio = wavfile.read(os.path.join(directory, row['filename'].split('_')[0], row['filename'] + ".wav"))
        audio_list.append(audio)
    return audio_list

def get_scaler(scaler_path):
    scaler = MinMaxScaler()
    scaler.min_ = np.loadtxt(os.path.join(scaler_path, "scaler-min_.txt"))
    scaler.datamin_ = np.loadtxt(os.path.join(scaler_path, "scaler-data_min_.txt"))
    scaler.data_max_ = np.loadtxt(os.path.join(scaler_path, "scaler-data_max_.txt"))
    scaler.data_range_ = np.loadtxt(os.path.join(scaler_path, "scaler-data_range_.txt"))
    scaler.scale_ = np.loadtxt(os.path.join(scaler_path, "scaler-scale_.txt"))

    return scaler


class SpecDataset(Dataset):
    def __init__(self, gen_list, anon_list, scaler):
        self.gen_list = gen_list
        self.anon_list = anon_list
        self.scaler = scaler
        self.gen_data = []
        self.anon_data = []
        self.signal_log = []
        self.length = []

        count = 0
        for audio in self.gen_list:
            signal_log, spec = Wav2Spec(audio)
            spec = torch.log(spec)

            #Normalize and scale spectrogram
            spec = self.scaler.transform(SpecNorm(spec))
            self.gen_data.append(np.expand_dims(spec,axis=0))

        for audio in self.anon_list:
            signal_log, spec = Wav2Spec(audio)
            spec = torch.log(spec)
            
            self.length.append(spec.shape[0])

            signal_log = SignalNorm(signal_log)
            self.signal_log.append(signal_log)

            #Normalize and scale spectrogram
            spec = self.scaler.transform(SpecNorm(spec))
            self.anon_data.append(np.expand_dims(spec,axis=0))

    def __len__(self):
        return len(self.gen_data)

    def __getitem__(self, idx):
        return self.gen_data[idx], self.anon_data[idx], self.signal_log[idx], self.length[idx]


class output_mfcc(Dataset):
    def __init__(self, output, true, sig_log, length, scaler):
        self.spec_rec = output.squeeze().cpu()
        self.true = true.squeeze().cpu()
        self.signal = sig_log
        self.length = length
        self.scaler = scaler
        self.mfcc = []

        for idx in tqdm(range(len(self.spec_rec))):
            spec_rec = self.spec_rec[idx]
            true = self.true[idx]
            signal_log = torch.Tensor(self.signal[idx])
            length = self.length[idx]

            signal_log = signal_log[:length]

            spec_rec -= scaler.min_
            spec_rec /= scaler.scale_
            spec_rec = spec_rec[:length, :]
            spec_rec = np.pad(spec_rec, ((0,0), (0,1)))
            spec_rec = np.exp(spec_rec)
            spec_rec = torch.Tensor(spec_rec)
            mfcc = Spec2MFCC(spec_rec, signal_log).numpy()

            self.mfcc.append(mfcc)
            
    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, idx):
        return self.mfcc[idx]


if __name__ == "__main__":

    if not os.path.exists(paths["save_mfcc_path"]):
            os.makedirs(paths["save_mfcc_path"])
        
    ### give trial list and will output anon audio to directory ###
    trial_list = pd.read_csv(protocol)
    print("DONE : reading trial list")
    gen_audio = read_audio(gen_dir, protocol)
    anon_audio = read_audio(anon_dir, protocol)
    print("DONE : reading audios")
    scaler = get_scaler(paths["load_scaler_path"])
    print("DONE : loaded scaler info")
    eval_data = SpecDataset(gen_audio, anon_audio, scaler)
    print("DONE : reading anonymized audio data")
    eval_loader = DataLoader(eval_data, batch_size=BATCHSIZE, shuffle=False)
    print("DONE : loading anonymized data to dataloader")

    model = UNet().to(device)
    model.load_state_dict(torch.load(os.path.join(paths["load_model_path"], files["load_model_name"]), map_location = 'cuda:0'))
    print('Loaded model: {}'.format(os.path.join(paths["load_model_path"], files["load_model_name"])))

    loss_function = nn.MSELoss()
    losses = []
    eval_losses = 0

    model.eval()
    mfcc_file = os.path.join(paths["save_mfcc_path"], files["save_mfcc_file"])
    if os.path.exists(mfcc_file):
        os.remove(mfcc_file)
    f_mfcc = open(mfcc_file, 'a+')
    count = 0
    print("EVALUATING")
    for data in eval_loader: 
        with torch.no_grad():
            true, anon = data[0].type(torch.FloatTensor).to(device), data[1].type(torch.FloatTensor).to(device)
            signal_log, spec_length = data[2], data[3]
            out = model(anon)
            loss = loss_function(out, true)
            eval_losses += loss.item()
            
            ##### save as mfcc ######
            mfcc = output_mfcc(out, true, signal_log, spec_length, scaler)
            for i in range(len(mfcc)):
                f_mfcc.write("{} [\n".format(trial_list["filename"][count]))
                data = mfcc[i]
                for row in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        f_mfcc.write(" {}".format(data[row, col]))
                    if row == data.shape[0] - 1 and i != len(mfcc)-1:
                        f_mfcc.write(" ]\n")
                    elif row == data.shape[0] - 1 and i == len(mfcc)-1:
                        f_mfcc.write(" ]")
                    else:
                        f_mfcc.write("\n")
                count += 1
                
    print("evaluation loss {}".format(eval_losses))
    f_mfcc.close
