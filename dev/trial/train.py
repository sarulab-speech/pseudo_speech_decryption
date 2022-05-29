#/usr/bin/env python3
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import sys
import os
import librosa
import librosa.display
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from utils.UNet_model import UNet

from utils.Normalization import *
from utils.Wav2MFCC import Wav2Spec


gen_dir = "vctk_dev_gen/"
anon_dir = "vctk_dev_anon/"

paths = {
    "save_model_path" : "save_models/",
    "save_scaler_path" : "scaler/",
    "save_train_info" : "train_log/"
}

files = {
    "save_train_log" : "train.log",
    "save_model_name" : "best.pth"
    }

protocol = "dev.protocol.csv"
BATCHSIZE=64

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### model path ####
load_model = False
load_path = "./save_models/best.pth"


def get_scaler(path, df):
    scaler = MinMaxScaler()
    n_observe = 0
    for row in tqdm(range(len(df))):
        row = df.iloc[row]
        #read data
        _, audio = wavfile.read(os.path.join(path, row['filename'] + ".wav")) #[time, freq]
        signal_log, spec = Wav2Spec(audio)
        spec = torch.log(spec)
        
        #Normalize spectrogram to fixed lengths
        spec = SpecNorm(spec)
        
        #update scaler parameters
        feature_range = [0,1]
        data_max_ = np.nanmax(spec, axis=0)
        data_min_ = np.nanmin(spec, axis=0)
        
        if n_observe == 0:
            data_min = data_min_
            data_max = data_max_
            
        data_min = np.minimum(data_min_, data_min)
        data_max = np.maximum(data_max_, data_max)
        
        data_range = data_max - data_min
        scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(data_range, copy=True)
        min_ = feature_range[0] - data_min * scale_
        n_observe += 1
        
    #get final parameters of scaler
    scaler.min_ = min_
    scaler.data_min_ = data_min
    scaler.data_max_ = data_max
    scaler.data_range_ = data_range
    scaler.scale_ = scale_

    #save scaler parameters
    np.savetxt(os.path.join(scaler_path, "scaler-min_.txt"), scaler.min_, delimiter=",")
    np.savetxt(os.path.join(scaler_path, "scaler-data_min_.txt"), scaler.data_min_, delimiter=",")
    np.savetxt(os.path.join(scaler_path, "scaler-data_max_.txt"), scaler.data_max_, delimiter=",")
    np.savetxt(os.path.join(scaler_path, "scaler-data_range_.txt"), scaler.data_range_, delimiter=",")
    np.savetxt(os.path.join(scaler_path, "scaler-scale_.txt"), scaler.scale_, delimiter=",")
    print("calculated scaler parameters using {} data".format(n_observe))
    
    return scaler

if __name__ == '__main__':
    
    for path in paths:
        os.makedirs(paths[path], exist_ok = True)
            
    #read audio
    gen_audio = read_audio(gen_dir, protocol)
    anon_audio = read_audio(anon_dir, protocol)

    #calculate scaler
    scaler = get_scaler(anon_audio, paths["save_scaler_path"])
    
    #read audio to spectrogram
    data = SpecDataset(gen_dir, anon_dir, scaler)

    train_datasize = np.int(len(data)*0.8)
    train, val = Data.random_split(data, [train_datasize,len(data)-train_datasize])

    train_loader = DataLoader(train, batch_size=BATCHSIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCHSIZE, shuffle=False)

    best_loss = float("inf")
    model = UNet().to(device)
    
    if load_model:
        model.load_state_dict(torch.load(load_path))
        print('Successfully loaded model: {}'.format(load_path))

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(100,1000,100), gamma=0.8)

    losses = []

    ##### Training ######
    print("TRAINING")
    f = open(os.path.join(paths["save_train_info"], files["save_train_log"]), "a+")
    epoch = 10
    for epoch in range(epoch):
        train_losses = 0
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            true, anon = data[0].type(torch.FloatTensor).to(device), data[1].type(torch.FloatTensor).to(device)
            out = model(anon)
            loss = loss_function(out, true)
            loss.backward()
            optimizer.step()
            train_losses += loss.item()

        val_losses = 0
        model.eval()
        for data in val_loader:
            with torch.no_grad():
                true, anon, length = data[0].type(torch.FloatTensor).to(device), data[1].type(torch.FloatTensor).to(device), data[2]
                out = model(anon)
                loss = loss_function(out, true)
                val_losses += loss.item()
        scheduler.step()

        print("epoch {} \t train loss {} \t val loss {} \t lr {}".format(epoch, train_losses, val_losses, scheduler.get_last_lr()))
        f.write("epoch {} \t train loss {} \t val loss {} \t lr {}\n".format(epoch, train_losses, val_losses, scheduler.get_last_lr()))
        if float(val_losses) < best_loss:
            print("saving model")
            f.write("saving model\n")
            torch.save(model.state_dict(), os.path.join(paths['save_model_path'], files["save_model_name"]))
            best_loss = val_losses
    f.close()
