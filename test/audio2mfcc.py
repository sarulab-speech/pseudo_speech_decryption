#!/usr/bin/env python3

import numpy as np
import os
from tqdm import tqdm
import librosa
import random
import pandas as pd
import soundfile as sf
from utils.Wav2MFCC import Wav2MFCC, Wav2Spec, Spec2MFCC

anon_dir = "sup/1-1/"

paths = {
    "save_mfcc_path" : "mfcc/"
}

files = {
    "save_mfcc_file" : "mfcc.txt"
    }

protocol = "test.protocol.csv"
fs = 16000


def get_mfcc(df):
    mfcc_list = []
    for row in tqdm(range(len(df))):
        row = df.iloc[row]
        audio, _ = librosa.load(os.path.join(anon_dir, row['filename'].split('_')[0], row['filename'] + ".wav"), fs)
        _, feature = Wav2MFCC(audio)
        mfcc_list.append(feature)
    return mfcc_list


if __name__ == "__main__":
    os.makedirs(paths["save_mfcc_path"], exist_ok = True)
    
    ### read anonymized audio directory and output mfcc
    trial_list = pd.read_csv(protocol)
    print("DONE : reading trial list")
    mfcc = get_mfcc(trial_list)
    print("DONE : audio data to MFCC")
    mfcc_file = os.path.join(paths['save_mfcc_path'], files['save_mfcc_file'])
    if os.path.exists(mfcc_file):
        os.remove(mfcc_file)
    f_mfcc = open(mfcc_file, 'a+')
    count = 0
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
    f_mfcc.close
