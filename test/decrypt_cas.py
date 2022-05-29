#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import librosa
import random
from joblib import Parallel, delayed
import soundfile as sf
from utils.voice_change_decrypt import *

anon_dir = "vctk_test_anon/"

paths = {
    "save_mfcc_path" : "mfcc/",
    "save_audio_path" : "cas/"
}

params = {
    "vtln" : 0.1,
    "resamp" : 0.8
    }

module = {
    "vtln" : vtln,
    "resamp" : resampling,
    "mcadams" : vp_baseline2,
    "modspec" : modspec_smoothing
    }

speakers = ["p226", "p227", "p232", "p243", "p254", "p256", "p258", "p259",
           "p270", "p273", "p274", "p278", "p279", "p286", "p287"]

protocol = "test.protocol.csv"
resample_tmp = "resample_tmp"
fs = 16000


def SeparateAudio(directory, protocol, **params):
    df = pd.read_csv(protocol)
    for row in tqdm(range(len(df))):
        row = df.iloc[row]
        audio, _ = librosa.load(os.path.join(directory, row['filename'].split('_')[0], row['filename'] + ".wav"), fs)

        for k, v in params.items():
            audio = module[k](audio, v) if k != "resamp" else module[k](audio, resample_tmp, v)

        filename = os.path.join(paths["save_audio_path"], row['filename'].split('_')[0], row['filename'] + ".wav")
        sf.write(filename, audio, fs)


if __name__ == "__main__":
    for spk in speakers:
            os.makedirs(os.path.join(paths["save_audio_path"], spk), exist_ok = True)
    
    SeparateAudio(anon_dir, protocol, **params)
    
