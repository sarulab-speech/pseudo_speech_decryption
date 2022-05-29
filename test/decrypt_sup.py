#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import librosa
import random
import soundfile as sf
from joblib import Parallel, delayed
from asteroid.models import BaseModel
from utils.voice_change_decrypt import *


anon_dir = "vctk_test_anon_sup/"

paths = {
    "save_mfcc_path" : "mfcc/",
    "save_audio_path" : "sup/"
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
BSS_model = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
fs = 16000


def SeparateAudio(directory, protocol, **params):
    df = pd.read_csv(protocol)
    for row in tqdm(range(len(df))):
        row = df.iloc[row]
        audio, _ = librosa.load(os.path.join(directory, row['filename'].split('_')[0], row['filename'] + ".wav"), fs)
        
        for k, v in params.items():
            mixture = audio.reshape(1, 1, audio.shape[0])
            output = np.squeeze(BSS_model.separate(mixture))
            
            audio = module[k](output[0], v) if k != "resamp" else module[k](output[0], resample_tmp, v)
            filename = os.path.join(paths["save_audio_path"], "audio1_{}".format(str(k)), row['filename'].split('_')[0], row['filename'] + ".wav")
            sf.write(filename, audio, fs)
            
            audio = module[k](output[1], v) if k != "resamp" else module[k](output[1], resample_tmp, v)
            filename = os.path.join(paths["save_audio_path"], "audio2_{}".format(str(k)), row['filename'].split('_')[0], row['filename'] + ".wav")
            sf.write(filename, audio, fs)
                

if __name__ == "__main__":
    for spk in speakers:
        for k, v in params.items():
            os.makedirs(os.path.join(paths["save_audio_path"], "audio1_{}".format(str(k)), spk), exist_ok = True)
            os.makedirs(os.path.join(paths["save_audio_path"], "audio2_{}".format(str(k)), spk), exist_ok = True)

    SeparateAudio(anon_dir, protocol, **params)
    
