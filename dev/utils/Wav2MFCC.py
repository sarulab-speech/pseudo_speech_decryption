#!/usr/bin/env python3

from typing import Tuple

import math
import numpy as np
import torch
from torch import Tensor


import torchaudio
import torchaudio._internal.fft
from torchaudio._internal.module_utils import deprecated

from scipy.io import wavfile
import librosa.display
from matplotlib import pyplot as plt

from utils.kaldiio_modules import *

# numeric_limits<float>::epsilon() 1.1920928955078125e-07
EPSILON = torch.tensor(torch.finfo(torch.float).eps)
# 1 milliseconds = 0.001 seconds
MILLISECONDS_TO_SECONDS = 0.001

# window types
HAMMING = 'hamming'
HANNING = 'hanning'
POVEY = 'povey'
RECTANGULAR = 'rectangular'
BLACKMAN = 'blackman'
WINDOWS = [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]

#### PARAMETERS ######################
waveform: Tensor
blackman_coeff: float = 0.42
channel: int = -1
dither: float = 1.0
energy_floor: float = 0.0
frame_length: float = 25.0
frame_shift: float = 10.0
min_duration: float = 0.0
preemphasis_coefficient: float = 0.97
raw_energy: bool = True
remove_dc_offset: bool = True
round_to_power_of_two: bool = True
sample_frequency: float = 16000.0
snip_edges: bool = False
subtract_mean: bool = False
window_type: str = POVEY

#fbank
high_freq: float = 7600.0
htk_compat: bool = False
low_freq: float = 20.0
num_mel_bins: int = 30
use_energy: bool = True
use_log_fbank: bool = True
use_power: bool = True
vtln_high: float = -500.0
vtln_low: float = 100.0
vtln_warp: float = 1.0

#mfcc
cepstral_lifter: float = 22.0
num_ceps: int = 30

# Spec2MFCC
padded_window_size: int = 512
##########################################

def Wav2Spec(utt):
    torch.manual_seed(100)
    audio = np.copy(utt)
    waveform=torch.Tensor(audio).unsqueeze(0)
    
    device, dtype = waveform.device, waveform.dtype
    epsilon = _get_epsilon(device, dtype)
    
    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient)
    
    strided_input, signal_log_energy = _get_window(
        waveform, padded_window_size, window_size, window_shift, window_type, blackman_coeff,
        snip_edges, raw_energy, energy_floor, dither, remove_dc_offset, preemphasis_coefficient)
    
    power_spectrum = torchaudio._internal.fft.rfft(strided_input).abs()
    if use_power:
        power_spectrum = power_spectrum.pow(2.)
        
    return signal_log_energy, power_spectrum

def Spec2MFCC(spectrum, signal_log_energy):
    device, dtype = spectrum.device, spectrum.dtype
    
    # size (num_mel_bins, padded_window_size // 2)
    mel_energies, _ = get_mel_banks(num_mel_bins, padded_window_size, sample_frequency,
                                    low_freq, high_freq, vtln_low, vtln_high, vtln_warp)
    mel_energies = mel_energies.to(device=device, dtype=dtype)
    
    # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
    mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode='constant', value=0)
    
    
    # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
    mel_energies = torch.mm(spectrum, mel_energies.T)
    if use_log_fbank:
        # avoid log of zero (which should be prevented anyway by dithering)
        mel_energies = torch.max(mel_energies, _get_epsilon(device, dtype)).log()
        
    # if use_energy then add it as the last column for htk_compat == true else first column
    if use_energy:
        signal_log_energy = signal_log_energy.unsqueeze(1)  # size (m, 1)
        # returns size (m, num_mel_bins + 1)
        if htk_compat:
            mel_energies = torch.cat((mel_energies, signal_log_energy), dim=1)
        else:
            mel_energies = torch.cat((signal_log_energy, mel_energies), dim=1)
            
        mel_energies = _subtract_column_mean(mel_energies, subtract_mean)
        
        feature = mel_energies
        
        if use_energy:
            # size (m)
            signal_log_energy = feature[:, num_mel_bins if htk_compat else 0]
            # offset is 0 if htk_compat==True else 1
            mel_offset = int(not htk_compat)
            feature = feature[:, mel_offset:(num_mel_bins + mel_offset)]
            
            # size (num_mel_bins, num_ceps)
        dct_matrix = _get_dct_matrix(num_ceps, num_mel_bins).to(dtype=dtype, device=device)
        
        # size (m, num_ceps)
        feature = feature.matmul(dct_matrix)
        
        if cepstral_lifter != 0.0:
            # size (1, num_ceps)
            lifter_coeffs = _get_lifter_coeffs(num_ceps, cepstral_lifter).unsqueeze(0)
            feature *= lifter_coeffs.to(device=device, dtype=dtype)
            
        # if use_energy then replace the last column for htk_compat == true else first column
        if use_energy:
            feature[:, 0] = signal_log_energy
            
        if htk_compat:
            energy = feature[:, 0].unsqueeze(1)  # size (m, 1)
            feature = feature[:, 1:]  # size (m, num_ceps - 1)
            if not use_energy:
                # scale on C0 (actually removing a scale we previously added that's
                # part of one common definition of the cosine transform.)
                energy *= math.sqrt(2)
                
            feature = torch.cat((feature, energy), dim=1)
            
        feature = _subtract_column_mean(feature, subtract_mean)
    return feature

def Wav2MFCC(utt):
    torch.manual_seed(100)
    audio = np.copy(utt)
    waveform=torch.Tensor(audio).unsqueeze(0)
    
    device, dtype = waveform.device, waveform.dtype #wav 2 mfcc
    epsilon = _get_epsilon(device, dtype)
    
    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient)
    
    strided_input, signal_log_energy = _get_window(
        waveform, padded_window_size, window_size, window_shift, window_type, blackman_coeff,
        snip_edges, raw_energy, energy_floor, dither, remove_dc_offset, preemphasis_coefficient)
    
    power_spectrum = torchaudio._internal.fft.rfft(strided_input).abs()
    if use_power:
        power_spectrum = power_spectrum.pow(2.)
    
    spectrum = power_spectrum #spec to mfcc
    device, dtype = spectrum.device, spectrum.dtype
    
    # size (num_mel_bins, padded_window_size // 2)
    mel_energies, _ = get_mel_banks(num_mel_bins, padded_window_size, sample_frequency,
                                    low_freq, high_freq, vtln_low, vtln_high, vtln_warp)
    mel_energies = mel_energies.to(device=device, dtype=dtype)
    
    # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
    mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode='constant', value=0)
    
    
    # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
    mel_energies = torch.mm(spectrum, mel_energies.T)
    if use_log_fbank:
        # avoid log of zero (which should be prevented anyway by dithering)
        mel_energies = torch.max(mel_energies, _get_epsilon(device, dtype)).log()
        
    # if use_energy then add it as the last column for htk_compat == true else first column
    if use_energy:
        signal_log_energy = signal_log_energy.unsqueeze(1)  # size (m, 1)
        # returns size (m, num_mel_bins + 1)
        if htk_compat:
            mel_energies = torch.cat((mel_energies, signal_log_energy), dim=1)
        else:
            mel_energies = torch.cat((signal_log_energy, mel_energies), dim=1)
            
        mel_energies = _subtract_column_mean(mel_energies, subtract_mean)

        feature = mel_energies
        
        if use_energy:
            # size (m)
            signal_log_energy = feature[:, num_mel_bins if htk_compat else 0]
            # offset is 0 if htk_compat==True else 1
            mel_offset = int(not htk_compat)
            feature = feature[:, mel_offset:(num_mel_bins + mel_offset)]
            
            # size (num_mel_bins, num_ceps)
        dct_matrix = _get_dct_matrix(num_ceps, num_mel_bins).to(dtype=dtype, device=device)
        
        # size (m, num_ceps)
        feature = feature.matmul(dct_matrix)
        
        if cepstral_lifter != 0.0:
            # size (1, num_ceps)
            lifter_coeffs = _get_lifter_coeffs(num_ceps, cepstral_lifter).unsqueeze(0)
            feature *= lifter_coeffs.to(device=device, dtype=dtype)
            
        # if use_energy then replace the last column for htk_compat == true else first column
        if use_energy:
            feature[:, 0] = signal_log_energy
            
        if htk_compat:
            energy = feature[:, 0].unsqueeze(1)  # size (m, 1)
            feature = feature[:, 1:]  # size (m, num_ceps - 1)
            if not use_energy:
                # scale on C0 (actually removing a scale we previously added that's
                # part of one common definition of the cosine transform.)
                energy *= math.sqrt(2)
                
            feature = torch.cat((feature, energy), dim=1)
            
    feature = _subtract_column_mean(feature, subtract_mean)
    return spectrum, feature

#if __name__ == '__main__':
