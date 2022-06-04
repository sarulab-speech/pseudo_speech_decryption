#!/bin/bash

# give MFCC filename for computing EER
mfcc_txt=mfcc/mfcc.txt

# eval directory name for storing the necessary files for computing EER
DIR=TEST 

kaldi_root=/home/yuunin/VoicePrivacy/vp_adv/Voice-Privacy-Challenge-2020/kaldi/src
feats=/featbin/copy-feats
vector=/bin/copy-vector
UTT2SPK=/home/yuunin/VoicePrivacy/vp_adv/Voice-Privacy-Challenge-2020/baseline/utils/utt2spk_to_spk2utt.pl

dir=/home/yuunin/VoicePrivacy/vp_adv/Voice-Privacy-Challenge-2020/baseline/data/

# create directory if it does not exist
if [[ ! -e $dir/$DIR ]]; then
    mkdir $dir/$DIR
fi

# removing existing files in eval directory
for files in data conf frame_shift utt2dur utt2num_frames vad.scp utt2dur; do
    rm -rf $dir/$DIR/$files
done

# necessary files to compute EER : trials, feats.ark, feats.scp, wav.scp, spk2utt, utt2spk   


# 1. make trials file
# since we only focus on the male dataset of the test dataset, we will copy vctk_test/trials_m_mic2 to the eval directory and rename as trials
cp $dir/vctk_test/trials_m_mic2 $dir/$DIR/trials


# 2. make feats.ark and feats.scp files
$kaldi_root$feats ark,t:$mfcc_txt ark,scp:"feats.ark","feats_tmp.scp"

#fix feats.scp file to absolute path
less "feats_tmp.scp" | while read line
do
    echo `echo $line | awk '{print $1}'` $dir"/"`echo $line | awk '{print $2}'` >> "feats.scp"
done

# removing feats_tmp.scp as it is unneeded anymore
rm -rf feats_tmp.scp


# 3. make wav.scp file
less feats.scp | awk '{print $1}' | while read line
do
    echo $line "/home/yuunin/VoicePrivacy/vp_adv/Voice-Privacy-Challenge-2020/baseline/data/vctk_test/wav/"`echo $line | awk -F[_] '{print $1}'`"/"$line".wav" >> wav.scp
done

# 4. make spk2utt and utt2spk files
less feats.scp | awk '{print $1}' | while read line
do
    echo $line `echo $line | awk -F[_] '{print $1}'` >> utt2spk
done

$UTT2SPK utt2spk > spk2utt


# 5. move files to eval directory
for files in feats{.ark,.scp} wav.scp utt2spk spk2utt; do
    mv $files $dir/$DIR/$files
done
