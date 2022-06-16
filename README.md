# Speech pseudonymization robust to decryption attack
In this repository, pseudonymized audios are decrypted by either counteracting the effect of the used voice modification methods or by using models specifically trained to recover the spectrogram of pseudonymized audio to its original spectrogram. We evaluate the decrypted audios by extracting MFCC from decrypted audio (or spectrogram) followed by computation of EER using x-vector-based speaker verification systems. For details regarding the speaker verification models used in this repository, refer to the [Voice Privacy 2020 Challenge evaluation protocol](https://www.voiceprivacychallenge.org/vp2020/docs/VoicePrivacy_2020_Eval_Plan_v1_4.pdf).


## Script descriptions
This repository consists of dev and test repository. The dev directory is used for training a model using spectrograms extracted from pseudonymized audios. The trained model is used in the test phase for recovering pseudonymized audio spectrograms to its original. The test directory contains python code for decrypting audios pseudonymized using cacading method and superposition method, respectively.

- dev/
     - `dev.protocol.csv`: audio list containing audio filenames for reading.
     - `train.py`: trains model for recovering spectrogram of pseudonymized audio to spectrogram of original audio.
          - \+ creates `save_models/` for saving best model.
		- \+ creates `scaler/` for saving MinMaxScaler parameters for later usage in test phase.
		- \+ creates `train_log/` for saving logs of training phase of model.
- test/
     - `audio2mfcc.py`: extracts MFCC from audios read using test.protocol.csv.
  		- \+ creates `mfcc/` for saving extracted MFCC.
     - `decrypt_cas.py`: decrypts audios pseudonymized using cascading method.
  		- \+ creates `cas/` for saving decrypted audios.
     - `decrypt_sup.py`: decrypts audios pseudonymized using superposition method.
  		- \+ creates `sup/` for saving decrypted audios
     - `eval.py`: recovers spectrogram of pseudonymized audio using pre-trained model.
  		- \+ creates `mfcc/` if it does not exist already.
     - `test.protocol.csv`: audio list containing audio filenames for reading.
- utils/
     - `kaldiio_modules.py`: contains pytorch modules replicating Kaldi's computational behaviors and results.
     - `Normalization.py`: contains pytorch code for spectrogram and signal normalization.
     - `UNet_model.py`: UNet model used for this project.
     - `Wav2Mfcc.py`: contains pytorch code for replicating Kaldi's computational behaviors and results (depends on `kaldiio_modules.py`)
     - `voice_change.py`: contains voice modification modules.
     - `voice_change_superpose.py`: contains voice modification modules, where each module returns array with same length as input.
     - `voice_change_decrypt.py`: contains voice modification modules designed to counteract each corresponding modules. 


## Script usage
0. Please install [VoicePrivacy 2020 repository modules](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020).
     - In order to use the datasets as well as ASV models and scripts to compute EER.
1. Prepare original and pseudonymized audio datasets where both datasets are structured as below.
```
  /
  |-- speaker1/                        # speaker label
  |     |-- audio1.wav, audio2.wav ... # wav files
  |-- speaker2/
		...
```
2. Run dev/train.py.
     - Original dataset and pseudonymized audio datasets are used to train UNet model. 
     - Pseudonymised audio datasets are normalized using MinMaxScaler, scaler parameters are saved to `scaler/` for later usage.
     - Best model saved to `save_models/`.
     - Training accuracies and validation accuracies are written to log file and saved to `train_log/`.
3. Decryption of signal processing-based methods
     1. Run pseudonymization. 
          - Please choose one from the following approaches. In order to evaluate the worst case possible, we pseudoymize audio on all possible permutations and create each corresponding directories for saving audios.
               - Option 1: run `test/decrypt_cas.py` for decrypting audios pseudonymized using cascading method. Decrypted audios saved in `cas/`.
               - Option 2: run test/decrypt_sup.py for decrypting audios pseudonymized using superposition method. Decrypted audios saved in `sup/`
     1. Run `audio2mfcc.py`. 
          - This will allow us to extract MFCC from audios saved in `cas/` or `sup/`. 
          - Note that you would need to specify root directory in sup/ correctly so that the structure is the same as the one shown above.
     1. Configure test/make_files.sh to match your environment. 
          - `mfcc_txt`: MFCC file that was generated in 3-2
          - `DIR` : Input a directory name to store necessary files for computing EER. This will be created under Voice-Privacy-Challenge-2020/baseline/data. 
          - Place the right paths for the remaining directories. 
          - The necessary files are created locally to which then is moved to directory DIR.
     1. In Voice-Privacy-Challenge-2020/baseline/local/asv_eval.sh
          - Change script so it does not generate the mfcc file (`vad.scp` and x-vector files are to be generated) so the one generated in 3-2 is not overwritten
          - EER will be computed at this stage
4. Decryption of machine learning-based methods
     1. Run test/eval.py
          - Model saved in `save_model/` are read, scaler parameters in `dev/scaler/` are loaded.
          - MFCC are extracted and saved to `mfcc/`.
     2. Go to 3-3

## Contributors
- Hiroto Kai (Tokyo Metropolitan University, Japan.) [main contributor]
- [Shinnosuke Takamichi](https://sites.google.com/site/shinnosuketakamichi/home) (The University of Tokyo, Japan.)
- [Sayaka Shiota](https://researchmap.jp/sayaka_shiota) (Tokyo Metropolitan University, Japan.)
- [Hitoshi Kiya](http://www-isys.sd.tmu.ac.jp/kiya/) (Tokyo Metropolitan University, Japan.)

## Reference 
- Hiroto Kai, Shinnosuke Takamichi, Sayaka Shiota and Hitoshi Kiya, "Robustness of signal processing-based pseudonymization method against decryption attack," Proc. Odyssey, June 2022.