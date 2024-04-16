import sys

sys.path.append('../../')
import json
import torchaudio
import os
import librosa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import torch
from academicodec.models.hificodec.vqvae import VQVAE
from academicodec.models.hificodec.meldataset import mel_spectrogram
from librosa.util import normalize


pretrained_config_path = './config_24k_960d.json'
pretrained_path = '50m_logs_4cb_960d/step_110k'
wav_path = './sample.wav'

with open(pretrained_config_path, 'r') as f:
    pretrained_config = json.load(f)
    sample_rate = pretrained_config['sampling_rate']

wav, sr = librosa.load(wav_path, sr=sample_rate)
print("wav.shape:",wav.shape)
assert sr == sample_rate

wav = normalize(wav) * 0.95
wav = torch.FloatTensor(wav).unsqueeze(0)

print(wav.size())

pretrained_model = VQVAE(
    pretrained_config_path,
    ckpt_path=pretrained_path,
    with_encoder=True)
pretrained_model.eval()

wav_new = pretrained_model.recontruct_with_interpolation(wav)

print(wav_new.size())

torchaudio.save('reconstruct_interpolate.wav', wav_new[0], sample_rate, channels_first=True)