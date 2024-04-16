import glob
from tqdm import tqdm
import os
from multiprocessing import Pool, Manager
import json

audio_dir = '/home/ubuntu/tts-dev-drive/libriheavy/libriheavy/vae_eval_samples'
id = 27323
files = os.listdir(audio_dir)
file_names = [str(id)+'/'+file for file in files]

with open('data/valid.lst','w') as f:
    f.write('\n'.join(file_names))