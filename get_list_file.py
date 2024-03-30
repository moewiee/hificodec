import glob
from tqdm import tqdm
import os
from multiprocessing import Pool, Manager
import json

audio_dir = '/home/ubuntu/tts-dev-drive/tts-en/data_en'

dirs = glob.glob(f"{audio_dir}/*")
dirs = [f for f in dirs if os.path.isdir(f)][:]
dirs_hash = {value: idx for idx, value in enumerate(dirs)}

# Initialize a multiprocessing Manager and use it to create a list
manager = Manager()
files = manager.list()

def get_from_dir(d):
    # Access the global list through the manager proxy
    global files
    df = glob.glob(f"{d}/audio/*.ogg")
    # Since this is a Manager list, it's safe to append to it from multiple processes
    files.extend(df)

p = Pool(128)
for _ in tqdm(p.imap_unordered(get_from_dir, dirs), total=len(dirs)):
    pass

train_list = files[10000:]
val_list = files[:10000]

def write_file(file_list,path):
    with open(path,'w') as f:
        for line in tqdm(file_list):
            line = line.split("/audio/")
            line = str(dirs_hash[line[0]]) + "/audio/" + line[1]
            f.write(line + '\n')

train_path = 'data/train_en_hash.lst'
val_path = 'data/valid_en_hash.lst'
hash_path = 'data/en_hash.json'

write_file(val_list,val_path)
write_file(train_list,train_path)

with open(hash_path, "w") as f:
    json.dump(dirs_hash, f)