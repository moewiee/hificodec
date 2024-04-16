import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import itertools
import os
import time
import argparse
from datetime import datetime
import json
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from academicodec.models.hificodec.env import AttrDict, build_env
from academicodec.models.hificodec.meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from academicodec.models.encodec.msstftd import MultiScaleSTFTDiscriminator
from academicodec.models.hificodec.models import Generator
from academicodec.models.hificodec.models import MultiPeriodDiscriminator
from academicodec.models.hificodec.models import MultiScaleDiscriminator
from academicodec.models.hificodec.models import feature_loss
from academicodec.models.hificodec.models import generator_loss
from academicodec.models.hificodec.models import discriminator_loss
from academicodec.models.hificodec.models import Encoder
from academicodec.models.hificodec.models import Quantizer
from academicodec.utils import scan_checkpoint
from academicodec.utils import load_checkpoint
from academicodec.utils import save_checkpoint

torch.backends.cudnn.benchmark = True


def train(rank, a, h, pretrained_h):
    torch.cuda.set_device(rank)
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    base_encoder = Encoder(pretrained_h).to(device)
    base_generator = Generator(pretrained_h).to(device)
    base_quantizer = Quantizer(pretrained_h).to(device)

    encoder = Encoder(h).to(device)
    generator = Generator(h).to(device)
    quantizer = Quantizer(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    mstftd = MultiScaleSTFTDiscriminator(32).to(device)
    if rank == 0:
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    last_epoch = -1
    if cp_g is None or cp_do is None:
        state_dict_do = None
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        
        try:
            generator.load_state_dict(state_dict_g['generator'])
        except:
            cmd = generator.state_dict()
            lsd = state_dict_g['generator']
            nsd={k:v if v.size() == cmd[k].size() else cmd[k] for k,v in zip(cmd.keys(), lsd.values())}
            
            generator.load_state_dict(nsd, strict=False)
        
        try:
            encoder.load_state_dict(state_dict_g['encoder'])
        except:
            cmd = encoder.state_dict()
            lsd = state_dict_g['encoder']
            nsd={k:v if v.size() == cmd[k].size() else cmd[k] for k,v in zip(cmd.keys(), lsd.values())}
            
            encoder.load_state_dict(nsd, strict=False)

        try:
            quantizer.load_state_dict(state_dict_g['quantizer'])
        except:
            cmd = quantizer.state_dict()
            lsd = state_dict_g['quantizer']
            nsd={k:v if v.size() == cmd[k].size() else cmd[k] for k,v in zip(cmd.keys(), lsd.values())}
            
            quantizer.load_state_dict(nsd, strict=False)

        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        mstftd.load_state_dict(state_dict_do['mstftd'])
        if a.continue_optim:
            steps = state_dict_do['steps'] + 1
            last_epoch = state_dict_do['epoch']

    state_dict_pretrain = load_checkpoint(a.pretrain_path, device)

    base_generator.load_state_dict(state_dict_pretrain['generator'])
    base_generator.eval()
    print(f'Generator loaded from {a.pretrain_path}.')
    
    base_encoder.load_state_dict(state_dict_pretrain['encoder'])
    base_encoder.eval()
    print(f'Encoder loaded from {a.pretrain_path}.')
    
    base_quantizer.load_state_dict(state_dict_pretrain['quantizer'])
    base_quantizer.eval()
    print(f'Quantizer loaded from {a.pretrain_path}.')

    if a.refiner_pretrain_path:
        state_dict_pretrain = load_checkpoint(a.refiner_pretrain_path, device)
        if 'generator' in state_dict_pretrain.keys():
            cmd = generator.state_dict()
            lsd = state_dict_pretrain['generator']
            nsd={k:v if v.size() == cmd[k].size() else cmd[k] for k,v in zip(cmd.keys(), lsd.values())}
            
            generator.load_state_dict(nsd, strict=False)
            print(f'Generator loaded from {a.refiner_pretrain_path}.')
        if 'encoder' in state_dict_pretrain.keys():
            cmd = encoder.state_dict()
            lsd = state_dict_pretrain['encoder']
            nsd={k:v if v.size() == cmd[k].size() else cmd[k] for k,v in zip(cmd.keys(), lsd.values())}
            
            encoder.load_state_dict(nsd, strict=False)
            print(f'Encoder loaded from {a.refiner_pretrain_path}.')
        if 'quantizer' in state_dict_pretrain.keys():
            cmd = quantizer.state_dict()
            lsd = state_dict_pretrain['quantizer']
            nsd={k:v if v.size() == cmd[k].size() else cmd[k] for k,v in zip(cmd.keys(), lsd.values())}
            
            quantizer.load_state_dict(nsd, strict=False)
            print(f'Quantizer loaded from {a.refiner_pretrain_path}.')
        

    if h.num_gpus > 1:
        generator = DistributedDataParallel(
            generator, device_ids=[rank]).to(device)
        encoder = DistributedDataParallel(encoder, device_ids=[rank]).to(device)
        quantizer = DistributedDataParallel(
            quantizer, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
        mstftd = DistributedDataParallel(mstftd, device_ids=[rank]).to(device)

    optim_g = torch.optim.Adam(
        itertools.chain(generator.parameters(),
                        encoder.parameters(), quantizer.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.Adam(
        itertools.chain(msd.parameters(), mpd.parameters(),
                        mstftd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])
    
    if state_dict_do is not None and a.continue_optim:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, _ = get_dataset_filelist(a)

    print("Number of training samples:", len(training_filelist))

    trainset = MelDataset(
        training_filelist,
        a.input_hash_file,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        n_cache_reuse=0,
        shuffle=False if h.num_gpus > 1 else True,
        fmax_loss=h.fmax_for_loss,
        device=device)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True)

    if rank == 0:
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
        print("Done create dataset!")
    generator.train()
    encoder.train()
    quantizer.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            with torch.no_grad():
                base_c = base_encoder(y)
                base_q, _, _ = base_quantizer(base_c)
                base_y_g_hat = base_generator(base_q)

            c = encoder(base_y_g_hat)
            
            q, loss_q, c = quantizer(c)
            
            y_g_hat = generator(q)
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size, h.fmin,
                h.fmax_for_loss)  # 1024, 80, 24000, 240,1024
            y_r_mel_1 = mel_spectrogram(
                y.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512,
                h.fmin, h.fmax_for_loss)
            y_g_mel_1 = mel_spectrogram(
                y_g_hat.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512,
                h.fmin, h.fmax_for_loss)
            y_r_mel_2 = mel_spectrogram(
                y.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256, h.fmin,
                h.fmax_for_loss)
            y_g_mel_2 = mel_spectrogram(
                y_g_hat.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256,
                h.fmin, h.fmax_for_loss)
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g)

            y_disc_r, fmap_r = mstftd(y)
            y_disc_gen, fmap_gen = mstftd(y_g_hat.detach())
            loss_disc_stft, losses_disc_stft_r, losses_disc_stft_g = discriminator_loss(
                y_disc_r, y_disc_gen)
            loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_stft

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel1 = F.l1_loss(y_r_mel_1, y_g_mel_1)
            loss_mel2 = F.l1_loss(y_r_mel_2, y_g_mel_2)
            
            loss_mel = F.l1_loss(y_mel,
                                 y_g_hat_mel) * 45 + loss_mel1 + loss_mel2
            
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            y_stftd_hat_r, fmap_stftd_r = mstftd(y)
            y_stftd_hat_g, fmap_stftd_g = mstftd(y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_fm_stft = feature_loss(fmap_stftd_r, fmap_stftd_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_stft, losses_gen_stft = generator_loss(y_stftd_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_gen_stft + loss_fm_s + loss_fm_f + loss_fm_stft + loss_mel + loss_q * 10
            loss_gen_all.backward()
            optim_g.step()
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    print(
                        datetime.now().strftime("%b %d %Y %H:%M:%S") + ' | Steps : {:d}/{:d}, Gen Loss Total : {:4.3f}, Loss Q : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                        format(steps - epoch * len(train_loader), len(train_loader), loss_gen_all, loss_q, mel_error,
                               (time.time() - start)/(steps - epoch * len(train_loader) + 1)))
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path,
                                                           steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'generator': (generator.module if h.num_gpus > 1
                                          else generator).state_dict(),
                            'encoder': (encoder.module if h.num_gpus > 1 else
                                        encoder).state_dict(),
                            'quantizer': (quantizer.module if h.num_gpus > 1
                                          else quantizer).state_dict()
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path,
                                                            steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'mpd': (mpd.module
                                    if h.num_gpus > 1 else mpd).state_dict(),
                            'msd': (msd.module
                                    if h.num_gpus > 1 else msd).state_dict(),
                            'mstftd': (mstftd.module
                                       if h.num_gpus > 1 else msd).state_dict(),
                            'optim_g':
                            optim_g.state_dict(),
                            'optim_d':
                            optim_d.state_dict(),
                            'steps':
                            steps,
                            'epoch':
                            epoch
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all,
                                  steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(
                epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_training_file', required=True)
    parser.add_argument('--input_validation_file', required=True)
    parser.add_argument('--input_hash_file', required=True)
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--pretrained_config', required=True)
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=2, type=int)
    parser.add_argument('--stdout_interval', default=20, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--num_ckpt_keep', default=100, type=int)
    parser.add_argument('--pretrain_path', type=str, required=True)
    parser.add_argument('--refiner_pretrain_path', default='', type=str, required=False)
    parser.add_argument('--continue_optim', action='store_true')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    with open(a.pretrained_config) as f:
        data = f.read()
    pretrained_json_config = json.loads(data)
    pretrained_h = AttrDict(pretrained_json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h, pretrained_h, ))
    else:
        train(0, a, h, pretrained_h)


if __name__ == '__main__':
    main()
