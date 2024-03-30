import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from tqdm import tqdm
import os
import argparse
import json
import torch
torch.set_warn_always(False)
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader

from academicodec.models.hificodec.env import AttrDict
from academicodec.models.hificodec.meldataset import MelDataset, mel_spectrogram, get_validation_filelist
from academicodec.models.hificodec.models import Generator
from academicodec.models.hificodec.models import Encoder
from academicodec.models.hificodec.models import Quantizer
from academicodec.utils import plot_spectrogram
from academicodec.utils import load_checkpoint

torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    torch.cuda.set_device(rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    encoder = Encoder(h).to(device)
    generator = Generator(h).to(device)
    quantizer = Quantizer(h).to(device)

    state_dict_g = load_checkpoint(a.pretrained_path, device)
    generator.load_state_dict(state_dict_g['generator'])
    encoder.load_state_dict(state_dict_g['encoder'])
    quantizer.load_state_dict(state_dict_g['quantizer'])

    validation_filelist = get_validation_filelist(a)

    validset = MelDataset(
        validation_filelist,
        a.input_hash_file,
        h.segment_size * 15,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        split=True,
        shuffle=False,
        n_cache_reuse=0,
        fmax_loss=h.fmax_for_loss,
        device=device,
        fine_tuning=None,
        base_mels_path=None)
    validation_loader = DataLoader(
        validset,
        num_workers=16,
        shuffle=False,
        sampler=None,
        batch_size=128,
        pin_memory=True,
        drop_last=False)

    # Validation
    generator.eval()
    encoder.eval()
    quantizer.eval()
    val_err_tot = 0
    sw = SummaryWriter(a.pretrained_path + "_samples")
    with torch.no_grad():
        for j, batch in enumerate(tqdm(validation_loader)):
            x, y, _, y_mel = batch
            c = encoder(y.to(device).unsqueeze(1))
            q, loss_q, c = quantizer(c)
            y_g_hat = generator(q)
            y_mel = torch.autograd.Variable(y_mel.to(device))
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                h.sampling_rate, h.hop_size, h.win_size, h.fmin,
                h.fmax_for_loss)
            i_size = min(y_mel.size(2), y_g_hat_mel.size(2))
            val_err_tot += F.l1_loss(
                y_mel[:, :, :i_size],
                y_g_hat_mel[:, :, :i_size]).item()

            if j <= 8:
                sw.add_audio('gt/y_{}'.format(j), y[0],
                                0, h.sampling_rate)
                sw.add_figure('gt/y_spec_{}'.format(j),
                                plot_spectrogram(x[0]), 0)

                sw.add_audio('generated/y_hat_{}'.format(j),
                                y_g_hat[0], 0, h.sampling_rate)
                y_hat_spec = mel_spectrogram(
                    y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                    h.sampling_rate, h.hop_size, h.win_size,
                    h.fmin, h.fmax)
                sw.add_figure(
                    'generated/y_hat_spec_{}'.format(j),
                    plot_spectrogram(
                        y_hat_spec[0].cpu().numpy()),
                    0)

        val_err = val_err_tot / (j + 1)
        print(val_err)


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_validation_file', required=True)
    parser.add_argument('--input_hash_file', required=True)
    parser.add_argument('--pretrained_path', required=True)
    parser.add_argument('--config', required=True)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    train(0, a, h)


if __name__ == '__main__':
    main()
