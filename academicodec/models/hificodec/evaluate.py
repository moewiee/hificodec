import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from tqdm import tqdm
import argparse
import json
import torch
torch.set_warn_always(False)
import torch.nn.functional as F
from torch.utils.data import DataLoader

from academicodec.models.hificodec.env import AttrDict
from academicodec.models.hificodec.meldataset import MelDataset, mel_spectrogram, get_validation_filelist
from academicodec.models.hificodec.vqvae import VQVAE

torch.backends.cudnn.benchmark = True


def normalize(spectrogram):
    min_val = spectrogram.min()
    max_val = spectrogram.max()
    normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val)

    return normalized_spectrogram


def train(rank, a, h, refiner_h=None):
    torch.cuda.set_device(rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    base_model = VQVAE(a.config, a.pretrained_path, with_encoder=True).to(device)
    base_model.eval()

    if refiner_h is not None:
        refiner_model = VQVAE(a.refiner_config, a.refiner_path, with_encoder=True).to(device)
        refiner_model.eval()

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
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=False)

    # Validation
    base_f1 = 0
    base_mse = 0
    refined_f1 = 0
    refined_mse = 0

    with torch.no_grad():
        for j, batch in enumerate(tqdm(validation_loader)):
            x, y, _, y_mel = batch
            y_g_hat = base_model(y.to(device))
            y_mel = torch.autograd.Variable(y_mel.to(device))
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                h.sampling_rate, h.hop_size, h.win_size, h.fmin,
                h.fmax_for_loss)
            i_size = min(y_mel.size(2), y_g_hat_mel.size(2))
            base_f1 += F.l1_loss(
                normalize(y_mel[:, :, :i_size]),
                normalize(y_g_hat_mel[:, :, :i_size])).item()
            base_mse += F.mse_loss(
                    normalize(y_mel[:, :, :i_size]),
                    normalize(y_g_hat_mel[:, :, :i_size])).item()

            if refiner_h:
                y_g_hat = refiner_model(y_g_hat.squeeze(1))
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                    h.sampling_rate, h.hop_size, h.win_size, h.fmin,
                    h.fmax_for_loss)
                refined_f1 += F.l1_loss(
                    normalize(y_mel[:, :, :i_size]),
                    normalize(y_g_hat_mel[:, :, :i_size])).item()
                refined_mse += F.mse_loss(
                    normalize(y_mel[:, :, :i_size]),
                    normalize(y_g_hat_mel[:, :, :i_size])).item()

        base_f1 = base_f1 / (j + 1)
        base_mse = base_mse / (j + 1)
        print(f"Base F1 : {base_f1:.4f}")
        print(f"Base MSE: {base_mse:.4f}")

        if refiner_h:
            refined_f1 = refined_f1 / (j + 1)
            refined_mse = refined_mse / (j + 1)
            print(f"Refined F1 : {refined_f1:.4f}")
            print(f"Refined MSE: {refined_mse:.4f}")


def main():
    print('Initializing Evaluation Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_validation_file', required=True)
    parser.add_argument('--input_hash_file', required=True)
    parser.add_argument('--pretrained_path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--refiner_config', default='', required=False)
    parser.add_argument('--refiner_path', default='', required=False)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    if a.refiner_config != '':
        with open(a.refiner_config) as f:
            data = f.read()

        json_config = json.loads(data)
        h_refiner = AttrDict(json_config)

        if a.refiner_path == '':
            raise ValueError('Refiner path is required when refiner config is provided')
    else:
        h_refiner = None

    train(0, a, h, h_refiner)


if __name__ == '__main__':
    main()
