import json

import torch
import torch.nn as nn

from academicodec.models.hificodec.env import AttrDict
from academicodec.models.hificodec.models import Encoder
from academicodec.models.hificodec.models import Generator
from academicodec.models.hificodec.models import Quantizer


class VQVAE(nn.Module):
    def __init__(self,
                 config_path,
                 ckpt_path,
                 with_encoder=False):
        super(VQVAE, self).__init__()
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        if ckpt_path:
            ckpt = torch.load(ckpt_path)
            self.generator.load_state_dict(ckpt['generator'])
            self.quantizer.load_state_dict(ckpt['quantizer'])
        if with_encoder:
            self.encoder = Encoder(self.h)
            if ckpt_path:
                self.encoder.load_state_dict(ckpt['encoder'])

    def wav_to_acoustic_token(self, x):
        '''Input wav, output acoustic tokens'''

        batch_size = x.size(0)
        if len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        c = self.encoder(x.unsqueeze(1))
        _, _, c = self.quantizer(c)
        c = [code.reshape(batch_size, -1) for code in c]
        c = torch.stack(c, -1)

        return c

    def wav_to_text_target(self, x):
        '''Input wav, output target sequence for text encoder'''

        c = self.wav_to_acoustic_token(x)
        batch_size = c.size(0)

        return c.reshape(batch_size, -1)

    def acoustic_token_to_wav(self, x):
        '''Input acoutic tokens, output wav'''
        quantized_vector = self.quantizer.embed(x)
        quantized_vector = quantized_vector.transpose(1, 2)
        y_hat = self.generator(quantized_vector)

        return y_hat

    def text_target_to_wav(self, x):
        '''
        Input target sequence for text encoder, output wav
        Input shape: B, T*n_layers*n_codegroup
        '''
        
        batch_size = x.size(0)
        c = x.reshape(batch_size, -1, self.quantizer.n_code_groups * self.quantizer.residual_layer)
        q = self.quantizer.embed(c)
        q = q.transpose(1, 2)
        y_hat = self.generator(q)

        return y_hat

    def text_target_to_latent_image(self, x):
        '''
        Input target sequence for text encoder, output latent image
        Input shape: B, T*n_layers*n_codegroup
        Output shape: B, D, T
        '''
        
        batch_size = x.size(0)
        c = x.reshape(batch_size, -1, self.quantizer.n_code_groups * self.quantizer.residual_layer)
        q = self.quantizer.embed(c)
        q = q.transpose(1, 2)

        return q

    def forward(self, x):
        '''Reconstruct the input wav. Input wav, output reconstructed similar wav'''
        c = self.wav_to_acoustic_token(x)
        y_hat = self.acoustic_token_to_wav(c)

        return y_hat