import os
import torch
import torchaudio
import numpy as np
import utilities.audio as Audio
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read
import ipdb

MAX_WAV_VALUE = 32768.0



def wav2mel(filename, hop_length = 240,mel_channel = 256):

        audio, sampling_rate = load_wav(filename, 48000)

        if len(audio.shape)>1:
            audio = audio[:,0]
        audio = audio / MAX_WAV_VALUE

        audio = normalize(audio) * 0.95

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        # ipdb.set_trace()


        mel = mel_spectrogram(audio, 2048, mel_channel, 48000, hop_length, 2048, 0, 24000).T.squeeze()

        frame = int(15360/hop_length)

        # ipdb.set_trace()

        if mel.shape[0]< frame:
            padding_size = frame - mel.size(0)
    

            padding = torch.zeros(padding_size, mel.size(1))
    

            mel = torch.cat((mel, padding), dim=0)
        else:
            mel = mel[:frame,:]
        

        return mel

def load_wav(full_path, sr_target):
    sampling_rate, data = read(full_path)
    if sampling_rate != sr_target:
        print(f"sampling rate is wrong with {sampling_rate}hz and the file of {full_path}")
    return data, sampling_rate

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False,stft=None):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        # mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)

        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output