import torch
import numpy as np
from scipy.io.wavfile import write
import torchaudio

from utilities.audio.audio_processing import griffin_lim


def get_mel_from_wav(audio, _stft, return_numpy=True):
    stft_device = _stft.mel_basis.device
    if isinstance(audio, torch.Tensor):
        audio_tensor = audio.to(stft_device, dtype=torch.float32)
    else:
        audio_tensor = torch.tensor(audio, device=stft_device, dtype=torch.float32)

    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = torch.clip(audio_tensor, -1, 1)
    audio_tensor = torch.autograd.Variable(audio_tensor, requires_grad=False)
    melspec, magnitudes, phases, energy = _stft.mel_spectrogram(audio_tensor)

    melspec = torch.squeeze(melspec, 0)
    magnitudes = torch.squeeze(magnitudes, 0)
    energy = torch.squeeze(energy, 0)

    if return_numpy:
        melspec = melspec.detach().cpu().numpy().astype(np.float32)
        magnitudes = magnitudes.detach().cpu().numpy().astype(np.float32)
        energy = energy.detach().cpu().numpy().astype(np.float32)

    return melspec, magnitudes, energy


def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
    mel = torch.stack([mel])
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(
        torch.autograd.Variable(spec_from_mel[:, :, :-1]), _stft._stft_fn, griffin_iters
    )

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, _stft.sampling_rate, audio)
