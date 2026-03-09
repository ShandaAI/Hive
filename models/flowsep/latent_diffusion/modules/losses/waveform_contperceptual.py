import torch
import torch.nn as nn
import numpy as np

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from latent_encoder.wavedecoder import *


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class WaveformDiscriminatorLoss(nn.Module):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.segment_size = 8192
        # output log variance
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.discriminator_iter_start = disc_start
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.window = None

    def _spectrogram(self, y, n_fft, hop_size, win_size, center=False):
        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))

        if self.window is None:
            self.window = torch.hann_window(win_size).to(y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
            mode="reflect",
        )

        spec = torch.stft(
            y.squeeze(1),
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=self.window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = spectral_normalize_torch(spec)
        return y, spec

    def disc_waveform_loss(self, y, y_g_hat):
        y, y_g_hat = self.random_segment_y_y_hat(y, y_g_hat)

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
            y_df_hat_r, y_df_hat_g
        )

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
            y_ds_hat_r, y_ds_hat_g
        )

        return loss_disc_s + loss_disc_f

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def random_segment_y_y_hat(self, y, y_g_hat):
        wavelength = min(y.size(-1), y_g_hat.size(-1))
        random_start = int(self.random_uniform(0, int(wavelength - self.segment_size)))
        return (
            y[..., random_start : random_start + self.segment_size],
            y_g_hat[..., random_start : random_start + self.segment_size],
        )

    def gen_waveform_loss(self, y, y_g_hat, global_step):
        y_g_hat, y_g_hat_spec = self._spectrogram(y_g_hat.squeeze(1), 1024, 160, 1024)
        y, y_spec = self._spectrogram(y.squeeze(1), 1024, 160, 1024)

        loss_spec = F.l1_loss(y_spec, y_g_hat_spec) * 45

        disc_factor = adopt_weight(
            self.disc_factor, global_step, threshold=self.discriminator_iter_start
        )

        y, y_g_hat = self.random_segment_y_y_hat(y, y_g_hat)

        assert y.size() == y_g_hat.size(), "%s %s" % (y.size(), y_g_hat.size())

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

        return (
            loss_spec,
            (loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f) * disc_factor,
        )

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        waveform,
        rec_waveform,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()).mean()

        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator updat
            loss, disc_loss = self.gen_waveform_loss(
                waveform, rec_waveform, global_step
            )

            log = {
                "{}/stftloss".format(split): loss.clone().detach().mean(),
                "{}/disc_gen_loss".format(split): disc_loss.clone().detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
            }
            return loss + disc_loss + kl_loss * self.kl_weight + rec_loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            disc_loss = self.disc_waveform_loss(waveform, rec_waveform)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * disc_loss

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean()}
            return d_loss, log
