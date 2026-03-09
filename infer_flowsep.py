import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FLOWSEP_DIR = SCRIPT_DIR / "models" / "flowsep"
TEST_DATA_DIR = SCRIPT_DIR / "examples"
if not FLOWSEP_DIR.exists():
    raise FileNotFoundError(f"FlowSep directory not found: {FLOWSEP_DIR}")
sys.path.insert(0, str(FLOWSEP_DIR))

import torch
from huggingface_hub import hf_hub_download
from pytorch_lightning import seed_everything
from latent_diffusion.util import instantiate_from_config
import yaml
import numpy as np
import torchaudio

FLOWSEP_CHUNK_IN = 163840
FLOWSEP_CHUNK_OUT = 160000
FLOWSEP_SR = 16000


class FlowSepPreprocessor:
    def __init__(self, config):
        import utilities.audio as Audio

        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.duration = config["preprocessing"]["audio"]["duration"]
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def wav_feature_extraction(self, waveform):
        import utilities.audio as Audio

        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)
        log_mel_spec, stft, _ = Audio.tools.get_mel_from_wav(waveform, self.STFT)
        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)
        log_mel_spec = self._pad_spec(log_mel_spec)
        stft = self._pad_spec(stft)
        return log_mel_spec, stft

    def _pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[: self.target_length, :]
        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]
        return log_mel_spec

    def load_full_audio(self, filename):
        waveform, sr = torchaudio.load(filename)
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        waveform = waveform.numpy()[0, ...]
        return waveform

    def preprocess_chunk(self, chunk):
        chunk = chunk - np.mean(chunk)
        chunk = chunk / (np.max(np.abs(chunk)) + 1e-8)
        chunk = chunk * 0.5
        return chunk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FlowSep inference from the Hive wrapper."
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default=str(TEST_DATA_DIR / "acoustic_guitar.wav"),
        help="Input mixed audio file path.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="acoustic_guitar",
        help="Text query used for source separation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(TEST_DATA_DIR / "separated_audio.wav"),
        help="Output separated audio file path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    audio_file = Path(args.audio_file).expanduser()
    output_file = Path(args.output_file).expanduser()
    if not audio_file.exists():
        raise FileNotFoundError(f"Input audio file not found: {audio_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_file = hf_hub_download(
        repo_id="ShandaAI/FlowSep-hive",
        filename="config.yaml",
    )
    model_file = hf_hub_download(
        repo_id="ShandaAI/FlowSep-hive",
        filename="flowsep_hive.ckpt",
    )

    with open(config_file, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    preprocessor = FlowSepPreprocessor(configs)
    latent_diffusion = instantiate_from_config(configs["model"]).to(device)
    try:
        ckpt = torch.load(model_file, map_location=device, weights_only=False)["state_dict"]
    except TypeError:
        ckpt = torch.load(model_file, map_location=device)["state_dict"]
    latent_diffusion.load_state_dict(ckpt, strict=True)
    latent_diffusion.eval()

    with torch.no_grad():
        full_wav = preprocessor.load_full_audio(str(audio_file))
        input_len = full_wav.shape[0]

        def _process_chunk(chunk_wav):
            chunk_wav = preprocessor.preprocess_chunk(chunk_wav)
            if len(chunk_wav) < FLOWSEP_CHUNK_IN:
                pad = np.zeros(FLOWSEP_CHUNK_IN - len(chunk_wav), dtype=np.float32)
                chunk_wav = np.concatenate([chunk_wav, pad])
            chunk_wav = chunk_wav[:FLOWSEP_CHUNK_IN]
            mixed_mel, _ = preprocessor.wav_feature_extraction(chunk_wav.reshape(1, -1))
            batch = {
                "fname": [str(audio_file)],
                "text": [args.text],
                "caption": [args.text],
                "waveform": torch.rand(1, 1, FLOWSEP_CHUNK_IN).to(device),
                "log_mel_spec": torch.rand(1, 1024, 64).to(device),
                "sampling_rate": torch.tensor([FLOWSEP_SR]).to(device),
                "label_vector": torch.rand(1, 527).to(device),
                "stft": torch.rand(1, 1024, 512).to(device),
                "mixed_waveform": torch.from_numpy(
                    chunk_wav.reshape(1, 1, FLOWSEP_CHUNK_IN)
                ).to(device),
                "mixed_mel": mixed_mel.reshape(
                    1, mixed_mel.shape[0], mixed_mel.shape[1]
                ).to(device),
            }
            result = latent_diffusion.generate_sample(
                [batch],
                name="temp_result",
                unconditional_guidance_scale=1.0,
                ddim_steps=20,
                n_gen=1,
                save=False,
                save_mixed=False,
            )
            if isinstance(result, np.ndarray):
                out = result.squeeze()
            else:
                out = result.squeeze().cpu().numpy()
            return out[:FLOWSEP_CHUNK_OUT]

        if input_len <= FLOWSEP_CHUNK_IN:
            sep_audio = _process_chunk(full_wav.copy())
        else:
            out_list = []
            start = 0
            while start < input_len:
                end = min(start + FLOWSEP_CHUNK_IN, input_len)
                chunk = full_wav[start:end]
                out_chunk = _process_chunk(chunk.copy())
                need = min(FLOWSEP_CHUNK_OUT, input_len - start)
                out_list.append(out_chunk[:need])
                start += FLOWSEP_CHUNK_OUT
            sep_audio = np.concatenate(out_list)

        if len(sep_audio) > input_len:
            sep_audio = sep_audio[:input_len]
        elif len(sep_audio) < input_len:
            sep_audio = np.pad(
                sep_audio, (0, input_len - len(sep_audio)), mode="constant", constant_values=0
            )

        torchaudio.save(
            str(output_file),
            torch.from_numpy(np.asarray(sep_audio, dtype=np.float32)).view(1, -1).cpu(),
            FLOWSEP_SR,
        )


if __name__ == "__main__":
    main()
