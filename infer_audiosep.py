import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

script_dir = Path(__file__).resolve().parent
audiosep_dir = script_dir / "models" / "audiosep"
TEST_DATA_DIR = script_dir / "examples"
if not audiosep_dir.exists():
    raise FileNotFoundError(f"AudioSep directory not found: {audiosep_dir}")
sys.path.insert(0, str(audiosep_dir))

from models.clap_encoder import CLAP_Encoder
from utils import load_ss_model, parse_yaml

AUDIOSEP_SR = 32000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AudioSep inference from the Hive wrapper."
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default=str(TEST_DATA_DIR / "acoustic_guitar.wav"),
        help="Input audio file path.",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clap_ckpt = hf_hub_download(
        repo_id="ShandaAI/AudioSep-hive",
        filename="music_speech_audioset_epoch_15_esc_89.98.pt",
    )
    query_encoder = CLAP_Encoder(pretrained_path=clap_ckpt).eval()

    config_file = hf_hub_download(
        repo_id="ShandaAI/AudioSep-hive",
        filename="config.yaml",
    )
    checkpoint_file = hf_hub_download(
        repo_id="ShandaAI/AudioSep-hive",
        filename="audiosep_hive.ckpt",
    )
    configs = parse_yaml(config_file)
    model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_file,
        query_encoder=query_encoder,
    )
    model = model.to(device).eval()

    mixture, _ = librosa.load(str(audio_file), sr=AUDIOSEP_SR, mono=True)
    input_len = mixture.shape[0]

    with torch.no_grad():
        conditions = model.query_encoder.get_query_embed(
            modality="text", text=[args.text], device=device
        )
        input_dict = {
            "mixture": torch.tensor(mixture)[None, None, :].to(device),
            "condition": conditions,
        }
        if input_len > AUDIOSEP_SR * 10:
            sep_audio = model.ss_model.chunk_inference(input_dict).squeeze()
            if isinstance(sep_audio, torch.Tensor):
                sep_audio = sep_audio.data.cpu().numpy()
        else:
            sep_segment = model.ss_model(input_dict)["waveform"]
            sep_audio = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
        sep_audio = np.asarray(sep_audio[:input_len], dtype=np.float32)

    torchaudio.save(str(output_file), torch.from_numpy(sep_audio).view(1, -1), AUDIOSEP_SR)


if __name__ == "__main__":
    main()