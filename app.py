import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "audiosep"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "flowsep"))

import gradio as gr
import torch
import numpy as np
import torchaudio
import librosa
import yaml
from huggingface_hub import hf_hub_download
from pytorch_lightning import seed_everything

try:
    import spaces
except ImportError:
    spaces = None

_audiosep_model = None
_flowsep_model = None
_flowsep_preprocessor = None


def get_runtime_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def read_wav_file(self, filename):
        waveform, sr = torchaudio.load(filename)
        target_length = int(sr * self.duration)
        if waveform.shape[-1] > target_length:
            waveform = waveform[:, :target_length]
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        waveform = waveform.numpy()[0, ...]
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        waveform = waveform * 0.5
        waveform = waveform[None, ...]
        target_samples = int(self.sampling_rate * self.duration)
        if waveform.shape[-1] < target_samples:
            temp_wav = np.zeros((1, target_samples), dtype=np.float32)
            temp_wav[:, :waveform.shape[-1]] = waveform
            waveform = temp_wav
        return waveform

    def wav_feature_extraction(self, waveform):
        import utilities.audio as Audio

        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)
        log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)
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
            log_mel_spec = log_mel_spec[:self.target_length, :]
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


def load_audiosep():
    global _audiosep_model
    device = get_runtime_device()
    if _audiosep_model is not None:
        _audiosep_model = _audiosep_model.to(device).eval()
        return _audiosep_model

    from models.clap_encoder import CLAP_Encoder
    from utils import parse_yaml, load_ss_model

    clap_ckpt = hf_hub_download(repo_id="ShandaAI/AudioSep-hive", filename="music_speech_audioset_epoch_15_esc_89.98.pt")
    query_encoder = CLAP_Encoder(pretrained_path=clap_ckpt).eval()

    config_file = hf_hub_download(repo_id="ShandaAI/AudioSep-hive", filename="config.yaml")
    checkpoint_file = hf_hub_download(repo_id="ShandaAI/AudioSep-hive", filename="audiosep_hive.ckpt")
    configs = parse_yaml(config_file)
    model = load_ss_model(configs=configs, checkpoint_path=checkpoint_file, query_encoder=query_encoder)
    model = model.to(device).eval()
    _audiosep_model = model
    return model


def load_flowsep():
    global _flowsep_model, _flowsep_preprocessor
    device = get_runtime_device()
    if _flowsep_model is not None:
        _flowsep_model = _flowsep_model.to(device).eval()
        return _flowsep_model, _flowsep_preprocessor

    seed_everything(0)
    from latent_diffusion.util import instantiate_from_config

    config_file = hf_hub_download(repo_id="ShandaAI/FlowSep-hive", filename="config.yaml")
    model_file = hf_hub_download(repo_id="ShandaAI/FlowSep-hive", filename="flowsep_hive.ckpt")

    configs = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    configs["model"]["params"]["first_stage_config"]["params"]["reload_from_ckpt"] = None

    preprocessor = FlowSepPreprocessor(configs)

    model = instantiate_from_config(configs["model"]).to(device)
    try:
        ckpt = torch.load(model_file, map_location=device, weights_only=False)["state_dict"]
    except TypeError:
        ckpt = torch.load(model_file, map_location=device)["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    _flowsep_model = model
    _flowsep_preprocessor = preprocessor
    return model, preprocessor


AUDIOSEP_SR = 32000
FLOWSEP_CHUNK_IN = 163840
FLOWSEP_CHUNK_OUT = 160000
FLOWSEP_SR = 16000


def separate_audiosep(audio_path, text):
    device = get_runtime_device()
    model = load_audiosep()
    mixture, _ = librosa.load(audio_path, sr=AUDIOSEP_SR, mono=True)
    input_len = mixture.shape[0]

    with torch.no_grad():
        conditions = model.query_encoder.get_query_embed(
            modality='text', text=[text], device=device
        )
        input_dict = {
            "mixture": torch.Tensor(mixture)[None, None, :].to(device),
            "condition": conditions,
        }
        if input_len > AUDIOSEP_SR * 10:
            sep_audio = model.ss_model.chunk_inference(input_dict)
            sep_audio = sep_audio.squeeze()
        else:
            sep_segment = model.ss_model(input_dict)["waveform"]
            sep_audio = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
        sep_audio = sep_audio[:input_len]

    return (AUDIOSEP_SR, sep_audio)


def _flowsep_process_chunk(model, preprocessor, chunk_wav, text):
    device = get_runtime_device()
    chunk_wav = preprocessor.preprocess_chunk(chunk_wav)
    if len(chunk_wav) < FLOWSEP_CHUNK_IN:
        pad = np.zeros(FLOWSEP_CHUNK_IN - len(chunk_wav), dtype=np.float32)
        chunk_wav = np.concatenate([chunk_wav, pad])
    chunk_wav = chunk_wav[:FLOWSEP_CHUNK_IN]
    mixed_mel, stft = preprocessor.wav_feature_extraction(chunk_wav.reshape(1, -1))
    batch = {
        "fname": ["temp"],
        "text": [text],
        "caption": [text],
        "waveform": torch.rand(1, 1, FLOWSEP_CHUNK_IN).to(device),
        "log_mel_spec": torch.rand(1, 1024, 64).to(device),
        "sampling_rate": torch.tensor([FLOWSEP_SR]).to(device),
        "label_vector": torch.rand(1, 527).to(device),
        "stft": torch.rand(1, 1024, 512).to(device),
        "mixed_waveform": torch.from_numpy(chunk_wav.reshape(1, 1, FLOWSEP_CHUNK_IN)).to(device),
        "mixed_mel": mixed_mel.reshape(1, mixed_mel.shape[0], mixed_mel.shape[1]).to(device),
    }
    result = model.generate_sample(
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


def separate_flowsep(audio_path, text):
    device = get_runtime_device()
    model, preprocessor = load_flowsep()
    full_wav = preprocessor.load_full_audio(audio_path)
    input_len = full_wav.shape[0]

    with torch.no_grad():
        if input_len <= FLOWSEP_CHUNK_IN:
            sep_audio = _flowsep_process_chunk(model, preprocessor, full_wav.copy(), text)
        else:
            out_list = []
            start = 0
            while start < input_len:
                end = min(start + FLOWSEP_CHUNK_IN, input_len)
                chunk = full_wav[start:end]
                out_chunk = _flowsep_process_chunk(model, preprocessor, chunk.copy(), text)
                need = min(FLOWSEP_CHUNK_OUT, input_len - start)
                out_list.append(out_chunk[:need])
                start += FLOWSEP_CHUNK_OUT
            sep_audio = np.concatenate(out_list)

        if len(sep_audio) > input_len:
            sep_audio = sep_audio[:input_len]
        elif len(sep_audio) < input_len:
            sep_audio = np.pad(sep_audio, (0, input_len - len(sep_audio)), mode="constant", constant_values=0)

    return (FLOWSEP_SR, sep_audio)


def inference(audio, text, model_choice):
    if audio is None:
        raise gr.Error("Please upload an audio file / 请上传音频文件")
    if not text or not text.strip():
        raise gr.Error("Please enter a text query / 请输入文本描述")

    if model_choice == "AudioSep-hive":
        return separate_audiosep(audio, text)
    else:
        return separate_flowsep(audio, text)


if spaces is not None:
    @spaces.GPU(duration=120)
    def inference_entry(audio, text, model_choice):
        return inference(audio, text, model_choice)
else:
    def inference_entry(audio, text, model_choice):
        return inference(audio, text, model_choice)


DESCRIPTION = """
# Universal Sound Separation on HIVE

**Hive** is a high-quality synthetic dataset (2k hours) built via an automated pipeline that mines high-purity single-event segments and synthesizes semantically consistent mixtures. Despite using only ~0.2% of the data scale of million-hour baselines, models trained on Hive achieve competitive separation accuracy and strong zero-shot generalization.

This space provides two separation models trained on Hive:
- **AudioSep**: A foundation model for open-domain sound separation with natural language queries, based on [AudioSep](https://github.com/Audio-AGI/AudioSep).
- **FlowSep**: A flow-matching based separation model with text conditioning, based on [FlowSep](https://github.com/Audio-AGI/FlowSep).

**How to use:**
1. Upload an audio file (mix of sounds)
2. Describe what you want to separate (e.g., "piano", "speech", "dog barking")
3. Select a model and click Separate

[[Paper]](https://arxiv.org/abs/2601.22599) | [[Code]](https://github.com/ShandaAI/Hive) | [[Hive Dataset]](https://huggingface.co/datasets/ShandaAI/Hive) | [[Demo Page]](https://shandaai.github.io/Hive/)
"""

EXAMPLES = [
    ["examples/acoustic_guitar.wav", "acoustic guitar"],
    ["examples/laughing.wav", "laughing"],
    ["examples/ticktok_piano.wav", "A ticktock sound playing at the same rhythm with piano"],
    ["examples/water_drops.wav", "water drops"],
    ["examples/noisy_speech.wav", "speech"],
]

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Universal Sound Separation on HIVE",
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Input Mixture Audio", type="filepath")
            text_input = gr.Textbox(
                label="Text Query",
                placeholder='e.g. "dog barking", "piano playing"',
            )
            model_choice = gr.Dropdown(
                choices=["AudioSep-hive", "FlowSep-hive"],
                value="AudioSep-hive",
                label="Select Model",
            )
            submit_btn = gr.Button("Separate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Separated Audio")

    submit_btn.click(
        fn=inference_entry,
        inputs=[audio_input, text_input, model_choice],
        outputs=audio_output,
    )

    gr.Markdown("## Examples")
    gr.Examples(examples=EXAMPLES, inputs=[audio_input, text_input])

DEBUG = False

def run_debug():
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    test_path = os.path.join(examples_dir, "acoustic_guitar.wav")
    test_text = "acoustic guitar"
    print("\n" + "=" * 50)
    print("[DEBUG] Starting inference test for both models")
    print("=" * 50)

    if not os.path.exists(test_path):
        print(f"[DEBUG] Skip: {test_path} not found")
        return

    print(f"\n[DEBUG] Using test audio: {test_path}")

    print("\n" + "-" * 40)
    print("[DEBUG] AudioSep inference")
    print("-" * 40)
    print("[DEBUG] Loading AudioSep model...")
    out_audiosep = separate_audiosep(test_path, test_text)
    print(f"[DEBUG] AudioSep done. Output sr={out_audiosep[0]}, shape={np.array(out_audiosep[1]).shape}")

    print("\n" + "-" * 40)
    print("[DEBUG] FlowSep inference")
    print("-" * 40)
    print("[DEBUG] Loading FlowSep model...")
    out_flowsep = separate_flowsep(test_path, test_text)
    print(f"[DEBUG] FlowSep done. Output sr={out_flowsep[0]}, shape={np.array(out_flowsep[1]).shape}")

    print("\n" + "=" * 50)
    print("[DEBUG] Both models passed inference test")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    if DEBUG:
        run_debug()
    demo.queue()
    demo.launch()
