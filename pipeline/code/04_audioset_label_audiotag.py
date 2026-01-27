#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Xiaoyu Yang)
# 
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import csv
import json
import logging
import math
from typing import List, Dict
from tqdm.auto import tqdm

import kaldifeat
import torch
import torchaudio
from torchaudio import transforms
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_model, get_params


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )

    parser.add_argument(
        "--label-dict",
        type=str,
        help="""class_labels_indices.csv.""",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the data. "
        "The data to be processed.must be a json file",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output json file",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing audio files",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    add_model_arguments(parser)

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    ans = []
    for f in filenames:
        try:
            wave, sample_rate = torchaudio.load(f)
            if sample_rate != expected_sample_rate:
                logging.info(f"Sample rate mismatch for file {f}: found {sample_rate}, expected {expected_sample_rate}. Resampling...")
                resampler = transforms.Resample(orig_freq=sample_rate, new_freq=expected_sample_rate)
                wave = resampler(wave)
            ans.append(wave[0].contiguous())
        except Exception as e:
            logging.warning(f"Failed to load audio file {f}: {e}")
            ans.append(torch.zeros(1, dtype=torch.float32))
    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()

    params.update(vars(args))

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info("Creating model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    # get the label dictionary
    label_dict = {}
    with open(params.label_dict, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            label_dict[int(row[0])] = row[2]

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading input JSON file: {params.input_path}")
    with open(params.input_path, "r") as f:
        data = json.load(f)

    batch_size = params.batch_size
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    logging.info(f"Total samples: {num_samples}, batch size: {batch_size}, total batches: {num_batches}")

    progress_bar = tqdm(total=num_batches, desc="Processing audio batches", ncols=100, unit="batch", leave=True)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = data[start_idx:end_idx]
        
        audio_paths = [item["audio_path"] for item in batch_data]
        logging.info(f"Processing batch {batch_idx+1}/{num_batches}, samples {start_idx+1}-{end_idx}")
        
        waves = read_sound_files(
            filenames=audio_paths, expected_sample_rate=params.sample_rate
        )
        waves = [w.to(device) for w in waves]

        features = fbank(waves)
        feature_lengths = [f.size(0) for f in features]

        features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
        feature_lengths = torch.tensor(feature_lengths, device=device)

        try:
            encoder_out, encoder_out_lens = model.forward_encoder(features, feature_lengths)
            logits = model.forward_audio_tagging(encoder_out, encoder_out_lens)
        except Exception as e:
            logging.error(f"Inference error, skipping batch {batch_idx+1}: {e}")
            progress_bar.update(1)
            continue

        for i, (item, logit) in enumerate(zip(batch_data, logits)):
            topk_prob, topk_index = logit.sigmoid().topk(5)  
            top_labels = [label_dict[index.item()] for index in topk_index]
            top_probs = [f"{prob.item():.4f}" for prob in topk_prob]
            
            audio_path = item["audio_path"]
            audio_filename = audio_path.split('/')[-1]
            
            top_label = top_labels[0]
            data[start_idx + i]["text_label"] = [top_label]
            
        progress_bar.update(1)
            
        if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
            logging.info(f"Saving intermediate results to {params.output_path} after batch {batch_idx+1}")
            with open(params.output_path, "w") as f:
                json.dump(data, f, indent=2)
    
    progress_bar.close()

    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    
    if hasattr(args, 'output_path') and args.output_path:
        log_file = args.output_path.replace('.json', '.log')
    else:
        log_file = "processing.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format=formatter,
        handlers=[
            logging.FileHandler(log_file)
        ]
    )
    
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
            logging.root.removeHandler(handler)
    
    print(f"Log will be saved to: {log_file}")
    
    main()