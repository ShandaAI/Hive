import json
import argparse
import os
import torch
import numpy as np
import librosa
import soundfile as sf
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import time
import logging

import look2hear.models
from ml_collections import ConfigDict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_process.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_audio(file_path):
    audio, samplerate = librosa.load(file_path, mono=True, sr=44100)
    audio = audio[None, :]
    return torch.from_numpy(audio), samplerate

def get_config(config_path):
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
        return config

def _getWindowingArray(window_size, fade_size):
    fadein = torch.linspace(1, 1, fade_size)
    fadeout = torch.linspace(0, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

def prepare_audio_chunks(audio_path):
    try:
        test_data, samplerate = load_audio(audio_path)
        C = 10 * samplerate
        N = 2
        step = C // N
        fade_size = 3 * 44100
        
        border = C - step
        
        if len(test_data.shape) == 1:
            test_data = test_data.unsqueeze(0)
        
        original_length = test_data.shape[1]
        
        if test_data.shape[1] > 2 * border and (border > 0):
            test_data = torch.nn.functional.pad(test_data, (border, border), mode='reflect')
        
        chunks = []
        chunk_positions = []
        
        i = 0
        while i < test_data.shape[1]:
            part = test_data[:, i:i + C]
            length = part.shape[-1]
            if length < C:
                if length > C // 2 + 1:
                    part = torch.nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                else:
                    part = torch.nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
            
            chunks.append(part)
            chunk_positions.append((i, length))
            i += step
        
        return chunks, chunk_positions, test_data.shape, original_length, samplerate, C, step, fade_size, border
        
    except Exception as e:
        logger.error(f"Error preparing audio chunks {audio_path}: {e}")
        return None, None, None, None, None, None, None, None, None

def process_data_batch(audio_items, model, output_dir):
    results = []
    
    all_chunks = []
    audio_infos = []
    
    valid_items = []
    for item in audio_items:
        audio_path = item["audio_path"]
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            continue
        valid_items.append(item)
    
    if not valid_items:
        return results
    
    max_workers = min(len(valid_items), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(prepare_audio_chunks, item["audio_path"]): item 
                         for item in valid_items}
        
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                chunks, positions, shape, orig_len, sr, C, step, fade_size, border = future.result()
                if chunks is None:
                    continue
                    
                audio_infos.append({
                    'item': item,
                    'chunks_start_idx': len(all_chunks),
                    'chunks_count': len(chunks),
                    'positions': positions,
                    'shape': shape,
                    'original_length': orig_len,
                    'samplerate': sr,
                    'C': C,
                    'step': step,
                    'fade_size': fade_size,
                    'border': border
                })
                
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing audio file {item['audio_path']}: {e}")
                continue
    
    if not all_chunks:
        return results
    
    try:
        batch_chunks = torch.stack(all_chunks).to(device)
        with torch.no_grad():
            batch_outputs = model(batch_chunks).squeeze(1).cpu()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"CUDA out of memory: {e}")
            torch.cuda.empty_cache()
            return "CUDA_OOM"
        else:
            logger.error(f"Error during batch inference: {e}")
            return results
    except Exception as e:
        logger.error(f"Error during batch inference: {e}")
        return results
    
    chunk_idx = 0
    for audio_info in audio_infos:
        try:
            item = audio_info['item']
            audio_path = item["audio_path"]
            chunks_count = audio_info['chunks_count']
            positions = audio_info['positions']
            shape = audio_info['shape']
            original_length = audio_info['original_length']
            samplerate = audio_info['samplerate']
            C = audio_info['C']
            step = audio_info['step']
            fade_size = audio_info['fade_size']
            border = audio_info['border']
            
            audio_outputs = batch_outputs[chunk_idx:chunk_idx + chunks_count]
            chunk_idx += chunks_count
            
            windowingArray = _getWindowingArray(C, fade_size)
            result = torch.zeros((1,) + tuple(shape), dtype=torch.float32)
            counter = torch.zeros((1,) + tuple(shape), dtype=torch.float32)
            
            for idx, (out, (pos_i, length)) in enumerate(zip(audio_outputs, positions)):
                window = windowingArray.clone()
                if idx == 0:
                    window[:fade_size] = 1
                elif pos_i + C >= shape[1]:
                    window[-fade_size:] = 1
                
                result[..., pos_i:pos_i+length] += out[..., :length] * window[..., :length]
                counter[..., pos_i:pos_i+length] += window[..., :length]
            
            final_output = result / counter
            final_output = final_output.squeeze(0).numpy()
            np.nan_to_num(final_output, copy=False, nan=0.0)
            
            if shape[1] > 2 * border and (border > 0):
                final_output = final_output[..., border:-border]
            
            output_filename = os.path.basename(audio_path)
            output_path = os.path.join(output_dir, output_filename)
            sf.write(output_path, final_output.T, samplerate)
            
            item["audio_path"] = output_path
            results.append(item)
            
        except Exception as e:
            logger.error(f"Error reconstructing audio {audio_info['item']['audio_path']}: {e}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--output_audio_dir", type=str, required=True, help="Output audio directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of audio files to process in parallel")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    output_audio_dir = args.output_audio_dir
    batch_size = args.batch_size
    original_batch_size = batch_size

    os.makedirs(output_audio_dir, exist_ok=True)

    apollo_uni_config = get_config('configs/config_apollo_uni.yaml')
    model = look2hear.models.BaseModel.from_pretrain('my_weights/apollo_model_uni.ckpt', **apollo_uni_config['model']).to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    else:
        logger.info(f"Using single GPU: {device}")

    with open(input_path, "r") as f:
        data = json.load(f)
    
    results = []
    processed_count = 0
    
    i = 0
    pbar = tqdm(total=len(data), desc="Batch processing audio files")
    
    while i < len(data):
        batch_data = data[i:i + batch_size]
        batch_results = process_data_batch(batch_data, model, output_audio_dir)
        
        if batch_results == "CUDA_OOM":
            if batch_size == 1:
                logger.error(f"\nError: CUDA out of memory even with batch_size=1, cannot continue")
                logger.error(f"Please try smaller audio files or GPU with more memory")
                return
            
            batch_size = max(1, batch_size // 2)
            logger.warning(f"\nCUDA out of memory detected, reducing batch_size from {original_batch_size} to {batch_size}, retrying...")
            original_batch_size = batch_size
            continue
        
        results.extend(batch_results)
        i += len(batch_data)
        processed_count += len(batch_data)
        pbar.update(len(batch_data))

        if processed_count > 0 and processed_count % 1024 == 0:
            if batch_size < args.batch_size:
                new_batch_size = min(batch_size * 2, args.batch_size)
                logger.info(f"\nProcessed {processed_count} files, attempting to increase batch_size from {batch_size} to {new_batch_size}...")
                batch_size = new_batch_size
                original_batch_size = batch_size
    
    pbar.close()
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    if batch_size != args.batch_size:
        logger.warning(f"\nNote: Due to CUDA memory limitations, actual batch_size used is {batch_size} (initial value: {args.batch_size})")
    
    logger.info(f"Batch processing complete, processed {len(results)} files")
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
