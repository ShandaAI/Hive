import soundfile as sf
import logging
logging.basicConfig(level=logging.ERROR)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

import time
import json
from tqdm import tqdm

import os
import argparse

def process_text(texts: list[str], length: int):
    results = []
    for text in texts:
        parts = text.split('assistant\n')
        if len(parts) != 2:
            return None

        if parts[1].startswith('True'):
            results.append(True)
        elif parts[1].startswith('False'):
            results.append(False)
        else:
            return None
    if len(results) != length:
        return None
    return results

def batch_query(audios: list[str], model, processor):
    conversations = []
    for audio in audios:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Please analyze this audio and determine if it contains only one type of sound. After analysis, please return only True or False, where True means the audio contains only one type of sound, and False means the audio contains multiple types of sounds. Here is your response:"}
                ],
            }
        ]
        conversations.append(conversation)

    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)

    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return process_text(text, len(audios))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audios_path', type=str, required=True, help='Audio file path list')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for inference')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum retries per batch')
    parser.add_argument('--output_path', type=str, default=None, help='Result output path')
    parser.add_argument('--error_output_path', type=str, default=None, help='Error result output path')
    args = parser.parse_args()
    
    audios_path = args.audios_path
    model_path = args.model_path
    batch_size = args.batch_size
    max_retries = args.max_retries
    output_path = args.output_path
    error_output_path = args.error_output_path
    
    if output_path is None:
        input_filename = os.path.basename(audios_path).split('.')[0]
        output_path = f"{input_filename}.json"
    
    if error_output_path is None:
        input_filename = os.path.basename(audios_path).split('.')[0]
        error_output_path = f"{input_filename}_error.json"

    audio_datas = []
    audio_paths = []
    print("Loading audio data...")
    with open(audios_path, 'r') as f:
        audio_datas = json.load(f)
        for audio_data in audio_datas:
            audio_paths.append(audio_data['audio_path'])
    print(f"Audio data loaded, total {len(audio_paths)} audios")

    print(f"Loading model {model_path}...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    print("Model loaded")
    
    single_sound_data = []
    multi_sound_data = []
    total_batches = (len(audio_paths) + batch_size - 1) // batch_size
    batch_counter = 0
    
    batch_indices_list = [list(range(i, min(i+batch_size, len(audio_paths)))) for i in range(0, len(audio_paths), batch_size)]
    
    for batch_indices in tqdm(batch_indices_list, desc="Processing batches", unit="batch"):
        batch_audios = [audio_paths[i] for i in batch_indices]
        batch_results = None
        retry_count = 0
        
        batch_counter += 1
        
        while batch_results is None and retry_count < max_retries:
            if retry_count > 0:
                tqdm.write(f"Retry {retry_count}")
            
            try:
                batch_results = batch_query(batch_audios, model, processor)
            except Exception as e:
                tqdm.write(f"Error occurred: {e}")
                if "CUDA out of memory" in str(e):
                    if len(batch_audios) > 1:
                        tqdm.write(f"CUDA out of memory, reducing batch_size and retrying")
                        mid_point = len(batch_audios) // 2
                        
                        try:
                            first_half_results = batch_query(batch_audios[:mid_point], model, processor)
                            
                            second_half_results = batch_query(batch_audios[mid_point:], model, processor)
                            
                            if first_half_results is not None and second_half_results is not None:
                                batch_results = first_half_results + second_half_results
                                continue
                        except Exception as sub_e:
                            tqdm.write(f"Batch split still failed: {sub_e}")
                
                batch_results = None
            
            retry_count += 1
        
        if batch_results is None:
            tqdm.write(f"Batch {batch_counter} failed, reached maximum retries {max_retries}")
            continue
        
        for idx, result in zip(batch_indices, batch_results):
            if result is True:
                single_sound_data.append(audio_datas[idx])
            else:
                multi_sound_data.append(audio_datas[idx])
        
        if batch_counter % 5 == 0:
            with open(output_path, 'w') as f:
                json.dump(single_sound_data, f, indent=2)
            with open(error_output_path, 'w') as f:
                json.dump(multi_sound_data, f, indent=2)
    
    with open(output_path, 'w') as f:
        json.dump(single_sound_data, f, indent=2)
    with open(error_output_path, 'w') as f:
        json.dump(multi_sound_data, f, indent=2)
    
    print(f"Processing complete, processed {len(audio_paths)} audios, found {len(single_sound_data)} single-sound samples, {len(multi_sound_data)} multi-sound samples")

if __name__ == "__main__":
    main()
    