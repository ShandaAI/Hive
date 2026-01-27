import json
import os
import argparse
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

def cut_audio(audio_path, seg_folder_path, energy_threshold=0.0005):
    os.makedirs(seg_folder_path, exist_ok=True)
    
    try:
        data, samplerate = sf.read(audio_path)
    except Exception as e:
        print(f"Cannot read audio {audio_path}: {e}")
        return []
    
    duration = len(data) / samplerate
    
    if duration <= 10:
        return [audio_path]
    
    file_name = Path(audio_path).stem
    file_ext = Path(audio_path).suffix
    
    window_size = 10 * samplerate
    step_size = 5 * samplerate
    
    segments_paths = []
    chunk_index = 1
    
    for start in range(0, len(data), step_size):
        end = min(start + window_size, len(data))
        segment = data[start:end]
        
        if end == len(data) and (end - start) / samplerate < 5:
            break
        
        if len(segment.shape) > 1:
            energy = np.mean(np.sum(segment**2, axis=1)) / len(segment)
        else:
            energy = np.sum(segment**2) / len(segment)
            
        if energy < energy_threshold:
            continue
            
        output_filename = f"{file_name}_chunk{chunk_index}{file_ext}"
        output_path = os.path.join(seg_folder_path, output_filename)
        sf.write(output_path, segment, samplerate)
        
        segments_paths.append(output_path)
        chunk_index += 1
        
        if end == len(data):
            break
    
    return segments_paths

def main():
    parser = argparse.ArgumentParser(description="Audio cutting tool")
    parser.add_argument("--input_path", required=True, help="Input JSON file path")
    parser.add_argument("--output_path", required=True, help="Output JSON file directory")
    parser.add_argument("--seg_output_path", required=True, help="Parent directory for cut audio")
    parser.add_argument("--energy_threshold", type=float, default=0.0005, help="Energy threshold for filtering empty segments")
    
    args = parser.parse_args()
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    input_json_name = Path(args.input_path).stem
    
    output_json_path = os.path.join(args.output_path, f"{input_json_name}_cut.json")
    
    seg_folder_name = f"{input_json_name}_seg"
    seg_folder_path = os.path.join(args.seg_output_path, seg_folder_name)
    os.makedirs(seg_folder_path, exist_ok=True)
    
    new_data = []
    
    print(f"Start processing {len(data)} audio files...")
    for item in tqdm(data, desc="Processing audio files"):
        audio_path = item["audio_path"]
        
        segment_paths = cut_audio(audio_path, seg_folder_path, args.energy_threshold)
        
        if not segment_paths or segment_paths[0] == audio_path:
            new_data.append(item)
        else:
            for seg_path in segment_paths:
                new_item = {
                    "text_label": item["text_label"],
                    "audio_path": seg_path,
                    "captions": item["captions"]
                }
                new_data.append(new_item)
    
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving result JSON file {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    
    print(f"Processing complete! Processed {len(data)} audios, generated {len(new_data)} results.")
    print(f"Output JSON file: {output_json_path}")
    print(f"Cut audio saved in: {seg_folder_path}")

if __name__ == "__main__":
    main()
