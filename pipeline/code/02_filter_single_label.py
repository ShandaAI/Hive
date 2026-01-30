import json
import os
from tqdm import tqdm

def filter_single(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    result = []
    for item in tqdm(data, desc="Processing audio files"):
        labels = item["text_label"]
        audio_path = item["audio_path"]
        if os.path.exists(audio_path) and (isinstance(labels, str) or len(labels) == 1 or len(labels) == 0):
            result.append(item)
    return result

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for json_file in tqdm(json_files, desc="Processing files"):
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file)
        
        result = filter_single(input_path)
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"{json_file}: Filtered {len(result)} data items")

if __name__ == "__main__":
    input_dir = "path/to/input_dir"
    output_dir = "path/to/output_dir"
    process_directory(input_dir, output_dir)
