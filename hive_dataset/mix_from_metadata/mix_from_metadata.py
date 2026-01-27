import json
import argparse
import soundfile as sf
import os
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import multiprocessing
import tarfile
import io
import glob
import warnings
import librosa
import time

sr_mismatch_warned = False

def process_json_file(args):
    json_path, prefix_map, output_dir, metadata_dir = args
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        tar_samples = []
        for metadata in metadata_list:
            sample = process_single_metadata(metadata, prefix_map)
            if sample is not None:
                tar_samples.append(sample)
        
        if tar_samples:
            relative_dir = os.path.relpath(os.path.dirname(json_path), metadata_dir)
            json_filename = os.path.basename(json_path)
            tar_filename = json_filename.split('_tar')[-1].replace('.json', '.tar')
            tar_filename = 'tar_' + tar_filename
            tar_path = os.path.join(output_dir, relative_dir, tar_filename)
            os.makedirs(os.path.dirname(tar_path), exist_ok=True)
            
            write_tar_file(tar_path, tar_samples)
            return True
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False
    
    return False

def process_single_metadata(metadata, prefix_map):
    mix_id = None
    try:
        mix_id = metadata['mix_id']
        sample_rate = metadata['dataset_info']['sample_rate']
        target_duration = metadata['dataset_info']['target_duration']
        sources = metadata['sources']
        mixing_params = metadata['mixing_params']
        
        target_length = int(target_duration * sample_rate)
        processed_sources = []
        
        for i, source in enumerate(sources):
            path = source['path']
            path_prefix = path.split('/')[0]
            
            if path_prefix not in prefix_map:
                print(f"Error [mix_id: {mix_id}]: prefix '{path_prefix}' in path '{path}' not found in prefix_map")
                return None
            
            full_path = os.path.join(prefix_map[path_prefix], '/'.join(path.split('/')[1:]))
            
            if not os.path.exists(full_path):
                print(f"Error [mix_id: {mix_id}]: audio file not found: {full_path}")
                return None
            
            with sf.SoundFile(full_path) as f:
                sr = f.samplerate
                total_frames = len(f)
                
                chunk_start = int(source['chunk_start_second'] * sr)
                chunk_end = int(source['chunk_end_second'] * sr)
                
                if chunk_end > total_frames:
                    diff = chunk_end - total_frames
                    if diff <= 10:
                        chunk_end = total_frames
                    else:
                        print(f"Error [mix_id: {mix_id}]: chunk range significantly exceeds audio length, chunk_end: {chunk_end}, length: {total_frames}, file: {full_path}")
                        return None
                
                chunk_start = max(0, min(chunk_start, total_frames))
                
                f.seek(chunk_start)
                chunk_audio = f.read(chunk_end - chunk_start)
            
            if chunk_audio.ndim > 1:
                chunk_audio = np.mean(chunk_audio, axis=1)
            
            if sr != sample_rate:
                global sr_mismatch_warned
                if not sr_mismatch_warned:
                    # print(f"Sample rate mismatch, expected {sample_rate}, got {sr}, resampling, file: {full_path}")
                    sr_mismatch_warned = True
                chunk_audio = librosa.resample(chunk_audio, orig_sr=sr, target_sr=sample_rate)
            
            crop_start = int(source['crop_start_second'] * sample_rate)
            crop_end = int(source['crop_end_second'] * sample_rate)
            
            if crop_end > len(chunk_audio):
                chunk_audio = np.pad(chunk_audio, (0, crop_end - len(chunk_audio)))
            
            cropped_audio = chunk_audio[crop_start:crop_end]
            
            if len(cropped_audio) < target_length:
                cropped_audio = np.pad(cropped_audio, (0, target_length - len(cropped_audio)))
            elif len(cropped_audio) > target_length:
                cropped_audio = cropped_audio[:target_length]
            
            weighted_audio = cropped_audio * source['applied_weight']
            processed_sources.append(weighted_audio)
        
        mixed_audio = np.sum(processed_sources, axis=0)
        mixed_audio = mixed_audio * mixing_params['global_normalization_factor']
        
        return {
            'mix_id': mix_id,
            'mix': mixed_audio,
            'sources': processed_sources,
            'sample_rate': sample_rate,
            'metadata': metadata
        }
    except Exception as e:
        import traceback
        print(f"Error [mix_id: {mix_id}]: exception occurred while processing metadata")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print(f"Detailed traceback:")
        traceback.print_exc()
        return None

def create_wav_bytes(audio_data, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV')
    return buffer.getvalue()

def add_to_tar(tar, filename, data):
    tarinfo = tarfile.TarInfo(name=filename)
    tarinfo.size = len(data)
    tar.addfile(tarinfo, io.BytesIO(data))

def write_tar_file(tar_path, samples):
    with tarfile.open(tar_path, 'w') as tar:
        for sample in samples:
            sample_id = sample['mix_id']
            sample_rate = int(sample['sample_rate'])
            
            mix_data = create_wav_bytes(sample['mix'], sample_rate)
            add_to_tar(tar, f"{sample_id}.mix.wav", mix_data)
            
            for i, source_audio in enumerate(sample['sources']):
                source_data = create_wav_bytes(source_audio, sample_rate)
                add_to_tar(tar, f"{sample_id}.s{i+1}.wav", source_data)
            
            metadata_bytes = json.dumps(sample['metadata'], ensure_ascii=False).encode('utf-8')
            add_to_tar(tar, f"{sample_id}.json", metadata_bytes)

def find_all_json_files(metadata_dir):
    json_files = []
    for root, dirs, files in os.walk(metadata_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def get_tar_path_for_json(json_path, metadata_dir, output_dir):
    relative_dir = os.path.relpath(os.path.dirname(json_path), metadata_dir)
    json_filename = os.path.basename(json_path)
    tar_filename = json_filename.split('_tar')[-1].replace('.json', '.tar')
    tar_filename = 'tar_' + tar_filename
    return os.path.join(output_dir, relative_dir, tar_filename)

def clean_incomplete_tars(output_dir):
    if not os.path.exists(output_dir):
        return
    
    for root, dirs, files in os.walk(output_dir):
        tar_files = [f for f in files if f.endswith('.tar')]
        if not tar_files:
            continue
        
        print(f"Checking directory {root}")
        count = 0
        for tar_file in tar_files:
            tar_path = os.path.join(root, tar_file)
            if os.path.getsize(tar_path) < 1024 * 1024:
                os.remove(tar_path)
                count += 1
        if count > 0:
            print(f"  Removed tar files smaller than 1MB: {count}")
        
        tar_files = [f for f in os.listdir(root) if f.endswith('.tar')]
        if tar_files:
            tar_files_with_num = []
            for tar_file in tar_files:
                try:
                    if tar_file.startswith('tar_'):
                        num_part = tar_file.replace('tar_', '').replace('.tar', '')
                        num = int(num_part)
                        tar_files_with_num.append((num, tar_file))
                except:
                    continue
            
            if tar_files_with_num:
                tar_files_with_num.sort(key=lambda x: x[0], reverse=True)
                max_tar_file = tar_files_with_num[0][1]
                max_tar_path = os.path.join(root, max_tar_file)
                os.remove(max_tar_path)
                print(f"  Removed tar file with largest number: {max_tar_file}")

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./hive_dataset")
    parser.add_argument("--dataset_paths", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=8)
    
    args = parser.parse_args()
    
    with open(args.dataset_paths, 'r') as f:
        dataset_paths = json.load(f)
    
    prefix_map = {os.path.basename(path): path for path in dataset_paths.values()}
    
    print("="*60)
    print("Mixing Configuration:")
    print("="*60)
    print(f"Metadata directory: {args.metadata_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset paths file: {args.dataset_paths}")
    print(f"Number of processes: {args.num_processes}")
    print(f"Number of datasets: {len(dataset_paths)}")
    print("="*60)
    print()
    
    json_files = find_all_json_files(args.metadata_dir)
    print(f"Found {len(json_files)} JSON files")
    
    print("\nCleaning incomplete tar files...")
    clean_incomplete_tars(args.output_dir)
    
    tasks = []
    completed_count = 0
    for json_file in json_files:
        tar_path = get_tar_path_for_json(json_file, args.metadata_dir, args.output_dir)
        if not os.path.exists(tar_path):
            tasks.append((json_file, prefix_map, args.output_dir, args.metadata_dir))
        else:
            completed_count += 1
    
    print(f"Completed {completed_count} files")
    print(f"Need to process {len(tasks)} JSON files")
    
    if len(tasks) == 0:
        print("All files have been processed")
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        print(f"Total mixing time: {hours:02d}:{minutes:02d}:{seconds:06.3f}")
        return
    
    with Pool(processes=args.num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_json_file, tasks),
            total=len(json_files),
            initial=completed_count,
            desc="Processing JSON files"
        ))
    
    success_count = sum(results)
    print(f"Successfully processed {success_count}/{len(tasks)} files")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"Total mixing time: {hours:02d}:{minutes:02d}:{seconds:06.3f}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
