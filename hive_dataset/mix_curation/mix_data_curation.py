import json
import argparse
import os
import tarfile
import io
import tempfile
import requests
import shutil
import atexit
import base64
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict
from queue import Queue, Empty
from threading import Lock

TEMP_AUDIO_DIR = "/tmp/audio_curation"

def cleanup_temp_dir():
    if os.path.exists(TEMP_AUDIO_DIR):
        try:
            shutil.rmtree(TEMP_AUDIO_DIR)
            print(f"Cleaned up temp directory: {TEMP_AUDIO_DIR}")
        except Exception as e:
            print(f"Failed to clean up temp directory: {e}")

atexit.register(cleanup_temp_dir)

class OntologyTree:
    def __init__(self, ontology_path):
        try:
            with open(ontology_path, 'r', encoding='utf-8') as f:
                self.nodes = json.load(f)
            
            self.id_to_node = {node['id']: node for node in self.nodes}
            self.name_to_id = {node['name']: node['id'] for node in self.nodes}
            self.id_to_parent = {}
            
            for node in self.nodes:
                for child_id in node.get('child_ids', []):
                    self.id_to_parent[child_id] = node['id']
            
            print(f"OntologyTree initialized: loaded {len(self.nodes)} nodes, {len(self.id_to_node)} entries in id_to_node, {len(self.name_to_id)} entries in name_to_id")
        except Exception as e:
            print(f"Failed to load ontology file: {e}")
            raise
    
    def get_siblings(self, label_input):
        label_id = self.get_label_id(label_input)
        if not label_id:
            return []
        
        if label_id not in self.id_to_parent:
            return []
        
        parent_id = self.id_to_parent[label_id]
        parent_node = self.id_to_node.get(parent_id)
        if not parent_node:
            return []
        
        siblings = []
        for sibling_id in parent_node.get('child_ids', []):
            if sibling_id != label_id:
                sibling_node = self.id_to_node.get(sibling_id)
                if sibling_node:
                    siblings.append(sibling_node['name'])
        
        return siblings
    
    def get_label_name(self, label_input):
        if not label_input:
            return None
        
        label_str = str(label_input).strip()
        
        if label_str.startswith('/m/'):
            node = self.id_to_node.get(label_str)
            if node:
                return node['name']
        else:
            if label_str in self.name_to_id:
                return label_str
        
        return None
    
    def get_label_id(self, label_input):
        if not label_input:
            return None
        
        label_str = str(label_input).strip()
        
        if label_str.startswith('/m/'):
            if label_str in self.id_to_node:
                return label_str
        else:
            return self.name_to_id.get(label_str)
        
        return None

def read_tar_samples(tar_path):
    samples = {}
    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                name = member.name
                try:
                    if name.endswith('.json'):
                        sample_id = name.split('.')[0]
                        if sample_id not in samples:
                            samples[sample_id] = {}
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            samples[sample_id]['metadata'] = json.load(file_obj)
                    elif name.endswith('.wav'):
                        if '.mix.wav' in name:
                            sample_id = name.replace('.mix.wav', '')
                            if sample_id not in samples:
                                samples[sample_id] = {}
                            samples[sample_id]['mix_path'] = member.name
                            file_obj = tar.extractfile(member)
                            if file_obj:
                                samples[sample_id]['mix_data'] = file_obj.read()
                        elif '.s' in name and name.endswith('.wav'):
                            parts = name.split('.s')
                            if len(parts) == 2:
                                sample_id = parts[0]
                                source_part = parts[1]
                                source_idx = source_part.split('.')[0]
                                if sample_id not in samples:
                                    samples[sample_id] = {}
                                if 'sources' not in samples[sample_id]:
                                    samples[sample_id]['sources'] = {}
                                file_obj = tar.extractfile(member)
                                if file_obj:
                                    samples[sample_id]['sources'][source_idx] = file_obj.read()
                except Exception as e:
                    print(f"Error reading tar member {name}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading tar file {tar_path}: {e}")
        raise
    
    return samples

def call_qwen3_omni(audio_data, label_pool, api_url, max_retries=3):
    label_pool_dict = {str(i+1): label for i, label in enumerate(label_pool)}
    label_pool_json = json.dumps(label_pool_dict, ensure_ascii=False, indent=2)
    
    prompt = f"""You are an audio classification expert. Please listen to this audio carefully, then select the most matching label from the following label pool.

Label pool (JSON format, key is number, value is label name):
{label_pool_json}

Please only return the corresponding number key (e.g., 1, 2, 3, etc.), do not return anything else. If the audio does not match any label, return 0.

Your answer (only return number):"""
    
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    audio_data_url = f"data:audio/wav;base64,{audio_base64}"
    
    for attempt in range(max_retries):
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                        {"type": "text", "text": prompt}
                    ]}
                ]
            }
            
            response = requests.post(api_url, json=payload, timeout=60)
            
            if response.status_code != 200:
                error_detail = f"Status code: {response.status_code}"
                try:
                    error_body = response.text[:500]
                    error_detail += f", Response: {error_body}"
                except:
                    pass
                print(f"API call error (attempt {attempt + 1}/{max_retries}): {error_detail}")
                print(f"Label pool: {label_pool}")
                if attempt < max_retries - 1:
                    continue
                return None
            
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                answer = result['choices'][0]['message']['content'].strip()
                
                answer_normalized = answer.strip('"').strip("'").strip().strip('.').strip()
                
                try:
                    key = int(answer_normalized)
                    if key == 0:
                        return None
                    elif str(key) in label_pool_dict:
                        return label_pool_dict[str(key)]
                    else:
                        print(f"Model output key {key} not in label pool range, label pool keys: {list(label_pool_dict.keys())}")
                        if attempt < max_retries - 1:
                            continue
                        return None
                except ValueError:
                    print(f"Model output is not a valid number: {answer_normalized}")
                    if attempt < max_retries - 1:
                        continue
                    return None
            else:
                print(f"Abnormal API response format: {result}")
                if attempt < max_retries - 1:
                    continue
                return None
        except requests.exceptions.RequestException as e:
                error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_body = e.response.text[:500]
                    error_msg += f", Response: {error_body}"
                except:
                    pass
            print(f"API call exception (attempt {attempt + 1}/{max_retries}): {error_msg}")
            if attempt < max_retries - 1:
                continue
            return None
        except Exception as e:
            print(f"API call exception (attempt {attempt + 1}/{max_retries}): {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            if attempt < max_retries - 1:
                continue
            return None
    
    return None

def score_source(audio_data, true_label_id, ontology_tree, api_url):
    true_label_name = ontology_tree.get_label_name(true_label_id)
    if not true_label_name:
        return 0.0, None
    
    siblings = ontology_tree.get_siblings(true_label_id)
    if not siblings:
        return 1.0, true_label_name
    
    label_pool = [true_label_name] + siblings
    
    predicted_label = call_qwen3_omni(audio_data, label_pool, api_url)
    
    if predicted_label == true_label_name:
        return 1.0, predicted_label
    else:
        return 0.0, predicted_label

def score_sample(sample_data, ontology_tree, api_url):
    metadata = sample_data.get('metadata', {})
    labels = metadata.get('labels', [])
    sources = sample_data.get('sources', {})
    
    if not labels or not sources:
        return 0.0, [], []
    
    scores = []
    source_labels = []
    model_selected_labels = []
    
    for i, label_id in enumerate(labels, 1):
        source_idx = str(i)
        if source_idx in sources:
            true_label_name = ontology_tree.get_label_name(label_id)
            if true_label_name:
                source_labels.append(true_label_name)
            
            source_score, predicted_label = score_source(sources[source_idx], label_id, ontology_tree, api_url)
            scores.append(source_score)
            model_selected_labels.append(predicted_label if predicted_label else "")
    
    if not scores:
        return 0.0, [], []
    
    avg_score = sum(scores) / len(scores)
    return avg_score, source_labels, model_selected_labels

def update_score_json(score_file, new_records):
    try:
        if os.path.exists(score_file):
            with open(score_file, 'r', encoding='utf-8') as f:
                existing_records = json.load(f)
        else:
            existing_records = []
        
        existing_records.extend(new_records)
        
        with open(score_file, 'w', encoding='utf-8') as f:
            json.dump(existing_records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to update score.json file: {e}")
        raise

def load_tar_file_wrapper(args):
    tar_path, tar_file, nmix_dir = args
    try:
        samples = read_tar_samples(tar_path)
        tar_samples = []
        for sample_id, sample_data in samples.items():
            sample_data['tar_file'] = tar_file
            sample_data['sample_id'] = sample_id
            tar_samples.append(sample_data)
        return tar_file, tar_samples
    except Exception as e:
        print(f"Error reading tar file {tar_file}: {e}")
        return tar_file, None

class TarFileCache:
    def __init__(self, max_size=3):
        self.cache_queue = Queue(maxsize=max_size)
        self.max_size = max_size
        self.lock = Lock()
        self.loading_set = set()
    
    def put(self, tar_file, tar_samples):
        if tar_samples is not None:
            try:
                self.cache_queue.put_nowait((tar_file, tar_samples))
                print(f"[Cache] Loaded and cached: {tar_file} (cache size: {self.cache_queue.qsize()}/{self.max_size})")
            except:
                print(f"[Cache] Cache full, cannot add: {tar_file}")
    
    def get(self, timeout=1):
        try:
            result = self.cache_queue.get(timeout=timeout)
            print(f"[Cache] Retrieved from cache: {result[0]} (cache size: {self.cache_queue.qsize()}/{self.max_size})")
            return result
        except Empty:
            return None
    
    def is_full(self):
        return self.cache_queue.qsize() >= self.max_size
    
    def size(self):
        return self.cache_queue.qsize()

def process_nmix_directory(nmix_dir, ontology_tree, filtered_nmix_dir, api_url, save_score, score_file, max_workers=8, relative_tar_path=None):
    all_tar_files = [f for f in os.listdir(nmix_dir) if f.endswith('.tar')]
    all_tar_files.sort()
    
    os.makedirs(filtered_nmix_dir, exist_ok=True)
    
    existing_tar_files = set()
    if os.path.exists(filtered_nmix_dir):
        existing_tar_files = {f for f in os.listdir(filtered_nmix_dir) if f.endswith('.tar')}
    
    tar_files = [f for f in all_tar_files if f not in existing_tar_files]
    skipped_count = len(all_tar_files) - len(tar_files)
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} existing tar files")
    
    cache = TarFileCache(max_size=3)
    tar_file_index = 0
    
    with ProcessPoolExecutor(max_workers=3) as load_executor, ThreadPoolExecutor(max_workers=max_workers) as score_executor:
        load_futures = {}
        score_futures = {}
        
        def load_next_tar():
            nonlocal tar_file_index
            total_occupied = cache.size() + len(load_futures)
            if tar_file_index < len(tar_files) and total_occupied < cache.max_size:
                tar_file = tar_files[tar_file_index]
                tar_path = os.path.join(nmix_dir, tar_file)
                tar_file_index += 1
                print(f"[Cache] Start loading tar file: {tar_file} (started: {tar_file_index}/{len(tar_files)}, cache: {cache.size()}, loading: {len(load_futures)})")
                future = load_executor.submit(load_tar_file_wrapper, (tar_path, tar_file, nmix_dir))
                load_futures[tar_file] = future
            elif total_occupied >= cache.max_size:
                print(f"[Cache] Total occupied full, pause loading (cache: {cache.size()}, loading: {len(load_futures)}, total: {total_occupied}/{cache.max_size})")
        
        for _ in range(min(3, len(tar_files))):
            load_next_tar()
        
        processed_count = 0
        
        with tqdm(total=len(tar_files), desc=f"Processing {os.path.basename(nmix_dir)}") as pbar:
            while processed_count < len(tar_files):
                for tar_file, future in list(load_futures.items()):
                    if future.done():
                        try:
                            loaded_file, tar_samples = future.result()
                            if tar_samples:
                                cache.put(loaded_file, tar_samples)
                        except Exception as e:
                            print(f"Failed to load tar file {tar_file}: {e}")
                        del load_futures[tar_file]
                        load_next_tar()
                
                if cache.size() == 0 and len(load_futures) == 0 and tar_file_index >= len(tar_files):
                    print(f"[Cache] All tar files processed")
                    break
                
                cached_item = cache.get(timeout=0.1)
                if cached_item:
                    tar_file, tar_samples = cached_item
                    
                    load_next_tar()
                    
                    if not tar_samples:
                        processed_count += 1
                        pbar.update(1)
                        continue
                    
                    print(f"  Processing {tar_file}, {len(tar_samples)} samples")
                    
                    tar_sample_scores = []
                    sample_futures = []
                    
                    for sample_data in tar_samples:
                        future = score_executor.submit(score_sample, sample_data, ontology_tree, api_url)
                        sample_futures.append((sample_data, future))
                    
                    for sample_data, future in tqdm(sample_futures, desc=f"Scoring {tar_file}", leave=False):
                        try:
                            score, source_labels, model_selected_labels = future.result()
                            tar_sample_scores.append((sample_data, score, source_labels, model_selected_labels))
                        except Exception as e:
                            print(f"Error scoring sample {sample_data.get('sample_id', 'unknown')}: {e}")
                            continue
                    
                    if not tar_sample_scores:
                        print(f"  {tar_file} has no valid scores, skipping")
                        processed_count += 1
                        pbar.update(1)
                        load_next_tar()
                        continue
                    
                    tar_sample_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    keep_count = len(tar_sample_scores) // 2
                    kept_samples = tar_sample_scores[:keep_count]
                    
                    print(f"  Keeping top {keep_count} samples (out of {len(tar_sample_scores)})")
                    
                    tar_score_records = []
                    for sample_data, score, source_labels, model_selected_labels in tar_sample_scores:
                        tar_path_rel = os.path.join(relative_tar_path, sample_data['tar_file']) if relative_tar_path else sample_data['tar_file']
                        tar_score_records.append({
                            'tar_file': tar_path_rel,
                            'sample_id': sample_data['sample_id'],
                            'score': score,
                            'source_labels': source_labels,
                            'model_selected_labels': model_selected_labels
                        })
                    
                    if save_score and tar_score_records:
                        try:
                            update_score_json(score_file, tar_score_records)
                        except Exception as e:
                            print(f"Failed to save score records for tar file {tar_file}: {e}")
                    
                    output_tar_path = os.path.join(filtered_nmix_dir, tar_file)
                    try:
                        with tarfile.open(output_tar_path, 'w') as tar:
                            for sample_data, score, source_labels, model_selected_labels in kept_samples:
                                sample_id = sample_data['sample_id']
                                
                                try:
                                    if 'mix_data' in sample_data:
                                        tarinfo = tarfile.TarInfo(name=f"{sample_id}.mix.wav")
                                        tarinfo.size = len(sample_data['mix_data'])
                                        tar.addfile(tarinfo, io.BytesIO(sample_data['mix_data']))
                                    
                                    if 'sources' in sample_data:
                                        for source_idx, source_data in sample_data['sources'].items():
                                            tarinfo = tarfile.TarInfo(name=f"{sample_id}.s{source_idx}.wav")
                                            tarinfo.size = len(source_data)
                                            tar.addfile(tarinfo, io.BytesIO(source_data))
                                    
                                    if 'metadata' in sample_data:
                                        metadata_bytes = json.dumps(sample_data['metadata'], ensure_ascii=False).encode('utf-8')
                                        tarinfo = tarfile.TarInfo(name=f"{sample_id}.json")
                                        tarinfo.size = len(metadata_bytes)
                                        tar.addfile(tarinfo, io.BytesIO(metadata_bytes))
                                except Exception as e:
                                    print(f"Error saving sample {sample_id} to tar file: {e}")
                                    continue
                    except Exception as e:
                        print(f"Error saving tar file {tar_file}: {e}")
                    
                    processed_count += 1
                    pbar.update(1)
                    load_next_tar()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset_path", type=str, required=True, help="Source dataset path")
    parser.add_argument("--ontology_path", type=str, required=True, help="Ontology file path")
    parser.add_argument("--filtered_dataset_path", type=str, required=True, help="Filtered dataset save path")
    parser.add_argument("--save_score", action="store_true", help="Whether to save score.json file")
    parser.add_argument("--api_url", type=str, required=True, help="API endpoint URL for audio classification model")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of concurrent processing threads")
    
    args = parser.parse_args()
    
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    print(f"Using temp directory: {TEMP_AUDIO_DIR}")
    
    print("Loading ontology tree...")
    try:
        ontology_tree = OntologyTree(args.ontology_path)
    except Exception as e:
        print(f"Initialization failed: {e}")
        cleanup_temp_dir()
        return
    
    score_file = None
    if args.save_score:
        score_file = os.path.join(os.getcwd(), 'score.json')
        try:
            if os.path.exists(score_file):
                print(f"Found existing score.json file, will append to it")
            else:
                with open(score_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                print(f"Created score.json file")
        except Exception as e:
            print(f"Failed to initialize score.json file: {e}")
            return
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.source_dataset_path, split)
        if not os.path.exists(split_dir):
            continue
        
        filtered_split_dir = os.path.join(args.filtered_dataset_path, split)
        os.makedirs(filtered_split_dir, exist_ok=True)
        
        print(f"\nProcessing {split} dataset...")
        
        for nmix in ['2mix', '3mix', '4mix', '5mix']:
            nmix_dir = os.path.join(split_dir, nmix)
            if not os.path.exists(nmix_dir):
                continue
            
            print(f"\nProcessing {split}/{nmix}...")
            filtered_nmix_dir = os.path.join(filtered_split_dir, nmix)
            relative_tar_path = os.path.join(split, nmix)
            
            process_nmix_directory(nmix_dir, ontology_tree, filtered_nmix_dir, args.api_url, args.save_score, score_file, args.max_workers, relative_tar_path)
    
    if args.save_score:
        print(f"\nAll scores saved to {score_file}")
    
    cleanup_temp_dir()

if __name__ == "__main__":
    main()

