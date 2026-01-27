import json
import argparse
import logging
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import os
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = 'logs_qwen_new'
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f'batch_processing_{timestamp}.log')

class CustomLogFilter(logging.Filter):
    def filter(self, record):
        if "System prompt modified, audio output may not work as expected" in record.getMessage():
            return False
        return True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
    ]
)

for handler in logging.getLogger().handlers:
    handler.addFilter(CustomLogFilter())
logger = logging.getLogger(__name__)

class LabelTree:
    def __init__(self, original_ontology_path, modified_ontology_path):
        with open(original_ontology_path, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)
        
        with open(modified_ontology_path, 'r', encoding='utf-8') as f:
            self.modified_data = json.load(f)
        
        self.original_id_to_node = {node['id']: node for node in self.original_data}
        self.modified_id_to_node = {node['id']: node for node in self.modified_data}
        self.original_name_to_id = {node['name']: node['id'] for node in self.original_data}
        self.modified_name_to_id = {node['name']: node['id'] for node in self.modified_data}
        
        self.child_to_parent = {}
        for node in self.original_data:
            for child_id in node.get('child_ids', []):
                self.child_to_parent[child_id] = node['id']
    
    def is_middle_node_by_name(self, node_name):
        if node_name in self.modified_name_to_id:
            node_id = self.modified_name_to_id[node_name]
            node = self.modified_id_to_node[node_id]
            return bool(node.get('child_ids', []))
        return False
    
    def is_leaf_node_by_name(self, node_name):
        if node_name in self.modified_name_to_id:
            node_id = self.modified_name_to_id[node_name]
            node = self.modified_id_to_node[node_id]
            return not bool(node.get('child_ids', []))
        return False
    
    def is_external_node_by_name(self, node_name):
        return node_name not in self.modified_name_to_id
    
    def get_leaf_nodes_by_name(self, node_name):
        if self.is_external_node_by_name(node_name):
            return []
        
        node_id = self.modified_name_to_id[node_name]
        leaf_nodes = []
        
        def collect_leaves(current_id):
            current_node = self.modified_id_to_node[current_id]
            child_ids = current_node.get('child_ids', [])
            
            if not child_ids:
                leaf_nodes.append(current_node['name'])
            else:
                for child_id in child_ids:
                    if child_id in self.modified_id_to_node:
                        collect_leaves(child_id)
        
        collect_leaves(node_id)
        return leaf_nodes
       
    
    def get_root_leaf_node_by_name(self, node_name):
        if not self.is_external_node_by_name(node_name):
            return None
        
        if node_name not in self.original_name_to_id:
            return None
        
        current_id = self.original_name_to_id[node_name]
        
        while current_id:
            if current_id in self.modified_id_to_node:
                current_node = self.modified_id_to_node[current_id]
                if not current_node.get('child_ids', []):
                    return current_node['name']
            
            current_id = self.child_to_parent.get(current_id)
        
        return None

def batch_query(data: list[dict], model, processor, label_tree: LabelTree):
    results = [None] * len(data)
    conversations = []
    
    for index, audio_data in enumerate(data):
        text_label = audio_data['text_label'][0]
        audio = audio_data['audio_path']
        leaf_nodes = label_tree.get_leaf_nodes_by_name(text_label)
        if leaf_nodes == []:
            root_leaf_node = label_tree.get_root_leaf_node_by_name(text_label)
            audio_data_copy = audio_data.copy()
            audio_data_copy['text_label'] = [root_leaf_node]
            results[index] = audio_data_copy
            continue
        if len(leaf_nodes) == 1:
            results[index] = audio_data
            continue
        leaf_labels = leaf_nodes
        prompt = f"Please analyze this audio and determine which type of sound it contains from {leaf_labels}. After analysis, please return only the index of the corresponding label (the minimum index value is 0). Please note that you should only return a single number, without any other symbols. At the same time, if you believe the content of the audio does not match any of the sounds described above, please return -1. Here is the index value you return:"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": {prompt}}
                ],
            }
        ]
        conversations.append(conversation)

    def process_text(texts: list[str]):
        results = []
        for text in texts:
            parts = text.split('assistant\n')
            if len(parts) != 2:
                return None
            try:
                index = int(parts[1])
                results.append(index)
            except ValueError:
                logger.info(f"Current model output: {texts}")
                return None
        return results
    
    if not conversations:
        return results,[]
        
    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)

    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    middle_results = process_text(text)
    if middle_results is None:
        logger.error("Failed to parse model output, skipping this batch")
        return None, []

    error_set = []
    middle_index = 0
    for index in range(len(results)):
        if results[index] is None:
            audio_data = data[index]
            leaf_nodes = label_tree.get_leaf_nodes_by_name(audio_data['text_label'][0])
            middle_result = middle_results[middle_index]
            middle_index += 1
            if middle_result == -1:
                audio_path = audio_data["audio_path"]
                logger.error(f"Audio {audio_path} label index is -1, skipping this audio")
                audio_data_copy = audio_data.copy()
                audio_data_copy['text_label'] = [None]
                results[index] = audio_data_copy
                error_set.append(audio_data)
                continue
            try:
                audio_data_copy = audio_data.copy()
                audio_data_copy['text_label'] = [leaf_nodes[middle_result]]
                results[index] = audio_data_copy
            except IndexError:
                audio_path = audio_data["audio_path"]
                logger.error(f"Audio {audio_path} label index is {middle_result}, but corresponding leaf node does not exist, skipping this audio")
                audio_data_copy = audio_data.copy()
                audio_data_copy['text_label'] = [None]
                results[index] = audio_data_copy
                error_set.append(audio_data)
                continue
    return results, error_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology_path", type=str, required=True)
    parser.add_argument("--modified_ontology_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--retry_times", type=int, default=1)
    parser.add_argument("--error_set_path", type=str, required=True)
    args = parser.parse_args()

    ontology_path = args.ontology_path
    modified_ontology_path = args.modified_ontology_path
    data_path = args.data_path
    batch_size = args.batch_size
    output_path = args.output_path
    model_path = args.model_path
    retry_times = args.retry_times
    error_set_path = args.error_set_path

    label_tree = LabelTree(ontology_path, modified_ontology_path)
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loading model {model_path}...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    logger.info("Model loaded")

    total_batches = (len(data) + batch_size - 1) // batch_size
    cached_results = []
    processed_count = 0
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data))
        batch_data = data[start_idx:end_idx]
        
        try:
            batch_results, error_set = batch_query(batch_data, model, processor, label_tree)
            
            if batch_results is None:
                logger.info(f"Batch {batch_idx + 1} failed, starting retry")
                for retry_idx in range(retry_times):
                    logger.info(f"Batch {batch_idx + 1} retry {retry_idx + 1}")
                    batch_results, error_set = batch_query(batch_data, model, processor, label_tree)
                    if batch_results is not None:
                        break
                
                if batch_results is None:
                    logger.info(f"Batch {batch_idx + 1} retry failed, skipping this batch")
                    continue
            
            if len(error_set) > 0:
                with open(error_set_path, 'a', encoding='utf-8') as f:
                    for error_audio in error_set:
                        f.write(json.dumps(error_audio, ensure_ascii=False, indent=2) + '\n')
                logger.info(f"Saved {len(error_set)} error audios to {error_set_path}")
            
            batch_results = [result for result in batch_results if result['text_label'][0] is not None and result['text_label'] != []]
            
            if len(batch_results) == 0:
                logger.error(f"All audios in batch {batch_idx + 1} have invalid 'text_label' values, skipping this batch")
                continue
            cached_results.extend(batch_results)
            processed_count += 1
            
            if processed_count % 20 == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(cached_results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(cached_results)} processing results to {output_path}")
                
        except Exception as e:
            logger.error(f"Error occurred during batch {batch_idx + 1} processing: {str(e)}, skipping this batch")
            continue
    
    if cached_results:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cached_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Processing complete, saved {len(cached_results)} results to {output_path}")
