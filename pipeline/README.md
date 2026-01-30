<h1 align="center">Single-Event Data Collection Pipeline</h1>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

<p align="center">
  <strong>Automated Pipeline for High-Purity Single-Event Audio Mining</strong>
</p>

---

## 📌 Table of Contents

- [💡 Overview](#-overview)
- [📂 Repository Structure](#-repository-structure)
- [📁 Data Preparation](#-data-preparation)
- [🛠️ Environment Installation](#️-environment-installation)
- [🚀 Quick Start](#-quick-start)
  - [Step 1: Audio Chunking](#step-1-audio-chunking)
  - [Step 2: Single Label Filtering](#step-2-single-label-filtering)
  - [Step 3: Single Event Filtering](#step-3-single-event-filtering)
  - [Step 4: AudioSet Label Tagging](#step-4-audioset-label-tagging)
  - [Step 5: Leaf Label Classification](#step-5-leaf-label-classification)
  - [Step 6: Audio Super-Resolution](#step-6-audio-super-resolution)

---

## 💡 Overview

The Single-Event Data Collection Pipeline is an automated framework for mining high-purity single-event audio segments from weakly-labeled datasets. It integrates multimodal AI models (Qwen3-Omni) and acoustic tagging (AudioTag) to eliminate event co-occurrence and weak label noise, producing semantically consistent training data for Universal Sound Separation.

**Key Features:**
- Coarse-to-Fine Labeling Strategy: Leverages ontology topology for precise label alignment - first predicting coarse-grained parent nodes via AudioTag, then refining to specific leaf nodes using Qwen3-Omni with restricted candidate subsets
- Multi-stage filtering for semantic-acoustic alignment
- Produces 44.1kHz high-fidelity audio outputs

---

## 📂 Repository Structure

```
pipeline/
├── code/
│   ├── 01_audio_chunking.py               # Chunk long audio into 10s segments with overlap
│   ├── 02_filter_single_label.py          # Filter samples with single label
│   ├── 03_filter_single_event_qwen.py     # Qwen3-Omni based single-event audio filtering
│   ├── 04_audioset_label_audiotag.py      # AudioSet ontology tagging using AudioTag model
│   ├── 05_leaf_label_qwen.py              # Leaf-level label refinement with Qwen3-Omni
│   └── 06_superres_apollo.py              # Audio super-resolution to 44.1kHz using Apollo
├── data/
│   ├── 01_source_data/
│   │   └── example.json                   # Example input format for raw data
│   ├── 02_single_label_data/
│   ├── 03_single_event_data/
│   ├── 04_audiotagged_data/
│   ├── 05_leaf_label_data/
│   └── 06_super_res_data/
├── ontology/
│   ├── audioset_ontology.json              # Original AudioSet ontology
│   └── hive_ontology.json                  # Modified ontology for Hive dataset
├── icefall/                                # AudioTag model repository
├── Apollo/                                 # Apollo model repository
├── requirements.txt                        # Pipeline dependencies
└── README.md                               # This file
```

---

## 📁 Data Preparation

Before running the pipeline, prepare your source audio data in JSON format and place it in the `data/01_source_data/` directory.

**Required JSON Format:**

```json
[
  {
    "text_label": ["Label1", "Label2"],
    "audio_path": "/path/to/your/audio.wav"
  },
  {
    "text_label": "SingleLabel",
    "audio_path": "/path/to/another/audio.wav"
  }
]
```

**Field Requirements:**
- `text_label`: Can be a string (single label) or array of strings (multiple labels)
- `audio_path`: Absolute path to the audio file

**Example:**
See `data/01_source_data/example.json` for reference format.

---

## 🛠️ Environment Installation

```bash
conda create -n hive_pipeline python=3.10
conda activate hive_pipeline
pip install -r requirements.txt
```

**Note:** Steps 4 and 6 require separate environments. Refer to their respective sections for setup instructions.

---

## 🚀 Quick Start

### Step 1: Audio Chunking

Chunk long audio files into 10-second segments with 50% overlap, filtering out low-energy segments.

```bash
python code/01_audio_chunking.py \
    --input_path data/01_source_data/example.json \
    --output_path data/01_source_data/output.json \
    --seg_output_path data/01_source_data/segments \
    --energy_threshold 0.0005
```

**Input JSON Format** (see `data/01_source_data/example.json`):
```json
[
  {
    "text_label": ["Label1", "Label2"],
    "audio_path": "/path/to/audio.wav"
  }
]
```

### Step 2: Single Label Filtering

Filter samples containing only single label.

**Note:** Modify `input_dir` and `output_dir` in the script before running:
- `input_dir`: Path to Step 1 output directory (e.g., `data/01_source_data/`)
- `output_dir`: Output directory (e.g., `data/02_single_label_data/`)

```bash
python code/02_filter_single_label.py
```

**Example Output Format** (see `data/02_single_label_data/example.json`):
```json
[
  {
    "text_label": "Label",
    "audio_path": "/path/to/segments/sample.wav"
  }
]
```
- Samples with multiple labels are filtered out
- `text_label` is now a single string
- Output file: `data/02_single_label_data/output.json`

### Step 3: Single Event Filtering

Use Qwen3-Omni model to filter audio which contains only one type of sound event.

**Model Download:**

Download the Qwen3-Omni model from [Qwen3-Omni-7B Hugging Face](https://huggingface.co/Qwen/Qwen3-Omni-7B)

```bash
python code/03_filter_single_event_qwen.py \
    --model_path /path/to/Qwen3-Omni-7B \
    --audios_path data/02_single_label_data/output.json \
    --output_path data/03_single_event_data/output.json \
    --batch_size 64
```

**Example Output Format** (see `data/03_single_event_data/example.json`):
- Same format as Step 2, only verified single-event samples remain
- Acoustically pure single-event segments pass filtering

### Step 4: AudioSet Label Tagging

Assign AudioSet ontology labels using the AudioTag acoustic tagging model (icefall implementation).

**Environment Setup:**

Refer to the installation instructions in the [icefall repository](https://github.com/k2-fsa/icefall):
- Repository: `icefall/egs/audioset/AT/zipformer/`
- Follow the environment setup guide provided by icefall

**Model Download:**

Download the AudioTag model checkpoint and label dictionary from [Hugging Face](https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12/tree/main):
- Checkpoint: `exp/pretrained.pt`
- Label dictionary: `data/class_labels_indices.csv`

**Setup:**
```bash
cp code/04_audioset_label_audiotag.py icefall/egs/audioset/AT/
cd icefall/egs/audioset/AT/
```

**Run:**
```bash
python 04_audioset_label_audiotag.py \
    --checkpoint /path/to/audiotag_checkpoint.pt \
    --label-dict /path/to/class_labels_indices.csv \
    --input_path ../../../../../data/03_single_event_data/output.json \
    --output_path ../../../../../data/04_audiotagged_data/output.json \
    --sample-rate 48000
```

**Note:** 
- Adjust `--sample-rate` to match your dataset's actual sample rate
- Audio with mismatched sample rates will be automatically resampled

**Example Output Format** (see `data/04_audiotagged_data/example.json`):
- Updates `text_label` to AudioSet ontology categories
- Same JSON structure maintained

### Step 5: Leaf Label Classification

Refine labels to leaf nodes in AudioSet ontology using Qwen3-Omni with confusion sets.

```bash
python code/05_leaf_label_qwen.py \
    --model_path /path/to/Qwen3-Omni-7B \
    --ontology_path ontology/audioset_ontology.json \
    --modified_ontology_path ontology/hive_ontology.json \
    --data_path data/04_audiotagged_data/output.json \
    --output_path data/05_leaf_label_data/output.json \
    --error_set_path data/05_leaf_label_data/error_set.json \
    --batch_size 32
```

**Example Output Format** (see `data/05_leaf_label_data/example.json`):
- Updates `text_label` to refined leaf node categories
- Same JSON structure maintained

### Step 6: Audio Super-Resolution

Upsample audio to 44.1kHz using Apollo super-resolution model for high-purity output.

**Environment Setup:**

Refer to the installation instructions in the [Apollo repository](https://github.com/JusperLee/Apollo)

**Model Download:**

Download the Apollo checkpoint from [Apollo Universal Model Release](https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/tag/uni)
- Place the downloaded checkpoint in `Apollo/` directory

**Setup:**
```bash
cp code/06_superres_apollo.py Apollo/
cd Apollo/
```

**Run:**
```bash
python 06_superres_apollo.py \
    --input_json ../data/05_leaf_label_data/output.json \
    --output_json ../data/06_super_res_data/output.json \
    --output_audio_dir ../data/06_super_res_data/audio \
    --config_path /path/to/apollo_config.yaml \
    --checkpoint_path /path/to/apollo_checkpoint.pth \
    --batch_size 16
```

**Example Output Format** (see `data/06_super_res_data/example.json`):
- Updates `audio_path` to super-resolved audio files (44.1kHz)
- `text_label` remains unchanged
- Final high-purity dataset ready for training
