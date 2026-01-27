<h1 align="center">Hive Dataset</h1>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/Paper-Arxiv%202026-red?logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface&logoColor=white"></a>
</p>

<p align="center">
  <strong>High-Quality Synthetic Dataset for Universal Sound Separation</strong>
</p>

---

## 📌 Table of Contents

- [💡 Highlights](#-highlights)
- [📋 Overview](#-overview)
  - [Dataset Scale](#dataset-scale)
- [📂 Repository Structure](#-repository-structure)
- [🛠️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
  - [📥 Download Metadata](#-download-metadata)
  - [💾 Prepare Source Datasets](#-prepare-source-datasets)
  - [✨ Data Curation](#-data-curation)

---

## 💡 Highlights

* **Purity over Scale**: 2.4k hours achieving competitive performance with million-hour baselines (~0.2% data scale)
* **Single-label Clean Supervision**: Rigorous semantic-acoustic alignment eliminating co-occurrence noise
* **Semantically Consistent Mixing**: Logic-based co-occurrence matrix ensuring realistic acoustic scenes

---

## 📋 Overview

**Hive** is a high-quality synthetic dataset designed for Universal Sound Separation (USS). Unlike traditional methods relying on weakly-labeled in-the-wild data, Hive leverages an automated data collection pipeline to mine high-purity single-event segments from complex acoustic environments and synthesizes mixtures with semantically consistent constraints.

### Dataset Scale

| Metric | Value |
|--------|-------|
| **Training Set Raw Audio** | 2,442 hours |
| **Val & Test Set Raw Audio** | 292 hours |
| **Mixed Samples** | 19.6M mixtures |
| **Total Mixed Duration** | ~22.4k hours |
| **Label Categories** | 283 classes |
| **Sample Rate** | 44.1 kHz |
| **Training Sample Duration** | 4 seconds |
| **Test Sample Duration** | 10 seconds |

**Dataset Split:**
- Training: 17.5M samples
- Validation: 1.75M samples  
- Test: 350k samples

---

## 📂 Repository Structure

```
Hive Dataset/
├── mix_from_metadata/
│   ├── mix_from_metadata.py        # Generate mixtures and sources from metadata
│   └── dataset_paths.json          # Source dataset path configuration
├── mix_curation/
│   ├── mix_data_curation.py        # Data Curation (optional)
│   └── ontology.json               # Hive ontology tree (283 classes)
├── README.md                       
└── requirements.txt                # Python dependencies
```

---

## 🛠️ Installation

```bash
conda create -n hive python=3.10
conda activate hive
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 📥 Download Metadata

Download the Hive Dataset metadata from:

🤗 [Hive Dataset Metadata Download](https://huggingface.co/datasets/ShandaAI/Hive)

After extraction, you will get the metadata files organized by split (train/val/test) and mixture complexity (2-5 sources).

<details>
<summary>📂 <b>Click to expand: Metadata Structure & JSON Schema</b></summary>

<br>

**Metadata Directory Structure:**
```
metadata/
├── train/
│   ├── 2mix/  (train_2mix_metadata_tar0001.json, ...)
│   ├── 3mix/
│   ├── 4mix/
│   └── 5mix/
├── val/
│   ├── 2mix/
│   ├── 3mix/
│   ├── 4mix/
│   └── 5mix/
└── test/
    ├── 2mix/
    ├── 3mix/
    ├── 4mix/
    └── 5mix/
```

**JSON File Structure:**

Each JSON object contains complete generation parameters for a mixture sample:

```json
{
  "mix_id": "sample_00000003",
  "dataset_info": {
    "split": "train",
    "sample_rate": 44100,
    "target_duration": 4.0
  },
  "sources": [
    {
      "source_id": "s1",
      "path": "relative/path/to/audio",
      "label": "Ocean",
      "crop_start_second": 1.396,
      "crop_end_second": 5.396,
      "chunk_start_second": 35.0,
      "chunk_end_second": 45.0,
      "rms_gain": 3.546,
      "snr_db": 0.0,
      "applied_weight": 3.546
    }
  ],
  "mixing_params": {
    "global_normalization_factor": 0.786,
    "final_max_amplitude": 0.95
  }
}
```

**Field Descriptions:**
- `chunk_start/end_second`: Reading interval from the original audio file
- `crop_start/end_second`: Precise cropping position for reproducible random extraction
- `rms_gain`: Energy normalization coefficient (target RMS = 0.1)
- `snr_db`: Signal-to-noise ratio relative to the first source
- `applied_weight`: Final weight for single source = rms_gain × 10^(SNR/20)
- `global_normalization_factor`: Scaling coefficient after audio superposition

</details>

### 💾 Prepare Source Datasets

Hive integrates **12 public datasets** with diverse acoustic characteristics to construct a long-tailed acoustic space supporting universal sound separation. The complete source pool contains **~898,564 clips** totaling **~2,442 hours** after cleaning.

<details>
<summary>📋 <b>Click to expand: Complete Dataset List (12 sources)</b></summary>

<br>

| # | Dataset | Clips | Duration (h) | License | Download |
|---|---------|-------|--------------|---------|----------|
| 1 | **BBC Sound Effects** | 369,603 | 1,020.62 | Remix License | [BBC Sound Effects](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/Zip_files/BBC_Sound_Effects) |
| 2 | **AudioSet** | 326,890 | 896.61 | CC BY | [AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) |
| 3 | **VGGSound** | 115,191 | 319.10 | CC BY 4.0 | [VGGSound](https://huggingface.co/datasets/Loie/VGGSound) |
| 4 | **MUSIC21** | 32,701 | 90.28 | YouTube Standard | [MUSIC21](https://github.com/roudimit/MUSIC_dataset) |
| 5 | **FreeSound** | 17,451 | 46.90 | CC0/BY/BY-NC | [FreeSound](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/Zip_files/FreeSound) |
| 6 | **ClothoV2** | 14,759 | 38.19 | Non-Commercial Research | [ClothoV2](https://zenodo.org/records/4783391) |
| 7 | **Voicebank-DEMAND** | 12,376 | 9.94 | CC BY 4.0 | [Voicebank-DEMAND](https://www.kaggle.com/datasets/jweiqi/voicebank-demand-16k/data) |
| 8 | **AVE** | 3,054 | 6.91 | CC BY-NC-SA | [AVE](https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view) |
| 9 | **SoundBible** | 2,501 | 5.78 | CC BY 4.0 | [SoundBible](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/Zip_files/SoundBible) |
| 10 | **DCASE** | 1,969 | 5.46 | Academic Use | [DCASE2024_T9](https://zenodo.org/records/11425256) |
| 11 | **ESC50** | 1,433 | 1.99 | CC BY-NC 3.0 | [ESC50](https://github.com/karolpiczak/ESC-50) |
| 12 | **FSD50K** | 636 | 0.80 | Creative Commons | [FSD50K](https://zenodo.org/records/4060432) |
| | **Total** | **898,564** | **2,442.60** | | |

</details>

#### Configuration

Edit `dataset_paths.json` to configure local paths:

```json
{
  "BBC_Sound_Effects": "/path/to/bbc_sound_effects",
  "AudioSet": "/path/to/audioset",
  "VGGSound": "/path/to/vggsound",
  "MUSIC21": "/path/to/music21",
  "FreeSound": "/path/to/freesound",
  "ClothoV2": "/path/to/clothov2",
  "Voicebank_DEMAND": "/path/to/voicebank_demand",
  "AVE": "/path/to/ave",
  "SoundBible": "/path/to/soundbible",
  "DCASE": "/path/to/dcase",
  "ESC50": "/path/to/esc50",
  "FSD50K": "/path/to/fsd50k"
}
```

**Note:** Dataset names in the configuration should match the prefix used in metadata paths. 

**Generate Mixed Audio:**

```bash
python mix_from_metadata/mix_from_metadata.py \
    --metadata_dir /path/to/downloaded/metadata \
    --output_dir ./hive_dataset \
    --dataset_paths dataset_paths.json \
    --num_processes 16
```

**Output Format (WebDataset Compatible):**
- Each tar file contains ~1000 samples
- Each sample includes:
  - `{mix_id}.mix.wav`: Mixed audio
  - `{mix_id}.s1.wav`, `{mix_id}.s2.wav`, ...: Separated sources
  - `{mix_id}.json`: Complete metadata

### ✨ Data Curation

```bash
python mix_curation/mix_data_curation.py \
    --source_dataset_path ./hive_dataset \
    --ontology_path /path/to/ontology.json \
    --filtered_dataset_path ./hive_filtered \
    --api_url http://your-multimodal-api \
    --save_score \
    --max_workers 16
```

**Filtering Mechanism:**
- Construct confusion pools using sibling labels from ontology
- Multimodal model predicts source labels
- Keep top-50% samples by accuracy
