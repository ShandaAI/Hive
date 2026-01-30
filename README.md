<h1 align="center">A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation</h1>
<p align="center">
  <img src="assert/logo.png" alt="Logo" width="250"/>
</p>
<p align="center">
  <strong>Kai Li<sup>*</sup>, Jintao Cheng<sup>*</sup>, Chang Zeng, Zijun Yan, Helin Wang, Zixiong Su, Bo Zheng, Xiaolin Hu</strong><br>
    <strong>Tsinghua University, Shanda AI, Johns Hopkins University</strong><br>
    <strong><sup>*</sup>Equal contribution</strong><br>
    <strong>Completed during Kai Li's internship at Shanda AI.</strong><br>
  <a href="#">📜 Arxiv 2026</a> | <a href="#">🎶 Demo</a> | <a href="https://huggingface.co/datasets/ShandaAI/Hive">🤗 Dataset</a>
</p>

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=ShandaAI.Hive" alt="访客统计" />
  <img src="https://img.shields.io/github/stars/ShandaAI/Hive?style=social" alt="GitHub stars">
  <img alt="Static Badge" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
</p>

---

## 📌 Table of Contents

- [📄 Abstract](#-abstract)
- [📂 Repository Structure](#-repository-structure)
- [🍯 Hive Dataset](#-hive-dataset)
- [⚙️ Data Collection Pipeline](#️-data-collection-pipeline)
- [📖 Citation](#-citation)
- [⚖️ License](#️-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 📄 Abstract

<p align="justify">
Query-based universal sound separation is fundamental to intelligent auditory systems, aiming to isolate specific sources from unconstrained mixtures. 
Despite recent advances, existing methods continue to suffer from residual interference in complex acoustic scenes. 
This performance limitation stems largely from a data bottleneck: ubiquitous in-the-wild datasets contain weak labels and severe event co-occurrence. 
These flaws induce models to learn spurious correlations between background noise and target categories instead of robust acoustic features. 
To address this, we propose an automated pipeline that eliminates co-occurrence noise by mining high-purity single-event segments 
from unconstrained recordings and synthesizing mixtures via semantically consistent strategies. 
Utilizing this pipeline, we constructed <i>Hive</i>, a high-quality synthetic dataset comprising 2k hours of audio. 
<a style="color:blue;">Experimental results demonstrate that, despite using only <b>~0.2%</b> of the data scale of million-hour baselines, 
models trained on Hive achieve competitive separation accuracy and perceptual quality.</a> 
Moreover, these models exhibit remarkable zero-shot generalization on out-of-distribution evaluation benchmarks such as MUSDB18-HQ and USS-Bench. 
These findings highlight that <a style="background-color:LightYellow;color:red;">prioritizing supervision purity enables significant data efficiency</a>, 
offering a new paradigm for training robust auditory foundation models with reduced computational costs.
</p>

---

## 📂 Repository Structure

```
.
├── hive_dataset/                           # Hive Dataset generation and curation
│   ├── mix_from_metadata/                  # Generate mixtures from metadata
│   │   ├── mix_from_metadata.py
│   │   └── dataset_paths.json
│   ├── mix_curation/                       # Data curation for mix audio
│   │   ├── mix_data_curation.py
│   │   └── ontology.json
│   ├── README.md                           # Dataset documentation
│   ├── requirements.txt
│   └── LICENSE
├── pipeline/                               # Single-Event Data Collection Pipeline
│   ├── code/                               # Pipeline scripts
│   │   ├── 01_audio_chunking.py
│   │   ├── 02_filter_single_label.py
│   │   ├── 03_filter_single_event_qwen.py
│   │   ├── 04_audioset_label_audiotag.py
│   │   ├── 05_leaf_label_qwen.py
│   │   └── 06_superres_apollo.py
│   ├── data/                               # Pipeline data directories
│   ├── ontology/                           # AudioSet ontologies
│   ├── icefall/                            # AudioTag model repository
│   ├── Apollo/                             # Apollo model repository
│   ├── requirements.txt                    # Pipeline dependencies
│   └── README.md                           # Pipeline documentation
├── LICENCE                                 # MIT License
└── README.md                              
```

---

## 🍯 Hive Dataset

**Hive** is a high-quality synthetic dataset with 2,442 hours of raw audio and 19.6M mixtures for Universal Sound Separation.

**Features:**
- 283 sound categories from AudioSet ontology
- Semantically consistent mixing logic
- 44.1kHz sample rate

Please refer to [`hive_dataset/`](hive_dataset/) for details

---

## ⚙️ Data Collection Pipeline

An automated 6-step pipeline for mining high-purity single-event audio from weakly-labeled sources.

**Pipeline Stages:**
1. Audio Chunking - Split long audio into segments
2. Single Label Filtering - Remove multi-label samples
3. Single Event Filtering - Verify acoustic purity with Qwen3-Omni
4. AudioSet Label Tagging - Assign ontology labels with AudioTag
5. Leaf Label Classification - Refine to leaf nodes with Qwen3-Omni
6. Audio Super-Resolution - Upsample to 44.1kHz with Apollo

Please refer to [`pipeline/`](pipeline/) for details

---

## 📖 Citation

If you use this code or the Hive Dataset, please cite:

```bibtex

```

---

## ⚖️ License

### Project License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

### Model Licenses

- **Qwen3-Omni**: Apache 2.0
- **AudioTag**: Apache 2.0
- **Apollo**: Check model repository for specific license

---

## 🙏 Acknowledgments

The Hive dataset is a collaborative achievement built upon the foundation of the open-source audio community. We extend our deepest gratitude to the researchers and organizations who curated the twelve foundational datasets. Their work provides the essential long-tailed acoustic space for advancing **Universal Sound Separation**.

### 🏛️ Foundational Data Sources

We gratefully acknowledge the following core datasets which provided the majority of our high-fidelity clips:

- **BBC Sound Effects** (369,603 clips, 1,020.62h) - Professional-grade recordings with broadcast-level fidelity under Remix License
- **AudioSet** (326,890 clips, 896.61h) - Large-scale benchmark from YouTube under CC BY (Google)
- **VGGSound** (115,191 clips, 319.10h) - Real-world acoustic diversity under CC BY 4.0 (University of Oxford)
- **FreeSound** (17,451 clips, 46.90h) - Rich crowdsourced soundscapes under CC0/BY/BY-NC (MTG-UPF)

### 🎯 Specialized Domain Contributors

Our sincere thanks go to the following datasets for providing the raw source audio that forms the specialized domains of the **Hive Dataset**:

**Music & Speech:**
- **MUSIC21** (32,701 clips, 90.28h) - Solo and ensemble instruments for harmonic structure modeling
- **Voicebank-DEMAND** (12,376 clips, 9.94h) - Clean speech signals under CC BY 4.0
- **FSD50K** (636 clips, 0.80h) - Finely annotated subset based on AudioSet ontology

**Environmental & Events:**
- **ClothoV2** (14,759 clips, 38.19h) - Audio captioning dataset with rich temporal evolution
- **AVE** (3,054 clips, 6.91h) - Audio-visual event localization under CC BY-NC-SA
- **SoundBible** (2,501 clips, 5.78h) - Curated short clips under CC BY 4.0
- **DCASE** (1,969 clips, 5.46h) - Acoustic scene detection challenges
- **ESC50** (1,433 clips, 1.99h) - Environmental sound classification benchmark under CC BY-NC 3.0

### ⚖️ License & Ethical Compliance

All source data were processed in strict accordance with their respective licenses (e.g., CC BY, CC0, Remix License). An automated data collection pipeline was employed to ensure that only semantically aligned and single-label pure segments were extracted, respecting the original intent of the data contributors while enhancing their utility for sound separation tasks.

**Important Note**: This repository releases only the **metadata** (JSON files containing mixing parameters and source references) for reproducibility. We do **not** redistribute the original audio files from the source datasets. Users must independently download and prepare the source datasets according to their respective licenses and terms of use.

*We thank all original contributors for their invaluable service to the scientific community.*
